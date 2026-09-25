from pathlib import Path
import pickle
from BIDS import Subject_Container, BIDS_Family, BIDS_FILE, to_nii, NII
from BIDS.stitching import stitching, GNC_stitch_T2w
from BIDS import No_Logger, Log_Type
from nipy.algorithms.registration.affine import Affine
import numpy as np
from BIDS.registration import registrate_nipy, apply_registration_nipy

log = No_Logger()


def make_and_load_stitched_vibe(subject: Subject_Container, override_vibe_stitching=False, verbose=False):
    """Loads Stitched Vibes from a Subject. If they not exist they are made"""
    ### Load and stitch VIBE ###
    vibe_q = subject.new_query()
    vibe_q.filter_format("vibe")
    vibe_q.filter("res", lambda x: False, required=False)
    vibe_q.filter("desc", lambda x: "stitched" not in str(x), required=False)
    vibe_q.filter("sequ", lambda x: "stitched" not in str(x), required=False)
    vibes = list(vibe_q.loop_dict(sort=True, key_addendum=["part"]))
    stitched_vibe: dict[str, BIDS_FILE] = {}
    for part in ["inphase", "outphase", "water", "fat"]:
        vibes_type = [v["vibe_part-" + part][0] for v in vibes]
        if len(vibes_type) == 0:
            return
        out_path = vibes_type[0].get_changed_bids("nii.gz", info={"chunk": None, "sequ": "stitched", "desc": None}, parent="rawdata")
        if not out_path.file["nii.gz"].exists() or override_vibe_stitching:
            stitching(*vibes_type, out=out_path, bias_field=False, verbose=verbose)
        stitched_vibe[part] = out_path
    return stitched_vibe


def make_stitched_T2w(subject: Subject_Container, derivative="derivatives_stir", stich_SHIP=True, __key="T2w"):
    t2w_s_q = subject.new_query(flatten=True)
    t2w_s_q.filter("desc", "n4stitched")
    t2w_s_q.filter_format(__key)
    stitched = list(t2w_s_q.loop_list())
    if len(stitched) == 1:
        # stitched_t2w = to_nii(stitched[0])
        t2w_path = stitched[0]
    else:
        t2w_q = subject.new_query(flatten=True)
        t2w_q.filter("res", lambda x: False, required=False)
        t2w_q.filter("chunk", lambda x: x in ["BWS", "LWS", "HWS"])
        t2w_q.filter("sequ", lambda x: str(x) not in "Composed", required=False)
        t2w_q.filter_format(__key)
        out_d: dict[str, BIDS_FILE] = {}
        for t2w in t2w_q.loop_list():
            out_d[str(t2w.get("chunk"))] = t2w
        if "HWS" in out_d:
            stitched_t2w = GNC_stitch_T2w(to_nii(out_d["HWS"]), to_nii(out_d["BWS"]), to_nii(out_d["LWS"]))
            t2w_path = out_d["HWS"].get_changed_path("nii.gz", parent=derivative, info={"desc": "n4stitched", "chunk": None})
            stitched_t2w.save(t2w_path)
            t2w_path = BIDS_FILE(t2w_path, out_d["HWS"].dataset)
        else:
            # if stich_SHIP:
            #    raise NotImplementedError()
            # SHIP
            t2w_q = subject.new_query(flatten=True)
            # t2w_q.filter("sequ", lambda x: "Composed" in x or "stitched" in x)
            t2w_q.filter("res", lambda x: False, required=False)
            t2w_q.filter("sequ", lambda x: f"{__key}-2" in str(x) or "stitched" in str(x))
            t2w_q.filter_format(__key)
            stitched = list(t2w_q.loop_list())
            assert len(stitched) == 1, stitched
            stitched_t2w = to_nii(stitched[0])
            if stitched_t2w.max() == 4095:
                arr = stitched_t2w.get_array()
                arr[arr >= 4000] = 0
                stitched_t2w.set_array_(arr)
            assert stitched_t2w.max() <= 4000, (stitched, stitched_t2w.max())

            t2w_path = stitched[0]
    return t2w_path  # , stitched_t2w


def make_stitched_T1w(subject: Subject_Container, derivative="derivatives_stir", **args):
    print("T1w", subject.name)
    return make_stitched_T2w(subject, derivative, __key="T1w", **args)


reg_buffer = None


def __compute_T(
    transform: None | Affine,
    key: str,
    stitched_vibe,
    stitched_t2w: NII,
    buffer_path="/DATA/NAS/datasets_processed/NAKO/notes/vibe2T2w_registration.pkl",
):
    global reg_buffer
    if reg_buffer is None:
        if Path(buffer_path).exists():
            reg_buffer = pickle.load(open(buffer_path, "rb"))
        else:
            reg_buffer = {}
    if transform is None:
        if key in reg_buffer:
            transform_buff: Affine = reg_buffer[key]
            if np.abs(transform_buff.translation).sum() >= 100:
                transform = None
            else:
                transform = transform_buff
        if transform is None:
            out, transform, others = registrate_nipy(
                to_nii(stitched_vibe["outphase"]), stitched_t2w
            )  # other_moving=[stitched_vibe[i] for i in ["inphase", "water", "fat"]
            if np.abs(transform.translation).sum() >= 100:
                out, transform, others = registrate_nipy(to_nii(stitched_vibe["outphase"]), stitched_t2w, similarity="pmi")
            if np.abs(transform.translation).sum() >= 100:
                log.print("Cant register", stitched_vibe["outphase"], type=Log_Type.FAIL)
                return transform
            reg_buffer[key] = transform
            pickle.dump(reg_buffer, open(buffer_path, "wb"))
        log.print(transform)

    return transform


def reg_VIBE_2_TXw(
    stitched_vibe: dict[str, BIDS_FILE],
    tXw_path: BIDS_FILE,
    res="iso",
    scale=(1, 1, -1),
    derivative="derivatives_stir",
    override_vibe_reg=False,
    save_vibe_reg=True,
):
    T = None
    t2w_cropped_path = tXw_path.get_changed_path("nii.gz", parent=derivative, info={"res": res, "chunk": None})
    stitched_t2w = to_nii(tXw_path).reorient(("P", "I", "R"))
    if not t2w_cropped_path.exists() or override_vibe_reg:
        stitched_t2w.rescale_(scale)
        key = str(stitched_vibe["outphase"].get("sub"))
        ### CROP HEAD ###
        T = __compute_T(
            T,
            key,
            stitched_vibe,
            stitched_t2w,
        )
        out = apply_registration_nipy(to_nii(stitched_vibe["outphase"]), stitched_t2w, T)
        crop = out.compute_crop_slice(minimum=20)
        stitched_t2w.apply_crop_slice_(crop)
        stitched_t2w_cropped = stitched_t2w
        stitched_t2w.save(t2w_cropped_path)
    else:
        stitched_t2w_cropped = to_nii(t2w_cropped_path)

    stitched_vibe_reg: dict[str, NII] = {}
    for key, vibe in stitched_vibe.items():
        out_p = vibe.get_changed_path("nii.gz", parent=derivative, info={"res": "T1w", "chunk": None})
        if out_p.exists() and not override_vibe_reg:
            out = to_nii(out_p)
        else:
            T = __compute_T(T, key, stitched_vibe, stitched_t2w)
            out = apply_registration_nipy(to_nii(vibe), stitched_t2w_cropped, T)
            out.save(out_p) if save_vibe_reg else None
        stitched_vibe_reg[key] = out

    return stitched_vibe_reg, stitched_t2w_cropped
