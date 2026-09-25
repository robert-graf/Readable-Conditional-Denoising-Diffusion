from pathlib import Path

import numpy as np
import scipy
from TPTBox import BIDS_FILE, BIDS_Global_info, Log_Type, Logger, No_Logger, Subject_Container, to_nii
from TPTBox.stitching import GNC_stitch_T2w

from scripts.stir_paper.inference.VIBE_t2w_2_t1w.run import to_t1w_name
from scripts.stir_paper.test_image_quality import eval_2_images, print_metric_list

log = No_Logger()


def make_stitched_T2w(
    subject: Subject_Container,
    derivative="derivatives_stir",
    stich_SHIP=True,
    _key="T2w",
):
    t2w_s_q = subject.new_query(flatten=True)
    t2w_s_q.filter("desc", "n4stitched")
    t2w_s_q.filter_format(_key)
    stitched = list(t2w_s_q.loop_list())
    out_d: dict[str, BIDS_FILE] = {}
    if len(stitched) == 1:
        # stitched_t2w = to_nii(stitched[0])
        return stitched[0]
    t2w_s_q = subject.new_query(flatten=True)
    t2w_s_q.filter_format(_key)
    stitched = list(t2w_s_q.loop_list())
    if len(stitched) == 1:
        # stitched_t2w = to_nii(stitched[0])
        t2w_path = stitched[0]
    else:
        t2w_q = subject.new_query(flatten=True)
        t2w_q.filter("res", lambda x: False, required=False)
        t2w_q.filter("chunk", lambda x: x in ["BWS", "LWS", "HWS"])
        t2w_q.filter("sequ", lambda x: str(x) not in "Composed", required=False)
        t2w_q.filter_format(_key)
        out_d: dict[str, BIDS_FILE] = {}
        for t2w in t2w_q.loop_list():
            out_d[str(t2w.get("chunk"))] = t2w
        if "HWS" in out_d:
            stitched_t2w = GNC_stitch_T2w(to_nii(out_d["HWS"]), to_nii(out_d["BWS"]), to_nii(out_d["LWS"]))
            t2w_path = out_d["HWS"].get_changed_path("nii.gz", parent=derivative, info={"desc": "n4stitched", "chunk": None})
            stitched_t2w.save(t2w_path)
            t2w_path = BIDS_FILE(t2w_path, out_d["HWS"].dataset)
        else:
            # SHIP
            t2w_q = subject.new_query(flatten=True)
            # t2w_q.filter("sequ", lambda x: "Composed" in x or "stitched" in x)
            t2w_q.filter("res", lambda x: False, required=False)
            t2w_q.filter("sequ", lambda x: f"{_key}-2" in str(x) or "stitched" in str(x))
            t2w_q.filter_format(_key)
            stitched = list(t2w_q.loop_list())
            assert len(stitched) == 1, stitched
            stitched_t2w = to_nii(stitched[0])
            if stitched_t2w.max() == 4095:
                arr = stitched_t2w.get_array()
                arr[arr >= 4000] = 0
                stitched_t2w.set_array_(arr)
            assert stitched_t2w.max() <= 4005, (stitched, stitched_t2w.max())

            t2w_path = stitched[0]
    return t2w_path  # , stitched_t2w


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
        out_path = vibes_type[0].get_changed_bids(
            "nii.gz",
            info={"chunk": None, "sequ": "stitched", "desc": None},
            parent="rawdata",
        )
        stitched_vibe[part] = out_path
    return stitched_vibe


def test_stir(
    subject: Subject_Container,
    native_mode=False,
    derivative="derivatives_stir",
    metric_list=None,
    dataset=None,
):
    if metric_list is None:
        metric_list = {}
    try:
        if native_mode:
            derivative += "_native"
        log.print("Start", subject.name, type=Log_Type.Yellow)

        ### Load and stitch T2w ###
        t2w_path = make_stitched_T2w(subject, derivative)
        seg = t2w_path.get_changed_path(format="msk", info={"seg": "spine", "mod": "T2w"})
        if not seg.exists():
            log.print("seg", seg, "does not exits", Log_Type.FAIL)
            return

        t1w_path_org = make_stitched_T2w(subject, derivative, _key="T1w")
        ### STIR PATH ###
        # stir_path = to_stir_name(t2w_path, parent=derivative,dataset)
        ### Load and stitch VIBE ###
        stitched_vibe = make_and_load_stitched_vibe(subject)
        if stitched_vibe is None:
            return
        key = str(stitched_vibe["outphase"].get("sub"))
        t2w_cropped_path = t2w_path.get_changed_path("nii.gz", parent=derivative, info={"res": "T1w", "chunk": None})
        #################
        stitched_vibe_reg: dict[str, Path] = {}
        for key, vibe in stitched_vibe.items():
            out_p = vibe.get_changed_path("nii.gz", parent=derivative, info={"res": "T1w", "chunk": None})
            stitched_vibe_reg[key] = out_p
        ### Make T1w from T2w and VIBE ###
        t1w_path = to_t1w_name(t2w_path, parent=derivative, out_dataset_path=dataset)
        # log.print("T2w:", t2w_cropped_path, type=Log_Type.Yellow)
        # log.print("Vibe", stitched_vibe_reg, type=Log_Type.Yellow)
        log.print("T1w_pred", t1w_path, type=Log_Type.Yellow)
        # log.print("stir_path", stir_path, type=Log_Type.Yellow)
        log.print("T1w_real", t1w_path_org, type=Log_Type.Yellow)
        # log.print(t2w_path, type=Log_Type.Yellow)
        # log.print(t2w_path, type=Log_Type.Yellow)

        b = to_nii(t1w_path)
        a = to_nii(t1w_path_org).resample_from_to_(b)
        # a.smooth_gaussian_(1)
        # b.smooth_gaussian_(1)
        a.match_histograms_(b)
        # a.save(t1w_path_org.get_changed_path(parent=derivative, info={"sequ": "T2w-2"}))

        mask = to_nii(seg, True).resample_from_to_(b).clamp(0, 1)
        eval_2_images(b * mask, a * mask, metric_list)

        print_metric_list(metric_list)

    except EOFError:
        pass
    except FileNotFoundError as e:
        log.print(str(e), Log_Type.FAIL)
        # raise e
    except Exception:
        # raise e

        log.print_error()


import pandas as pd


def run_on_test_and_val_ship():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    log.print("run_on_test_and_val", type=Log_Type.Yellow)
    df = pd.read_excel("/DATA/NAS/datasets_processed/SHIP/code/train_paired.xlsx")
    names = [str(p).rsplit("-", maxsplit=1)[-1] for p in list(df.Path)]
    phases = list(df.Phase)
    name_2_phase = {n: p for n, p in zip(names, phases, strict=False)}
    log.print("parse SHIP", type=Log_Type.Yellow)
    glob = BIDS_Global_info(datasets=["/DATA/NAS/datasets_processed/SHIP/"], parents=["rawdata/10105"])
    delay = []
    metric_list = {}
    for name, sub in glob.enumerate_subjects():
        # if name_2_phase.get(name, "train") != "train":
        if name_2_phase.get(name, "val") != "val":
            test_stir(sub, native_mode=True, metric_list=metric_list)
            # d = delayed(test_stir)(
            #    sub,
            #    native_mode=True,
            # )
            # delay.append(d)
            # break
            # if len(metric_list["L1"]) == 10:
            #    break
    # Parallel(n_jobs=1, backend="threading")(a for a in delay)


if __name__ == "__main__":
    # run_on_test_and_val_ship()

    model_names = [
        (None, "cGAN_02b_SHIP_vibe_T2w_to_T1w_2D", "cycleGAN"),
        (None, "CUT_02b_SHIP_T2w_to_T1w_2D", "Pix2Pix2D"),  # T2w -> T1w
        (None, "CUT_02c_SHIP_T2w_to_T1w", "Pix2Pix"),
        # T2w, T1dixon -> T1w 2D
        (None, "CUT_02b_SHIP_vibe_T2w_to_T1w_2D", "Pix2Pix2D"),
        (None, "CUT_02b_SHIP_vibe_T2w_to_T1w", "Pix2Pix"),  # T2w, T1dixon -> T1w
        # TODO add SHIP T2w to T1w 2D # T2w -> T1w 2D
        # TODO add cycleGAN
    ]

    log.print("run_on_test_and_val", type=Log_Type.Yellow)
    df = pd.read_excel("/DATA/NAS/datasets_processed/SHIP/code/train_paired.xlsx")
    names = [str(p).rsplit("-", maxsplit=1)[-1] for p in list(df.Path)]
    phases = list(df.Phase)
    name_2_phase = {n: p for n, p in zip(names, phases, strict=False)}
    log.print("parse SHIP", type=Log_Type.Yellow)

    log2 = Logger(Path(__file__).parent, "evaluation-T1.json", default_verbose=True)

    summery = {}
    for model_name, model_name_T1w, model_type in model_names:
        name = model_name if model_name is not None else ""
        if isinstance(model_name_T1w, str):
            name += "____" + f"{model_name_T1w}"

        dataset = Path(f"/DATA/NAS/ongoing_projects/robert/paper/paper_T1w/{name}")
        glob = BIDS_Global_info(
            datasets=[Path("/DATA/NAS/datasets_processed/SHIP/"), dataset],
            parents=["rawdata", "derivatives_stir_native"],
        )
        delay = []
        metric_list = {}

        for name, sub in glob.enumerate_subjects(sort=True):
            # if name_2_phase.get(name, "train") != "train":
            if name_2_phase.get(name, "val") == "test":
                print(sub)
                test_stir(sub, native_mode=True, metric_list=metric_list, dataset=dataset)
                # exit()
                # d = delayed(test_stir)(
                #    sub,
                #    native_mode=True,
                # )
                # delay.append(d)
                # break
                # if "L1" in metric_list and len(metric_list["L1"]) == 4:
                #    break
        log2.print(model_name_T1w)
        print_metric_list(metric_list, log2)
        summery[model_name_T1w] = metric_list.copy()
        # Parallel(n_jobs=1, backend="threading")(a for a in delay)

    for k, v in summery.items():
        print(k)
        print_metric_list(v)
        log2.print(model_name_T1w)
        print_metric_list(metric_list, log2)
        summery[model_name_T1w] = metric_list.copy()
        # Parallel(n_jobs=1, backend="threading")(a for a in delay)

    for k, v in summery.items():
        print(k)
        print_metric_list(v)
        keys = list(v.keys())
    for met in keys:
        for k, v in summery.items():
            for k2, v2 in summery.items():
                if k2 != k:
                    res = scipy.stats.wilcoxon(np.array(v[met]) - np.array(v2[met]))
                    print(met, k, k2, round(res.statistic, 4), round(res.pvalue, 4))
