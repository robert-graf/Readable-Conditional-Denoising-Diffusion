from pathlib import Path
import pickle
from BIDS import BIDS_Global_info, BIDS_FILE, NII, to_nii
from BIDS.snapshot2D import Snapshot_Frame, create_snapshot
from joblib import Parallel, delayed
from dataclasses import dataclass

dataset_path = "/DATA/NAS/datasets_processed/SHIP"
# dataset_path = "/DATA/NAS/ongoing_projects/robert/SHIP"
dataset = BIDS_Global_info([dataset_path], parents=["rawdata"])
import pandas as pd
import random
from BIDS.logger.log_file import No_Logger, Log_Type
from nipy.algorithms.registration.affine import Affine
import numpy as np
from BIDS.registration import registrate_nipy, apply_registration_nipy

from tqdm import tqdm

logger = No_Logger()
random.seed(42)
# same space
# Phase Path CT     T2w       T2w
# train path target condition condition ...
d = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "T2star": []}
d2 = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "fat": [], "water": [], "opp": [], "inp": []}
from BIDS.stitching import stitching

buffer_path = Path(dataset_path, "code/vibe2T2w_registration")
if Path(buffer_path).exists():
    reg_buffer = pickle.load(open(buffer_path, "rb"))
else:
    reg_buffer = {}
override_vibe_stitching = False
override_t1w_stitching = False
override_vibe_reg = False
override_snap = False
save_vibe_reg = True


def resample_axial_vibe(  # axial
    T1w_sequ_T1w_2: BIDS_FILE,
    T2w_sequ_T2w_2: BIDS_FILE,
    vibes: tuple[BIDS_FILE, BIDS_FILE, BIDS_FILE, BIDS_FILE],
    t1w: Path,
    t2w: Path,
    vibes_path: tuple[Path, Path, Path, Path],
):
    vibes_path = tuple(vibes_path)
    t1w_nii = T1w_sequ_T1w_2.open_nii()
    t2w_nii = T2w_sequ_T2w_2.open_nii()
    inphase = vibes[-1].open_nii()

    t1w_nii.resample_from_to_(inphase, c_val=0, verbose=False)
    t2w_nii.resample_from_to_(inphase, c_val=0, verbose=False)

    crop = t1w_nii.compute_crop_slice()
    crop = t2w_nii.compute_crop_slice(other_crop=crop)
    try:
        t1w_nii.apply_crop_slice_(crop).save(t1w, verbose=False)
        t2w_nii.apply_crop_slice_(crop).save(t2w)
        for bds, path in zip(vibes, tuple(vibes_path)):
            nii = bds.open_nii()
            nii.apply_crop_slice_(crop).save(path)
    except Exception as e:
        t1w.unlink(missing_ok=True)
        t2w.unlink(missing_ok=True)
        logger.print(list(vibes_path), type=Log_Type.FAIL)
        raise e


def resample_sag_vibe(  # sagittal
    T1w_sequ_T1w_2: BIDS_FILE,
    T2w_sequ_T2w_2: BIDS_FILE,
    vibes: tuple[BIDS_FILE, BIDS_FILE, BIDS_FILE, BIDS_FILE],
    t1w: Path,
    t2w: Path,
    vibes_path: tuple[Path, Path, Path, Path],
):
    if vibes_path[0].exists():
        return
    vibes_path = tuple(vibes_path)
    t1w_nii = T1w_sequ_T1w_2.open_nii()
    t2w_nii = T2w_sequ_T2w_2.open_nii()
    inphase = vibes[-1].open_nii()
    inphase.resample_from_to_(t1w_nii)
    t2w_nii.resample_from_to_(t1w_nii)

    crop = inphase.compute_crop_slice()
    crop = t1w_nii.compute_crop_slice(other_crop=crop)
    crop = t2w_nii.compute_crop_slice(other_crop=crop)
    try:
        t1w_nii.apply_crop_slice_(crop).save(t1w, verbose=False)
        t2w_nii.apply_crop_slice_(crop).save(t2w)
        for bds, path in zip(vibes, tuple(vibes_path)):
            nii = bds.open_nii()
            nii.resample_from_to_(t1w_nii).save(path)
    except Exception as e:
        t1w.unlink(missing_ok=True)
        t2w.unlink(missing_ok=True)
        logger.print(list(vibes_path), type=Log_Type.FAIL)
        raise e


def resample_zoom2d(
    phase,
    p: Path,
    T1w_sequ_T1w_1: BIDS_FILE,
    T1w_sequ_T1w_2: BIDS_FILE,
    T2w_sequ_T2w_1: BIDS_FILE,
    T2w_sequ_T2w_2: BIDS_FILE,
    T2star_sequ_Osteo: BIDS_FILE | None,
    zoom=(1, 1, -1),
    axcode_to=("P", "S", "R"),
):  # sagittal
    f = T1w_sequ_T1w_1, T1w_sequ_T1w_2, T2w_sequ_T2w_1, T2w_sequ_T2w_2, T2star_sequ_Osteo
    f2 = []
    for file in f:
        if file is None:
            f2.append(None)
            continue
        zoom = tuple((z if z != 1.0 else 1) for z in zoom)
        zoom = tuple((z if z != 2.0 else 2) for z in zoom)
        zoom_name = str(zoom[0]).replace(".", "-") + "x" + str(zoom[1]).replace(".", "-") + "-mm"
        out_file = file.get_changed_path("nii.gz", parent="resampled", info={"rec": zoom_name})
        f2.append(out_file)
        if out_file.exists():
            continue
        nii = T1w_sequ_T1w_1.open_nii()
        nii.rescale_and_reorient_(axcode_to, zoom)
        nii.save(out_file, make_parents=True)

    d["Phase"].append(phase)
    d["Phase"].append(phase)
    d["Path"].append(str(p).replace("rawdata", "resampled"))
    d["Path"].append(str(p))
    d["T1w"].append(rest_path(f2[0]))
    d["T1w"].append(rest_path(f2[1]))
    d["T2w"].append(rest_path(f2[2]))
    d["T2w"].append(rest_path(f2[3]))
    d["T2star"].append(None)
    d["T2star"].append(rest_path(f2[4]))


def equal(a: BIDS_FILE, b: BIDS_FILE):
    a = a.open_nii()
    b = b.open_nii()
    if a.shape != b.shape:
        return False
    if a.orientation != b.orientation:
        return False

    return True


def stich_tXw(a: BIDS_FILE, b: BIDS_FILE, name="T2w"):
    t2w_s_q = subj.new_query(flatten=True)
    t2w_s_q.filter("desc", "n4stitched")
    t2w_s_q.filter_format(name)
    stitched = list(t2w_s_q.loop_list())
    if len(stitched) == 1:
        t2w_path = stitched[0]
    else:
        t2w_path = a.get_changed_path("nii.gz", parent="resampled", info={"desc": "n4stitched", "chunk": None})
        t2w_path = BIDS_FILE(t2w_path, a.dataset)
        if not t2w_path.exists() or override_t1w_stitching:
            f = delayed(stitching)(a, b, out=t2w_path, bias_field=False)
            tasks_stitched.append(f)
    return t2w_path


def make_pair(
    t1w_file: BIDS_FILE,
    t2w_file: BIDS_FILE,
    stitched_vibe: dict[str, BIDS_FILE],
    phase,
    p,
    axcode_to=("P", "I", "R"),
    zoom=(-1, -1, -1),
    override=False,
):
    zoom = tuple((z if z != 1.0 else 1) for z in zoom)
    zoom = tuple((z if z != 2.0 else 2) for z in zoom)
    if zoom == (-1, -1, -1):
        zoom_name = str(t2w_file.get("sequ")) + "-vibe"
    else:
        zoom_name = str(zoom[0]).replace(".", "-") + "x" + str(zoom[1]).replace(".", "-") + "-mm"
    ### names ###

    t1w_out = t1w_file.get_changed_path("nii.gz", parent="resampled", info={"res": zoom_name, "chunk": None})
    t2w_out = t2w_file.get_changed_path("nii.gz", parent="resampled", info={"res": zoom_name, "chunk": None})
    stitched_vibe_reg_path: dict[str, Path] = {}
    for key, vibe in stitched_vibe.items():
        stitched_vibe_reg_path[key] = vibe.get_changed_path("nii.gz", parent="resampled", info={"res": zoom_name, "chunk": None})

    d2["Phase"].append(phase)
    d2["Path"].append(str(p).replace("rawdata", "resampled"))
    d2["T1w"].append(rest_path(t1w_out))
    d2["T2w"].append(rest_path(t2w_out))
    try:
        d2["inp"].append(rest_path(stitched_vibe_reg_path["inphase"]))
        d2["opp"].append(rest_path(stitched_vibe_reg_path["outphase"]))
        d2["water"].append(rest_path(stitched_vibe_reg_path["water"]))
        d2["fat"].append(rest_path(stitched_vibe_reg_path["fat"]))
    except Exception as e:
        logger.print_error()
        print(stitched_vibe_reg_path)
        exit()
        raise e

    f = delayed(__make_pair)(t1w_file, t2w_file, stitched_vibe, axcode_to, zoom, t1w_out, t2w_out, zoom_name, override)
    tasks.append(f)


def __make_pair(
    t1w_file: BIDS_FILE,
    t2w_file: BIDS_FILE,
    stitched_vibe: dict[str, BIDS_FILE],
    axcode_to,
    zoom,
    t1w_out: Path,
    t2w_out: Path,
    zoom_name,
    override=False,
):
    print(t1w_file.get("sub"), "    ", zoom, "       ", end="\r")
    if not override:
        if not t1w_file.exists():
            print(t1w_file, "does not exist")
            return
        if not t2w_file.exists():
            print(t1w_file, "does not exist")
            return
        for k, a in stitched_vibe.items():
            if not a.exists():
                print(t1w_file, "does not exist")
                return
        all_vibe = True
        for key, vibe in stitched_vibe.items():
            out_p = vibe.get_changed_path("nii.gz", parent="resampled", info={"res": zoom_name, "chunk": None}, make_parent=False)
            if not out_p.exists():
                all_vibe = False
        snap = Path(stitched_vibe["outphase"].dataset, "resampled", "snapshot")
        if t1w_out.exists() and t2w_out.exists() and all_vibe and snap.exists():
            print(t1w_file, "all exist", end="\r")
            return

    stitched_vibe_reg_path: dict[str, Path] = {}
    try:
        # print(zoom, axcode_to)
        t1w = to_nii(t1w_file).reorient(axcode_to).rescale_(zoom, verbose=True)
        t2w = to_nii(t2w_file).reorient(axcode_to).rescale_(zoom, verbose=True)

        key = str(stitched_vibe["outphase"].get("sub"))
        T = None
        if key in reg_buffer:
            T: Affine = reg_buffer[key]
            if np.abs(T.translation).sum() >= 30:
                T = None
            # if T.translation
        if T is None:
            ## ["slr", "mi", "pmi", "dpmi", "cc", "cr", "crl1"]
            logger.print("REG", type=Log_Type.WARNING)
            try:
                logger.print("REG0.5", type=Log_Type.WARNING)
                out, T, others = registrate_nipy(to_nii(stitched_vibe["outphase"]), t2w, similarity="cc")
                logger.print("REG1", type=Log_Type.WARNING)
            except EOFError as e:
                stitched_vibe["outphase"].file["nii.gz"].unlink()
                t2w_file.file["nii.gz"].unlink()
                logger.print("RAISE", type=Log_Type.FAIL)

                raise e
            off = np.abs(T.translation).sum()
            out, T1, others = registrate_nipy(to_nii(stitched_vibe["outphase"]), t2w, similarity="cr")
            off_new = np.abs(T1.translation).sum()
            logger.print("REG2", type=Log_Type.WARNING)
            if off_new < off:
                T = T1
            out, T1, others = registrate_nipy(to_nii(stitched_vibe["outphase"]), t2w, similarity="mi")
            off_new = np.abs(T1.translation).sum()
            if off_new < off:
                T = T1
            logger.print("REG3", type=Log_Type.WARNING)

            out, T1, others = registrate_nipy(to_nii(stitched_vibe["outphase"]), t2w, similarity="dpmi")
            off_new = np.abs(T1.translation).sum()
            if off_new < off:
                T = T1
            logger.print("REG-end", type=Log_Type.WARNING)
            # other_moving=[stitched_vibe[i] for i in ["inphase", "water", "fat"]
            if np.abs(T.translation).sum() >= 100:
                logger.print("Cant register", stitched_vibe["outphase"], type=Log_Type.FAIL)
                return
            if key not in reg_buffer:
                reg_buffer[key] = T
                pickle.dump(reg_buffer, open(buffer_path, "wb"))
            else:
                T = reg_buffer[key]
        logger.print("CROP HEAD", type=Log_Type.WARNING)
        ### CROP HEAD ###
        out = apply_registration_nipy(to_nii(stitched_vibe["outphase"]), t2w, T)
        crop = out.compute_crop_slice(minimum=20)
        t2w.apply_crop_slice_(crop).clamp(min=0)
        t1w.apply_crop_slice_(crop).clamp(min=0)
        logger.print("apply registration vibe", type=Log_Type.WARNING)
        #################

        for key, vibe in stitched_vibe.items():
            out_p = vibe.get_changed_path("nii.gz", parent="resampled", info={"res": zoom_name, "chunk": None})
            if not out_p.exists() or override_vibe_reg or override:
                try:
                    out = apply_registration_nipy(to_nii(vibe), t2w, T).clamp(min=0)
                except EOFError as e:
                    t2w_file.file["nii.gz"].unlink()
                    for key, vibe in stitched_vibe.items():
                        vibe.file["nii.gz"].unlink()
                    raise e
                out.save(out_p)  # if save_vibe_reg else None
            stitched_vibe_reg_path[key] = out_p
        if not t1w_out.exists() or override_vibe_reg or override:
            t1w.save(t1w_out)
            t2w.save(t2w_out)
        snap = Path(stitched_vibe["outphase"].dataset, "resampled", "snapshot")
        snap.mkdir(exist_ok=True)
        snap = Path(snap, t1w_out.name.replace("nii.gz", "jpg"))

        if not snap.exists() or override_snap or override:
            logger.print("Make SNAP-SHOT", type=Log_Type.WARNING)
            create_snapshot(
                snap,
                frames=[
                    Snapshot_Frame(t1w_out),
                    Snapshot_Frame(t2w_out),
                    Snapshot_Frame(stitched_vibe_reg_path["inphase"]),
                    Snapshot_Frame(stitched_vibe_reg_path["outphase"]),
                    Snapshot_Frame(stitched_vibe_reg_path["water"]),
                    Snapshot_Frame(stitched_vibe_reg_path["fat"]),
                ],
            )
    except Exception as e:
        logger.print_error()
        logger.print(stitched_vibe, type=Log_Type.FAIL)
        if "zlib.error" in str(e):
            for key, vibe in stitched_vibe_reg_path.items():
                vibe.unlink()
            t1w_out.unlink()
            t2w_out.unlink()


@dataclass
class DF2(pd.Series):
    Path: str
    T1w: str
    T2w: str
    fat: str
    water: str
    opp: str
    inp: str


def test(root, subpath):
    root = row.Path
    f = Path(root, subpath)
    if not f.exists():
        f = str(f).replace("rawdata", "derivatives")
    if not Path(f).exists():
        f = str(f).replace("derivatives", "rawdata")
    if not Path(f).exists():
        f = str(f).replace("rawdata", "resampled")
    if not Path(f).exists():
        f = str(f).replace("rawdata", "translated/rawdata")
    if not Path(f).exists():
        f = str(f).replace("rawdata", "translated/derivatives")
    return to_nii(f)


tasks = []
tasks_stitched = []
incomplete = ["10106491", "10002140", "10002011", "10104319", "10000952", "10000688"]
not_same_shape = []
unmatched_vibe = ["10105510", "10105798"]
not_bad_reg = [
    "10001369",
    "10001555",
    "10002686",
    "10004016",
    "10004294",
    "10104690",
    "10104883",
    "10105082",
    "10105268",
    "10105697",
    "10106541",
    "10106747",
    "10106947",
    "10107376",
    "10107478",
    "10107517",
    "10107698",
    "10108332",  # 18
]

# for key in not_bad_reg:
#    if key in reg_buffer:
#        del reg_buffer[key]
#        print("del", key)
#    else:
#        print("no del", key)

brocken = []
for name, row in pd.read_excel(Path(dataset_path, "code/train_paired.xlsx")).iterrows():  # type: ignore
    row: DF2
    path = row.Path
    cmp: list[NII] = []
    try:
        sub = BIDS_FILE(Path(path, row.T1w), path).get("sub")
        if sub in not_bad_reg:
            continue
        nii_a = test(path, row.T1w)
        nii_i = test(path, row.T2w)
        nii_b = test(path, row.fat)
        nii_c = test(path, row.water)
        nii_j = test(path, row.opp)
        nii_d = test(path, row.inp)
        if (
            nii_i.shape != nii_a.shape
            or nii_b.shape != nii_a.shape
            or nii_c.shape != nii_a.shape
            or nii_j.shape != nii_a.shape
            or nii_d.shape != nii_a.shape
        ):
            sub = BIDS_FILE(row.T1w, "").get("sub")
            print(sub, nii_i.shape, nii_b.shape, nii_c.shape, nii_j.shape, nii_d.shape, end="\r")
            brocken.append(sub)

            Path(path, row.T1w).unlink()
            Path(path, row.T2w).unlink()
            Path(path, row.fat).unlink()
            Path(path, row.water).unlink()
            Path(path, row.opp).unlink()
            Path(path, row.inp).unlink()
            continue

    except Exception as e:
        print(e)

print("enumerate_subjects")
not_bad_reg = set(not_bad_reg)
for name, subj in dataset.enumerate_subjects(sort=True):
    override = False
    if name not in not_bad_reg:
        override = False
    if name in brocken:
        override = True
    r1 = round((random.random() * 0.2 + 0.8), 2)
    r2 = round((random.random() * 0.5 + 0.75), 2)
    if r2 == r1:
        r2 += 0.1
        r2 = round(r2)
    r3 = round((random.random() * 0.5 + 0.75), 2)
    if r3 == r1 or r3 == r2:
        r3 += 0.05
        r3 = round(r3)
    r2_b = round((random.random() + 3), 2)

    if name in incomplete:
        # Only one file
        continue
    q = subj.new_query()
    q.flatten()
    q.filter_format(lambda x: x != "vibe")
    q.unflatten()
    split = random.random()
    if split < 0.8:
        phase = "train"
    elif split < 0.85:
        phase = "val"
    else:
        phase = "test"
    dic: dict[str, BIDS_FILE] = {}
    for fam in q.loop_dict(key_addendum=["sequ"]):
        for k, v in fam.items():
            dic[k] = v[0]
    try:
        # T2w_sequ_T2w_Composed = dic["T2w_sequ-T2w-Composed"]
        T2w_sequ_T2w_1 = dic["T2w_sequ-T2w-1"]
        T2w_sequ_T2w_2 = dic["T2w_sequ-T2w-2"]
        # T1w_sequ_T1w_Composed = dic["T1w_sequ-T1w-Composed"]
        T1w_sequ_T1w_1 = dic["T1w_sequ-T1w-1"]
        T1w_sequ_T1w_2 = dic["T1w_sequ-T1w-2"]
    except KeyError:
        print(dic)
        print()
        exit(name)
    try:
        T2star_sequ_Osteo = dic["T2star_sequ-Osteo"]
    except KeyError:
        T2star_sequ_Osteo = None

    p = T2w_sequ_T2w_1.file["nii.gz"].parent.parent

    def rest_path(x: BIDS_FILE | Path | None):
        if x is None:
            return None
        if isinstance(x, BIDS_FILE):
            x = x.file["nii.gz"]
        par = x.parent.name
        return par + "/" + x.name

    if equal(T1w_sequ_T1w_1, T2w_sequ_T2w_1):
        d["Phase"].append(phase)
        d["Path"].append(str(p))
        d["T1w"].append(rest_path(T1w_sequ_T1w_1))
        d["T2w"].append(rest_path(T2w_sequ_T2w_1))
        d["T2star"].append(None)

    else:
        not_same_shape.append(T1w_sequ_T1w_1.get("sub"))
    if equal(T1w_sequ_T1w_2, T2w_sequ_T2w_2):
        d["Phase"].append(phase)
        d["Path"].append(str(p))
        d["T1w"].append(rest_path(T1w_sequ_T1w_2))
        d["T2w"].append(rest_path(T2w_sequ_T2w_2))
        d["T2star"].append(rest_path(T2star_sequ_Osteo))
    else:
        not_same_shape.append(T1w_sequ_T1w_2.get("sub"))
    print(name, len(not_same_shape), end="\r")
    # resample_zoom2d(phase, p, T1w_sequ_T1w_1, T1w_sequ_T1w_2, T2w_sequ_T2w_1, T2w_sequ_T2w_2, T2star_sequ_Osteo)
    #########################################

    # STICH VIBE
    ### Load and stitch VIBE ###
    vibe_q = subj.new_query()
    vibe_q.filter_format("vibe")
    vibe_q.filter("res", lambda x: False, required=False)
    vibe_q.filter("desc", lambda x: "stitched" not in str(x), required=False)
    vibe_q.filter("sequ", lambda x: "stitched" not in str(x), required=False)
    vibes = list(vibe_q.loop_dict(sort=True, key_addendum=["part"]))
    stitched_vibe: dict[str, BIDS_FILE] = {}
    for part in ["inphase", "outphase", "water", "fat"]:
        try:
            vibes_type = [v["vibe_part-" + part][0] for v in vibes]
            if len(vibes_type) == 0:
                break
            out_path = vibes_type[0].get_changed_bids("nii.gz", info={"chunk": None, "sequ": "stitched", "desc": None}, parent="rawdata")
            if not out_path.file["nii.gz"].exists() or override_vibe_stitching:
                f = delayed(stitching)(*vibes_type, out=out_path, bias_field=False)
                tasks_stitched.append(f)
            stitched_vibe[part] = out_path
        except Exception as e:
            print(e)
            print(name, "no VIBE")
            break
    if len(stitched_vibe) != 4:
        continue
    # STICH T2w/T1w
    t1w_stich = stich_tXw(T1w_sequ_T1w_1, T1w_sequ_T1w_2, "T1w")
    t2w_stich = stich_tXw(T2w_sequ_T2w_1, T2w_sequ_T2w_2, "T2w")
    ############################
    make_pair(t1w_stich, t2w_stich, stitched_vibe, phase, p, override=override)
    if phase == "train":
        make_pair(t1w_stich, t2w_stich, stitched_vibe, phase, p, zoom=(r1, r1, -1), override=override)
        make_pair(t1w_stich, t2w_stich, stitched_vibe, phase, p, zoom=(r2, r2, -1), override=override)
        make_pair(t1w_stich, t2w_stich, stitched_vibe, phase, p, zoom=(r3, r3, r2_b), override=override)
    ################################################################################################################################

print(not_same_shape)
df = pd.DataFrame(data=d)
# df.to_excel(Path(dataset_path, "code/train_paired_native.xlsx"))
df2 = pd.DataFrame(data=d2)
# df2.to_excel(Path(dataset_path, "code/train_paired.xlsx"))


Parallel(n_jobs=32, backend="threading")(t for t in tasks_stitched)
Parallel(n_jobs=32, backend="threading")(t for t in tasks)

out: list[pd.Series] = []
out_q: list[float] = []
print("QA")

for name, row in tqdm(df2.iterrows(), total=len(df2)):  # type: ignore
    row: DF2
    path = row.Path
    cmp: list[NII] = []
    try:
        sub = BIDS_FILE(Path(path, row.T1w), path).get("sub")
        if sub in not_bad_reg:
            continue
        a = test(path, row.T1w)
        i = test(path, row.T2w)
        b = test(path, row.fat)
        c = test(path, row.water)
        j = test(path, row.opp)
        d = test(path, row.inp)
        if i.shape != a.shape or b.shape != a.shape or c.shape != a.shape or j.shape != a.shape or d.shape != a.shape:
            print(path, row.T1w)
            print(a.shape, i.shape, b.shape, c.shape, j.shape, d.shape)
            continue

        out.append(row)

    except Exception as e:
        print(e)
df2 = pd.DataFrame(out)
df2.to_excel(Path(dataset_path, "code/train_paired.xlsx"))
