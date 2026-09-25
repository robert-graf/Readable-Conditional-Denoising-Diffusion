from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE
from joblib import Parallel, delayed

dataset_path = "/DATA/NAS/datasets_processed/SHIP"
dataset = BIDS_Global_info([dataset_path], parents=["rawdata"])
import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type

logger = Reflection_Logger()
random.seed(42)
# same space
# Phase Path CT     T2w       T2w
# train path target condition condition ...
d = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "T2star": []}
d2 = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "fat": [], "water": [], "opp": [], "in": []}


def resample_axial_t1dixon(  # axial
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


def resample_sag_t1dixon(  # sagittal
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


def resample_iso(
    phase,
    p: Path,
    T1w_sequ_T1w_1: BIDS_FILE,
    T1w_sequ_T1w_2: BIDS_FILE,
    T2w_sequ_T2w_1: BIDS_FILE,
    T2w_sequ_T2w_2: BIDS_FILE,
    T2star_sequ_Osteo: BIDS_FILE | None,
):  # sagittal
    f = T1w_sequ_T1w_1, T1w_sequ_T1w_2, T2w_sequ_T2w_1, T2w_sequ_T2w_2, T2star_sequ_Osteo
    f2 = []
    for file in f:
        if file is None:
            f2.append(None)
            continue
        out_file = file.get_changed_path("nii.gz", parent="resampled", info={"rec": f"1x1-mm"})
        f2.append(out_file)
        if out_file.exists():
            continue
        nii = T1w_sequ_T1w_1.open_nii()
        nii.rescale_and_reorient_(("P", "I", "L"), (1, 1, -1))
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


tasks = []
incomplete = ["10106491", "10002140", "10002011", "10104319", "10000952", "10000688"]
unmatched_vibe = ["10105510", "10105798"]
for name, subj in dataset.enumerate_subjects():
    if name in incomplete:
        # Only one file
        continue
    q = subj.new_query()
    q.flatten()
    q.filter_format(lambda x: x != "t1dixon")
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

    d["Phase"].append(phase)
    d["Phase"].append(phase)
    d["Path"].append(str(p))
    d["Path"].append(str(p))
    d["T1w"].append(rest_path(T1w_sequ_T1w_1))
    d["T1w"].append(rest_path(T1w_sequ_T1w_2))
    d["T2w"].append(rest_path(T2w_sequ_T2w_1))
    d["T2w"].append(rest_path(T2w_sequ_T2w_2))
    d["T2star"].append(None)
    d["T2star"].append(rest_path(T2star_sequ_Osteo))
    resample_iso(phase, p, T1w_sequ_T1w_1, T1w_sequ_T1w_2, T2w_sequ_T2w_1, T2w_sequ_T2w_2, T2star_sequ_Osteo)
    #########################################
    q = subj.new_query()
    q.flatten()
    q.filter_format("t1dixon")
    q.unflatten()
    dic: dict[str, BIDS_FILE] = {}

    def filter_rec(f: list[list[BIDS_FILE]], value: str):
        for cf in f:
            for bf in cf:
                if bf.info["rec"] == value:
                    return bf
        raise ValueError(f)

    if name in unmatched_vibe:
        continue
    for fam in q.loop_dict():
        try:
            w = filter_rec(fam.values(), "water")
            f = filter_rec(fam.values(), "fat")
            o = filter_rec(fam.values(), "opp")
            i = filter_rec(fam.values(), "in")
        except ValueError:
            continue

        vibes = (w, f, o, i)
        vibes_new_path = tuple(
            x.get_changed_path("nii.gz", parent="resampled", info={"desc": f"{i.format}-{i.info['sequ']}"}) for x in vibes
        )
        t1w = T1w_sequ_T1w_2.get_changed_path("nii.gz", parent="resampled", info={"desc": f"{i.format}-{i.info['sequ']}"})
        t2w = T2w_sequ_T2w_2.get_changed_path("nii.gz", parent="resampled", info={"desc": f"{i.format}-{i.info['sequ']}"})

        # if not i_p.exists():  # requires to run twice
        d2["Phase"].append(phase)
        d2["Path"].append(str(p).replace("rawdata/", "resampled/"))
        (w, f, o, i) = vibes_new_path
        d2["water"].append(rest_path(w))
        d2["fat"].append(rest_path(f))
        d2["opp"].append(rest_path(o))
        d2["in"].append(rest_path(i))
        d2["T1w"].append(rest_path(t1w))
        d2["T2w"].append(rest_path(t2w))

        f = delayed(resample_sag_t1dixon)(T1w_sequ_T1w_2, T2w_sequ_T2w_2, vibes, t1w, t2w, vibes_new_path)
        tasks.append(f)
    ################################################################################################################################
df = pd.DataFrame(data=d)
df.to_excel(Path(dataset_path, "train_paired_native.xlsx"))
df2 = pd.DataFrame(data=d2)
df2.to_excel(Path(dataset_path, "train_paired_resampled_t1dixon.xlsx"))
# delayed(sub_log_call_func)(
Parallel(n_jobs=32, backend="threading")(t for t in tasks)
