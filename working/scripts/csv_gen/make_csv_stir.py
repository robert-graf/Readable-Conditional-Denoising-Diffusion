from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE, NII, BIDS_Family
from joblib import Parallel, delayed
import numpy as np

dataset_path_in = ["/DATA/NAS/datasets_processed/spine/dataset-DIXCTWS/", "/DATA/NAS/datasets_processed/spine/dataset-wsgan/"]

dataset_path = "/DATA/NAS/datasets_processed/combinde/dataset-stir/"


import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type

logger = Reflection_Logger()
random.seed(42)

file_format = set()
file_last = set()
file_chunk = set()
import shutil


def save(file: BIDS_FILE, path: str | Path, nii: NII | None = None):
    assert not Path(path).exists(), path
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    if nii is None:
        shutil.copyfile(file.file["nii.gz"], path)
    else:
        nii.save(path)
    # shutil.copyfile(file.file["json"], str(path).replace("nii.gz", "json"))


dataset = BIDS_Global_info(dataset_path_in, parents=["rawdata"])
from BIDS.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot
from BIDS.registration import register_native_res


def rescale(fam: BIDS_Family, phase: str, zoom=(1, 1, -1), axcodes_to=("P", "S", "R")):
    fat = fam["T2w_part-fat"][0] if "T2w_part-fat" in fam else None
    files = (fam["T2w_part-water"][0], fam["T2w_part-inphase"][0], fat, fam["T1w"][0])
    d["Phase"].append(phase)
    if zoom[0] == 1:
        out_snp = fam["T2w_part-inphase"][0].get_changed_path(file_type="png", parent="rawdata", format="snp", dataset_path=dataset_path)

        create_snapshot(
            [out_snp, Path(dataset_path, "snapshot_res", out_snp.name)],
            [Snapshot_Frame(f, coronal=True) for f in files if f is not None],
        )
    out = Path()
    for f, key in zip(files, ["water", "T2w", "fat", "T1w"]):
        if f is None:
            d[key].append(None)
            continue
        zoom = tuple((z if z != 1.0 else 1) for z in zoom)
        out = f.get_changed_path(
            parent="resampled",
            info={"res": str(zoom[0]).replace(".", "-") + "x" + str(zoom[1]).replace(".", "-") + "-mm"},
            dataset_path=dataset_path,
        )
        if not out.exists():
            if key == "T1w":
                nii_old: NII = nii
                nii = f.open_nii()
                nii.resample_from_to(nii_old, c_val=0).save(out)
            else:
                nii = f.open_nii().rescale_and_reorient(axcodes_to, voxel_spacing=zoom)
                nii.save(out)
        # Add 4 fils to to csv
        d[key].append(out.name)
    d["Path"].append(out.parent)


d = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "water": [], "fat": []}
counter_l = 0
for name, subject in dataset.enumerate_subjects():
    split = random.random()
    if split < 0.8:
        phase = "train"
    elif split < 0.85:
        phase = "val"
    else:
        phase = "test"

    q = subject.new_query(flatten=True)
    q.filter("filetype", "nii.gz")
    q.unflatten()
    q.filter("part", "water")
    q.filter_format(lambda x: x != "ct")
    chunk_t2: dict[str, BIDS_FILE] = {}
    chunk_t1: dict[str, BIDS_FILE] = {}
    # if counter_l % 100 == 5:
    #    df = pd.DataFrame(data=d)
    #    df.to_excel(Path(dataset_path, "train_paired_t1_t2.xlsx"))
    counter_l += 1

    for f in q.loop_dict(key_addendum=["part"]):
        if "T2w_part-inphase" in f and "T1w" in f:
            out = f["T1w"][0].get_changed_path(
                parent="resampled",
                info={"res": "*x*-mm"},
                dataset_path=dataset_path,
            )
            l = list(out.parent.glob(out.name))
            l = [str(x.name).split("-mm")[0].split("x")[-1].replace("-", ".") for x in l]
            fixed_zoom = [1.0, 0.5, 0.75]
            l = [float(x) for x in l if float(x) not in fixed_zoom]
            if len(l) == 0 or len(l) == 1:
                z1 = round((random.random() * 0.3 + 0.75) * 100) / 100
                z2 = round((random.random() * 0.3 + 0.75) * 100) / 100
            else:
                z1 = l[0]
                z2 = l[1]
            for z in fixed_zoom:
                rescale(f, phase, zoom=(z, z, -1))
            rescale(f, phase, zoom=(z1, z1, -1))
            rescale(f, phase, zoom=(z2, z2, -1))

        # Find T2w and T1w by chunk
        if "T2w_part-inphase" in f:
            fi = f["T2w_part-inphase"][0]

            if "chunk" not in fi.info:
                chunk_t2[fi.info["sequ"]] = fi
                chunk_t1[fi.info["sequ"]] = f["T1w"][0]
            else:
                assert fi.info["chunk"] not in chunk_t2, (fi, chunk_t2[fi.info["chunk"]])
                chunk_t2[fi.info["chunk"]] = fi
        if "T1w" in f:
            fi = f["T1w"][0]
            if "chunk" in fi.info:
                assert fi.info["chunk"] not in chunk_t1, (fi, chunk_t1[fi.info["chunk"]])
                chunk_t1[fi.info["chunk"]] = fi
        print(f.get_key_len())

    for key, t2w_file in chunk_t2.items():
        if key in chunk_t1:
            t1w_file = chunk_t1[key]
            # register T1 to T2 with affine
            t2w_nii = t2w_file.open_nii()
            out = t2w_file.get_changed_path(parent="rawdata", format="T1w", info={"part": None}, dataset_path=dataset_path)
            out_snp = t2w_file.get_changed_path(file_type="png", parent="rawdata", format="snp", dataset_path=dataset_path)
            fam = t2w_file.get_sequence_files(key_addendum=["part"])
            parent = out.parent
            if not out.exists():
                t1w_nii = t1w_file.open_nii()
                if str(t2w_nii) != str(t1w_nii):
                    t1w_nii, _, T, _ = register_native_res(t1w_nii, t2w_nii, optimizer="rigid")
                    t1w_nii.resample_from_to_(t2w_nii)
                    t1w_nii.set_dtype_(t2w_nii.dtype)
                t1w_nii.save(out)
                t2w_nii.save(Path(parent, t2w_file.file["nii.gz"].name))
                water = fam["T2w_part-water"][0]
                save(water, Path(parent, water.file["nii.gz"].name))
                if "T2w_part-fat" in fam:
                    fat = fam["T2w_part-fat"][0]
                    save(fat, Path(parent, fat.file["nii.gz"].name))
                # make Snapshot
                create_snapshot(
                    [out_snp, Path(dataset_path, "snapshot", out_snp.name)],
                    [
                        Snapshot_Frame(t1w_nii, coronal=True),
                        Snapshot_Frame(t2w_nii, coronal=True),
                        Snapshot_Frame(fam["T2w_part-water"][0], coronal=True),
                        Snapshot_Frame(fam["T2w_part-fat"][0], coronal=True) if "T2w_part-fat" in fam else None,  # type: ignore
                    ],
                )
            # Add 4 fils to to csv
            d["Phase"].append(phase)
            d["Path"].append(parent)
            d["T1w"].append(out.name)
            d["T2w"].append(t2w_file.file["nii.gz"].name)
            d["water"].append(fam["T2w_part-water"][0].file["nii.gz"].name)
            d["fat"].append(fam["T2w_part-fat"][0].file["nii.gz"].name if "T2w_part-fat" in fam else None)


df = pd.DataFrame(data=d)
df.to_excel(Path(dataset_path, "train_paired_t1_t2.xlsx"))
