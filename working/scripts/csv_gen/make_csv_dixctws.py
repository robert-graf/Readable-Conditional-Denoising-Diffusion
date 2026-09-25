from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE, NII, BIDS_Family
from joblib import Parallel, delayed
import numpy as np

dataset_path = "/DATA/NAS/datasets_processed/spine/dataset-DIXCTWS/"

import pandas as pd
import random
from BIDS.logger.log_file import Reflection_Logger, Log_Type

logger = Reflection_Logger()
random.seed(42)

file_format = set()
file_last = set()
file_chunk = set()

dataset = BIDS_Global_info([dataset_path], parents=["rawdata"])
from BIDS.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot
from BIDS.registration import register_native_res


def rescale(fam: BIDS_Family, phase: str, zoom=(1, 1, -1), axcodes_to=("P", "S", "R")):
    fat = fam["T2w_rec-fat"][0] if "T2w_rec-fat" in fam else None
    files = (fam["T2w_rec-water"][0], fam["T2w_rec-in"][0], fat, fam["T1w"][0])
    d["Phase"].append(phase)
    d["Path"].append(f"resampled/sub-{files[0].info['sub']}")

    for f, key in zip(files, ["water", "T2w", "fat", "T1w"]):
        if f is None:
            d[key].append(None)
            continue
        out = f.get_changed_path(
            parent="resampled", info={"res": str(zoom[0]).replace(".", "-") + "x" + str(zoom[1]).replace(".", "-") + "-mm"}
        )
        if not out.exists():
            nii = f.open_nii().rescale_and_reorient(axcodes_to, voxel_spacing=zoom)
            nii.save(out)
        # Add 4 fils to to csv
        d[key].append(out.name)


d = {"Phase": [], "Path": [], "T1w": [], "T2w": [], "water": [], "fat": []}
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
    q.filter_format(lambda x: x != "ct")
    chunk_t2: dict[str, BIDS_FILE] = {}
    chunk_t1: dict[str, BIDS_FILE] = {}
    for f in q.loop_dict(key_addendum=["rec"]):
        if "T2w_rec-in" in f and "T1w" in f:
            rescale(f, phase, zoom=(1, 1, -1))
            rescale(f, phase, zoom=(0.5, 0.5, -1))
            rescale(f, phase, zoom=(0.75, 0.75, -1))
        # Find T2w and T1w by chunk
        if "T2w_rec-in" in f:
            fi = f["T2w_rec-in"][0]
            assert fi.info["chunk"] not in chunk_t2, (fi, chunk_t2[fi.info["chunk"]])
            chunk_t2[fi.info["chunk"]] = fi
            continue
        if "T1w" in f:
            fi = f["T1w"][0]
            assert fi.info["chunk"] not in chunk_t1, (fi, chunk_t1[fi.info["chunk"]])
            chunk_t1[fi.info["chunk"]] = fi
        print(f.get_key_len())

    for key, t2w_file in chunk_t2.items():
        if key in chunk_t1:
            t1w_file = chunk_t1[key]
            # register T1 to T2 with affine
            t2w_nii = t2w_file.open_nii()
            out = t2w_file.get_changed_path(parent="rawdata", format="T1w", info={"rec": None})
            out_snp = t2w_file.get_changed_path(file_type="png", parent="rawdata", format="snp")
            fam = t2w_file.get_sequence_files(key_addendum=["rec"])

            if not out.exists():
                t1w_nii, _, T, _ = register_native_res(t1w_file.open_nii(), t2w_nii, optimizer="rigid")
                t1w_nii.resample_from_to_(t2w_nii)
                t1w_nii.set_dtype_(t2w_nii.dtype)
                t1w_nii.save(out)
                # make Snapshot
                create_snapshot(
                    [out_snp, Path(dataset_path, "snapshot", out_snp.name)],
                    [
                        Snapshot_Frame(t1w_nii, coronal=True),
                        Snapshot_Frame(t2w_nii, coronal=True),
                        Snapshot_Frame(fam["T2w_rec-water"][0], coronal=True),
                        Snapshot_Frame(fam["T2w_rec-fat"][0], coronal=True) if "T2w_rec-fat" in fam else None,  # type: ignore
                    ],
                )
            # Add 4 fils to to csv
            d["Phase"].append(phase)
            d["Path"].append(f"rawdata/sub-{t2w_file.info['sub']}")
            d["T1w"].append(out.name)
            d["T2w"].append(t2w_file.file["nii.gz"].name)
            d["water"].append(fam["T2w_rec-water"][0].file["nii.gz"].name)
            d["fat"].append(fam["T2w_rec-fat"][0].file["nii.gz"].name if "T2w_rec-fat" in fam else None)


df = pd.DataFrame(data=d)
df.to_excel(Path(dataset_path, "train_paired_t1_t2.xlsx"))
