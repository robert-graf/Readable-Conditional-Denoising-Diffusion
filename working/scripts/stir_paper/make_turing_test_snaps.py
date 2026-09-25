from pathlib import Path
from BIDS.snapshot2D import create_snapshot, Snapshot_Frame
import pandas as pd
import random
import secrets


# 0. Randomly select files for all datasets and save them in a sheet
# 1. Crop away negative space
# 2. resample to (1.1,1.1, *)
# 3. Random Crop or a look up corp if only a certain region is relevant
# (The real data set if pathological but the SHIP/NAKO are really non-health, so we picked only closed; the SHIP does not contain heads)
# 4. Make a snapshot, give it an id and save it in an excel with a ground truth mask
T1w_test = True
if T1w_test:
    random.seed(42)
else:
    random.seed(1337)


### PATHS ###
from dataclasses import dataclass

file_limit = 30


@dataclass()
class DATASET:
    key: str
    path: Path
    snp_path: Path
    rawdata: str = "rawdata"
    file_limit: int = file_limit
    t1w_real: bool = False
    stir_real: bool = False

    def get_sheet_name(self):
        return "raw_" + self.key


datasets: list[DATASET] = []
# DIXCTWS
dixctws_path = Path("/DATA/NAS/datasets_processed/MRI_spine/dataset-DIXCTWS")
dixctws_snp_path = Path(dixctws_path, "snapshot")
datasets.append(DATASET("dixctws", dixctws_path, dixctws_snp_path, stir_real=True, t1w_real=True))

# WSGAN
wsgan_path = Path("/DATA/NAS/datasets_processed/MRI_spine/dataset-wsgan")
wsgan_snp_path = Path(wsgan_path, "snapshot")
datasets.append(DATASET("WSGAN", wsgan_path, wsgan_snp_path, stir_real=True, t1w_real=True))

# NAKO
nako_path = Path("/DATA/NAS/datasets_processed/NAKO/MRT/")
snp_path = Path(nako_path, "derivatives_stir/snapshot_selected")
datasets.append(DATASET("NAKO", nako_path, snp_path, rawdata="derivatives_stir"))

# SHIP
ship_path = Path("/DATA/NAS/datasets_processed/SHIP/")
snp_path = Path(ship_path, "derivatives_stir_native/snapshot_native_selected")
datasets.append(DATASET("SHIP_FAKE", ship_path, snp_path, rawdata="derivatives_stir_native"))
# SHIP
ship_path = Path("/DATA/NAS/datasets_processed/SHIP/")
snp_path = Path(ship_path, "derivatives_stir_native/snapshot_native_selected")
datasets.append(DATASET("SHIP_REAL", ship_path, snp_path, rawdata="derivatives_stir_native", t1w_real=True))


if T1w_test:
    excel_path = Path(Path(__file__).parent, "turing_test_solution_T1w.xlsx")
    out_folder = Path("/DATA/NAS/ongoing_projects/robert/paper/02_stir/t1w_turing")
    datasets.pop(0)
else:
    excel_path = Path(Path(__file__).parent, "turing_test_solution_STIR.xlsx")
    out_folder = Path("/DATA/NAS/ongoing_projects/robert/paper/02_stir/stir_turing")
    datasets.pop()


out_folder.mkdir(exist_ok=True)
##############

##### Step 0: Randomly select files for all datasets and save them in a sheet #####
# We give every file a random priority and take the first n, witch are not excluded.

data: dict[str, pd.DataFrame] = {}
if excel_path.exists():
    try:
        p = excel_path.rename(Path(excel_path.parent, "old.xlsx"))
        xls_org = pd.ExcelFile(p)
        for sn in xls_org.sheet_names:
            data[str(sn)] = pd.read_excel(xls_org, sn)
    except ValueError:
        excel_path.unlink()
import time

random.seed(time.time_ns())

for ds in datasets:  # TODO
    snap_folder = ds.snp_path
    sheet_name = ds.get_sheet_name()
    if sheet_name in data:
        continue
    pandas_dict = {"UID": [], "name": []}
    empty = []
    id = 0
    l = list(snap_folder.iterdir())
    random.shuffle(l)  # PRODUCE random priority
    for id, path in enumerate(l):
        print(f"Step 0 - {id:5} {path}               ", end="\r")
        name = path.name
        pandas_dict["UID"].append(secrets.token_urlsafe(6).replace("-", "A").replace("_", "Z"))
        pandas_dict["name"].append(name)

        bids_keys = [tuple(name.split("-", maxsplit=1)) for name in name.split("_")[:-1]]
        bids_keys = {k: v for k, v in bids_keys}

        for k, v in bids_keys.items():
            if k not in pandas_dict:
                pandas_dict[k] = empty.copy()
            while len(pandas_dict[k]) == id - 1:
                pandas_dict[k].append("")
            pandas_dict[k].append(v)
        empty.append("")
    pandas_dict["exclusion_reason"] = empty.copy()
    pandas_dict["crop_random"] = empty.copy()
    pandas_dict["used"] = empty.copy()
    for k, v in pandas_dict.items():
        while len(pandas_dict[k]) == id + 1:
            pandas_dict[k].append("")

    print(sheet_name, "                                                          ")
    df = pd.DataFrame.from_dict(pandas_dict)
    # with xls as xls2:
    data[sheet_name] = df.copy()


def save_xls():
    with pd.ExcelWriter(excel_path) as xls:
        for sheet_name, df in data.items():
            df.to_excel(xls, sheet_name=sheet_name, index=False)


save_xls()
from BIDS import BIDS_FILE, to_nii

evaluation_sheet = {"NAME": [], "Comment": [], "T1w_real": [], "STIR_real": []}
#################################################
for ds in datasets:
    # for sheet_name, df in data.items():
    sheet_name = ds.get_sheet_name()
    df = data[sheet_name]
    limit = ds.file_limit
    for id, row in df.iterrows():
        if "sequ" in row and row["sequ"] == "HWS":
            df.loc[id, "exclusion_reason"] = "HWS"  # type: ignore
            limit += 1
            continue
        if "chunk" in row and row["chunk"] == "HWS":
            df.loc[id, "exclusion_reason"] = "HWS"  # type: ignore
            limit += 1
            continue
        snp = out_folder / f"{row['UID']}.jpg"
        out_folder2 = Path(str(out_folder) + "_gt")
        out_folder2.mkdir(exist_ok=True)
        snp2 = out_folder2 / f"{ds.key}_{int(id)+2}.jpg"  # type: ignore

        if str(row["exclusion_reason"]) != "nan" and str(row["exclusion_reason"]) != "":
            limit += 1
            snp.unlink(missing_ok=True)
            snp2.unlink(missing_ok=True)
            continue
        if id == limit:
            break

        if snp.exists() and str(row["crop_random"]) != "nan" and str(row["crop_random"]) != "":
            evaluation_sheet["NAME"].append(row["UID"])
            evaluation_sheet["Comment"].append(f"{ds.key}_{int(id)+2}")  # type: ignore
            evaluation_sheet["T1w_real"].append(1 if ds.t1w_real else 0)  # type: ignore
            evaluation_sheet["STIR_real"].append(1 if ds.stir_real else 0)  # type: ignore
            continue
        name = row["sub"]
        if "NAKO" in ds.key.upper():
            img_folder = ds.path / ds.rawdata / f"{str(name)[:3]}" / f"sub-{name}"
            T1w_path = img_folder / "T1w" / f"sub-{name}_sequ-{row['sequ']}_acq-sag_desc-generated_T1w.nii.gz"
            T2w_path = img_folder / "T2w" / f"sub-{name}_sequ-{row['sequ']}_acq-sag_res-T1w_desc-n4stitched_T2w.nii.gz"
            STIR_path = img_folder / "T2w" / f"sub-{name}_sequ-{row['sequ']}_acq-sag_desc-generated_STIR.nii.gz"
        elif "SHIP" in ds.key.upper():
            img_folder = ds.path / ds.rawdata / f"{str(name)[:5]}" / f"sub-{name}"
            if "FAKE" in ds.key:
                T1w_path = img_folder / "T1w" / f"sub-{name}_sequ-{row['sequ']}_desc-generated_T1w.nii.gz"
            elif "REAL" in ds.key:
                img_folder2 = ds.path / "rawdata" / f"{str(name)[:5]}" / f"sub-{name}"
                seq = str(row["sequ"]).replace("T2w", "T1w")
                T1w_path = img_folder2 / "T1w" / f"sub-{name}_sequ-{seq}_T1w.nii.gz"
            else:
                raise NotImplementedError()
            T2w_path = img_folder / "T2w" / f"sub-{name}_sequ-{row['sequ']}_res-T1w_T2w.nii.gz"
            STIR_path = img_folder / "T2w" / f"sub-{name}_sequ-{row['sequ']}_desc-generated_STIR.nii.gz"
        elif "DIX" in name:
            img_folder = ds.path / ds.rawdata / f"sub-{name}"
            img_folder = img_folder / str(row["chunk"])
            T1w_path = img_folder / f"sub-{name}_sequ-{row['chunk']}_T1w.nii.gz"
            T2w_path = img_folder / f"sub-{name}_sequ-{row['chunk']}_part-inphase_T2w.nii.gz"
            STIR_path = img_folder / f"sub-{name}_sequ-{row['chunk']}_part-water_T2w.nii.gz"
        elif "wsgan" in name.lower():
            img_folder = ds.path / ds.rawdata / f"sub-{name}"
            try:
                img_folder = next(img_folder.glob(f'ses-{row["ses"]}*')) / str(row["sequ"])
            except StopIteration:
                limit += 1
                df.loc[id, "exclusion_reason"] = "SessionNotFound"  # type: ignore
                continue
            T1w_path = img_folder / f"sub-{name}_ses-{row['ses']}_sequ-{row['sequ']}_T1w.nii.gz"
            T2w_path = img_folder / f"sub-{name}_ses-{row['ses']}_sequ-{row['sequ']}_part-inphase_T2w.nii.gz"
            STIR_path = img_folder / f"sub-{name}_ses-{row['ses']}_sequ-{row['sequ']}_part-water_T2w.nii.gz"
        else:
            raise NotImplementedError
        if not STIR_path.exists():
            print("FileNotFound", STIR_path)
            df.loc[id, "exclusion_reason"] = "FileNotFound STIR"  # type: ignore
            limit += 1
            continue
        print(f"Step 1-4 - {id:5}/{limit} {T2w_path.name}               ", end="\r")  # end="\r"
        zoom = (1.1, 1.1, -1)
        T1w = to_nii(T1w_path)
        T2w = to_nii(T2w_path)
        STIR = to_nii(STIR_path)
        if "SHIP_REAL" in ds.key:
            T1w.resample_from_to_(T2w, verbose=False)

        T1w.reorient_().rescale_(zoom)
        T2w.reorient_().rescale_(zoom)
        STIR.reorient_().rescale_(zoom)
        if str(row["crop_random"]) != "nan" and str(row["crop_random"]) != "":
            # print(str(row["crop_random"]))
            out = str(row["crop_random"]).replace("(", "").replace(")", "").split(",")
            bb = (slice(int(out[0]), int(out[1])), slice(int(out[2]), int(out[3])), slice(int(out[4]), int(out[5])))
        else:
            shape = T1w.shape
            if "NAKO" in ds.key:
                bb = [
                    slice(int(shape[0] * 0.1), shape[0]),
                    slice(int(shape[2] * 0.40), shape[1]),
                    slice(0, shape[2]),
                ]
            elif "SHIP" in ds.key:
                bb = [
                    slice(int(shape[0] * 0.1), shape[0]),
                    slice(0, shape[1]),
                    slice(0, shape[2]),
                ]
            else:
                bb = [slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])]

            # ((32, 182), (9, 159), (1, 10))
            bb = T1w.compute_crop_slice(other_crop=tuple(bb))
            bb = T2w.compute_crop_slice(other_crop=bb)
            bb = STIR.compute_crop_slice(other_crop=bb)
            crop_size_max = 150
            bb = list(bb)
            for i in range(2):
                if bb[i].stop - bb[i].start > crop_size_max:
                    start = bb[i].start
                    end = bb[i].stop - crop_size_max
                    s = random.randint(start, end)
                    bb[i] = slice(s, s + crop_size_max)
            bb = tuple(bb)
            df.loc[id, "crop_random"] = str(tuple(((i.start, i.stop) for i in bb)))  # type: ignore
        T1w.normalize_(min_out=0, quantile=0.99)
        T2w.normalize_(min_out=0, quantile=0.99)
        STIR.normalize_(min_out=0, quantile=0.99)
        T1w.apply_crop_slice_(bb)
        T2w.apply_crop_slice_(bb)
        STIR.apply_crop_slice_(bb)
        create_snapshot(
            [snp, snp2],  # type: ignore
            [
                Snapshot_Frame(T2w, rescale_to_iso=False),
                Snapshot_Frame(T1w, rescale_to_iso=False) if T1w_test else Snapshot_Frame(STIR, rescale_to_iso=False),
            ],
        )
        evaluation_sheet["NAME"].append(row["UID"])
        evaluation_sheet["Comment"].append(f"{ds.key}_{int(id)+2}")  # type: ignore
        if T1w_test:
            evaluation_sheet["T1w_real"].append(1 if ds.t1w_real else 0)  # type: ignore
        else:
            evaluation_sheet["STIR_real"].append(1 if ds.stir_real else 0)  # type: ignore
        df.loc[id, "used"] = "1"  # type: ignore

### MAKE fill-out
if T1w_test:
    del evaluation_sheet["STIR_real"]
else:
    del evaluation_sheet["T1w_real"]
data["evaluation_sheet_solution"] = pd.DataFrame.from_dict(evaluation_sheet).sort_values("NAME")
l = len(evaluation_sheet["Comment"])
if T1w_test:
    evaluation_sheet["T1w_real"] = ["" for i in range(l)]

else:
    evaluation_sheet["STIR_real"] = ["" for i in range(l)]

evaluation_sheet["Comment"] = ["" for i in range(l)]
data["evaluation_sheet"] = pd.DataFrame.from_dict(evaluation_sheet).sort_values("NAME")
save_xls()

# df1 = pd.read_excel(xls, "Sheet1")
# df2 = pd.read_excel(xls, "Sheet2")
# with pd.ExcelWriter("output.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="Sheet_name_1")
#    df2.to_excel(writer, sheet_name="Sheet_name_2")
