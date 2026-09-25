import csv
import os
from pathlib import Path
import traceback
from BIDS.bids_files import BIDS_FILE, BIDS_Global_info, Searchquery, Subject_Container
from BIDS.registration.ridged_points.script_body_posterior_two_step import ridged_point_registration, error_dict
from BIDS.registration.ridged_points.reg_segmentations import ridged_registration_by_dict, generate_file_path_reg
from BIDS.nii_utils import (
    to_nii,
    calc_centroids_from_subreg_vert,
    to_dict,
    is_Point3D_tuple,
    nii2arr,
    is_Orientation_tuple,
    reorient_same_as,
    reorient_to,
    resample_mask_to,
)
import nibabel as nib


def rename_vertebras_pre_alignd(
    fix_dict: dict[str, BIDS_FILE | list[BIDS_FILE]], modified_dict: dict[str, BIDS_FILE | list[BIDS_FILE]]
):
    fix_vert_nii = to_nii(fix_dict["vert"][0])
    mod_vert_nii = to_nii(modified_dict["vert"][0])
    fix_subreg_nii = to_nii(fix_dict["subreg"][0])
    mod_subreg_nii = to_nii(modified_dict["subreg"][0])
    fix_cdt = calc_centroids_from_subreg_vert(fix_vert_nii, fix_subreg_nii, subreg_id=50)
    mod_cdt = calc_centroids_from_subreg_vert(mod_vert_nii, mod_subreg_nii, subreg_id=50)

    closest_dict = {}

    ##Confusion Matrix
    for cdt in mod_cdt:
        if is_Point3D_tuple(cdt):
            m_id, x, y, z = cdt
            for cdt in fix_cdt:
                if is_Point3D_tuple(cdt):
                    f_id, x2, y2, z2 = cdt
                    dist = (x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2
                    dist_old = 0
                    if m_id in closest_dict:
                        _, dist_old = closest_dict[m_id]
                        if dist_old < dist:
                            continue
                    closest_dict[m_id] = (f_id, dist)
                    # print(m_id, "-->", f_id, f"\t{dist: 3.1f}\t|\t from {dist_old:3.1f}")
    # Check doubles
    for key, (f_id, dist) in closest_dict.items():
        if f_id == 0:
            continue
        for key_other, (f_id_other, dist_other) in closest_dict.items():
            if f_id_other != f_id:
                continue
            if key == key_other:  # Ignore yourself!
                continue
            if dist_other > dist:
                continue
            closest_dict[key] = (0, 1000_000_000)
            # print(key, "-/->", f_id, f"\t{dist: 3.1f}\t|\t from {dist_other:3.1f}")

    print("mapping")
    mod_vert_arr = nii2arr(mod_vert_nii) + 1000
    save = False
    for key, (f_id, dist) in closest_dict.items():

        print(key, "-->", f_id, f"\t{dist if f_id != 0 else -1: 3.1f}")
        if f_id == 0:
            continue
        if key == 0:
            continue
        if key != f_id:
            save = True
        mod_vert_arr[mod_vert_arr == (key + 1000)] = f_id
    mod_vert_arr[mod_vert_arr == 1000] = 0
    if save:
        nib.save(
            nib.Nifti1Image(mod_vert_arr, mod_vert_nii.affine, mod_vert_nii.header),
            modified_dict["vert"][0].file["nii.gz"],
        )


######################################


def _parallelized_preprocess_scan(subj_name, subject: Subject_Container, mri_rep="T1c", force_override_A=True):
    query1: Searchquery = subject.new_query()
    # It must exist a dixon and a msk
    query1.filter("format", mri_rep)
    query1.flatten()
    query1.filter("format", lambda x: x != "ct")
    query1.unflatten()
    # A nii.gz must exist
    query1.filter("Filetype", "nii.gz")
    query1.filter("format", "msk")

    if len(query1.candidates) == 0:
        return

    def key_transform(x: BIDS_FILE):
        if "seg" not in x.info:
            return None
        # seg-subreg_ctd.nii.gz
        if "subreg" == x.info["seg"] and x.format == "ctd":
            return "ctd"
        return None

    query2 = subject.new_query()
    # Only files with a seg-subreg + ctd file.
    query2.filter("seg", "subreg", required=True)
    # It must exist a ct
    query2.filter("format", "ct")
    query2.filter_non_existence("format", mri_rep)

    for dict_A in query1.loop_dict(key_transform=key_transform):
        for dict_B in query2.loop_dict(key_transform=key_transform):
            ##### SKIT if no change ####
            time_A = max([os.stat(v[0].file["nii.gz"]).st_mtime for k, v in dict_A.items() if "nii.gz" in v[0].file])
            time_B = max([os.stat(v[0].file["nii.gz"]).st_mtime for k, v in dict_B.items() if "nii.gz" in v[0].file])
            fixed_id = dict_A[mri_rep][0].info["sequ"]
            moving_id = dict_B["ct"][0].info["sequ"]
            folder_name = f"target-{fixed_id}_from-{moving_id}"
            file = generate_file_path_reg(dict_A[mri_rep][0], moving_id, folder_name)
            if os.path.exists(file) and os.stat(file).st_mtime >= max(time_A, time_B):
                continue
            else:
                if os.path.exists(file):
                    print(os.stat(file).st_mtime, max(time_A, time_B), os.stat(file).st_mtime <= max(time_A, time_B))
                else:
                    print("fist time", subj_name)
            try:
                ridged_registration_by_dict(
                    dict_A,
                    mri_rep,
                    dict_B,
                    "ct",
                    ids=list(range(40, 51)),
                    axcodes_to=("P", "S", "R"),
                    voxel_spacing=(1, 1, -1),
                )
            except Exception as e:
                print("FAILED", subj_name)
                print(str(traceback.format_exc()))
                raise e
            # ridged_point_registration(dict_A, dict_B)
    global error_dict

    dataset = next(iter(subject.sequences.values()))[0].dataset
    filepath = Path(dataset, f"registration/{subj_name}/reg_error.csv")
    if not filepath.exists():
        return
    with open(str(filepath), "w") as output:
        writer = csv.writer(output)
        for key, value in error_dict.items():
            value = [round(v, ndigits=3) for v in value]
            writer.writerow([key, *value])
    error_dict = {}


def parallel_execution(n_jobs, path, mri_rep="T1c", force_override_A=False):
    from joblib import Parallel, delayed

    global_info = BIDS_Global_info(
        [path],
        [
            "rawdata",
            "rawdat___",
            "derivatives",
            "derivatives_old",
        ],  # "sourcedata"
        additional_key=["sequ", "seg", "e", "ovl"],
        verbose=False,
    )
    print(f"Found {len(global_info.subjects)} subjects in {global_info.datasets}")

    if n_jobs > 1:
        print("[*] Running {} parallel jobs. Note that stdout will not be sequential".format(n_jobs))

    Parallel(n_jobs=n_jobs)(
        delayed(_parallelized_preprocess_scan)(subj_name, subject, mri_rep, force_override_A)
        for subj_name, subject in global_info.enumerate_subjects()
    )
    with open(str(Path(path, f"registration/reg_error.csv")), "w") as output:
        writer = csv.writer(output)
        for subj_name, subject in global_info.enumerate_subjects(sort=True):
            path2 = Path(path, f"registration/{subj_name}/reg_error.csv")
            if not path2.exists():
                continue
            with open(str(path2), "r") as input:
                reader = csv.reader(input)
                for row in reader:
                    writer.writerow(row)
            path2.unlink()

    return None


if __name__ == "__main__":
    a = ""
    # ridged_point_registration()
    from pathlib import Path

    src = "/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/ML_translation"
    for path in Path(src).rglob("*_ct_.nii.gz"):
        path.rename(str(path).replace("_ct_.nii", "_ct.nii"))
    parallel_execution(12, path=src)
    # profile()
