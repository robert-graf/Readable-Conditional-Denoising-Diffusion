import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz

from numpy.typing import NDArray
from BIDS import BIDS_Global_info, Subject_Container, BIDS_FILE, to_nii, load_centroids, calc_centroids_labeled_buffered
from BIDS.registration.ridged_points import reg_segmentation
from joblib import Parallel, delayed

import pandas as pd


def setup_reg_affine(start_path: Path, key: str, sequ: str, axis: int, filter=None):

    s = sequ.split("_")
    folder = f"target-{s[0]}_from-{s[1]}"
    dixon = None
    ct = None
    seg = None
    # /media/data/robert/datasets/spinegan_T2w_1p_reg/registration/spinegan0001/target-302_from-None
    # /media/data/robert/datasets/spinegan_T2w_1p_reg/registration/spinegan0001/target-302_from-None
    print(Path(start_path, "registration", key, folder))
    for p in Path(start_path, "registration", key, folder).iterdir():
        if p.name.endswith("dixon.nii.gz"):
            dixon = p
        elif p.name.endswith("ct.nii.gz"):
            ct = p
        elif p.name.endswith("seg-vert_msk.nii.gz"):
            seg = p
    #
    ct_arr = to_nii(ct).reorient_(("R", "I", "P")).rescale_((-1, 1, 1), verbose=True)
    mi_arr = to_nii(dixon).reorient_(("R", "I", "P")).rescale_((-1, 1, 1))
    sg_arr = to_nii(seg, True).reorient_(("R", "I", "P")).rescale_((-1, 1, 1))
    arr_dic = {"CT": ct_arr.get_array(), "T2": mi_arr.get_array(), "SG": sg_arr.get_array()}
    affine_dic = {"CT": ct_arr.affine, "T2": mi_arr.affine, "SG": sg_arr.affine}

    interpolation_lvl = {"CT": 3, "T2": 3, "SG": 0}
    normalize = {
        "CT": lambda x, volumes: (x + 1000) / 2000,
        "T2": lambda x, volumes: x / max(0.00001, np.max(volumes)),
        "SG": lambda x, volumes: x,
    }

    return arr_dic, interpolation_lvl, normalize, filter, affine_dic


def main(n_jobs=12, png=False):

    Parallel(n_jobs=n_jobs)(delayed(__main)(png, j) for j in range(1 if png else 12))


def __main(png, j):
    for search_path in ["/media/data/robert/datasets/spinegan_T2w_1p_reg/"]:

        try:
            search_path = Path(search_path)
            # print(search_path)
            df = pd.read_excel(str(Path("/media/data/robert/datasets/spinegan_T2w/", "info.xlsx")))
            if j < 10:
                split = "train"
            elif j == 10:
                split = "val"
            elif j == 11:
                split = "test"
            else:
                assert False
            df = df[df["Split"] == split]

            if png:
                out_path = "/media/data/robert/datasets/spinegan_T2w_1p_reg/img/deform"
            elif j == 10:
                out_path = f"/media/data/robert/datasets/spinegan_T2w_1p_reg/val/"
            elif j == 11:
                out_path = f"/media/data/robert/datasets/spinegan_T2w_1p_reg/test/"
            else:
                out_path = f"/media/data/robert/datasets/spinegan_T2w_1p_reg/train/{j}/"
            for key, sequ in list(zip(df["Name"], df["Sequ"])):
                key = f"spinegan{int(key):04d}"
                try:
                    arr_dic, interpolation_lvl, normalize, filter, _ = setup_reg_affine(
                        search_path, key, sequ, 2, filter="SG"
                    )
                except ValueError:
                    print("file not found")
                    continue
                except FileNotFoundError:
                    print("file not found")
                    continue
                except EOFError:
                    print("EOFError")
                    continue
                # print(filter)
                if png:
                    make_np_to_PNG(
                        out_path,
                        key + "_" + str(j),
                        arr_dic=arr_dic,
                        interpolation_lvl=interpolation_lvl,
                        normalize=normalize,
                        filter=filter,
                        deform=True,
                        crop3D="T2",
                        crackup=0.8,
                        single_png=True,
                    )
                else:
                    if Path(out_path, key + "_" + str(j) + "_0.npz").exists():
                        continue
                    make_np_to_npz(
                        out_path,
                        key + "_" + str(j),
                        arr_dic=arr_dic,
                        interpolation_lvl=interpolation_lvl,
                        normalize=normalize,
                        filter=filter,
                        deform=j < 9,
                        crop3D="T2",
                        crackup=0.8,
                    )

        except Exception as e:
            raise e


# Save registered file
def generate_file_path_reg(file: BIDS_FILE, other_file_id, folder, keys=["sub", "ses", "sequ", "reg", "acq"]):
    info = file.info.copy()
    info["reg"] = other_file_id
    file.info.clear()

    def pop(key):
        if key in info:
            file.info[key] = info.pop(key)

    for k in keys:
        pop(k)
    for k, v in info.items():
        file.info[k] = v
    out_file: Path = file.get_changed_path(
        dataset_path="/media/data/robert/datasets/spinegan_T2w_1p_reg2/",
        file_type="nii.gz",
        parent="registration",
        path="{sub}/" + folder,
        info=info,
        from_info=True,
    )
    return out_file


def register(n_jobs=10):
    global_info = BIDS_Global_info(
        ["/media/data/robert/datasets/spinegan_T2w/raw"],
        ["sourcedata", "rawdata", "rawdata_ct", "rawdata_dixon", "derivatives"],
        additional_key=["sequ", "seg", "ovl", "e"],
    )

    Parallel(n_jobs=n_jobs)(
        delayed(__register)(subj_name, subject) for subj_name, subject in global_info.enumerate_subjects()
    )


def __register(subj_name, subject: Subject_Container):
    # subject: Subject_Container
    query = subject.new_query()
    # It must exist a dixon and a msk
    query.filter("format", "dixon")
    # A nii.gz must exist
    query.filter("Filetype", "nii.gz")
    query.filter("format", "msk")
    # print(query)
    d = None
    for bids_file in query.loop_dict():
        # Do something with the files.
        # print(bids_file)
        dixon = bids_file["dixon"]
        msk = bids_file["msk"][0]
        e = msk.get("e")

        for d in dixon:
            if d.get("e") == e:
                break

        assert d is not None and d.get("e") == e

        query = subject.new_query(flatten=False)
        # A nii.gz must exist
        query.filter("Filetype", "nii.gz")
        # It must be a ct
        query.filter("format", "ct")
        query.filter("seg", "vert")
        for bids_file in query.loop_dict():
            assert len(bids_file["ct"]) == 1, bids_file["ct"]
            assert len(bids_file["msk_vert"]) == 1, bids_file["msk_vert"]
            # assert len(bids_file["msk"]) == 1, bids_file
            ct = bids_file["ct"][0]
            vert = bids_file["msk_vert"][0]
            subreg = bids_file["msk_subreg"][0]

            ctd_mr = calc_centroids_labeled_buffered(msk_reference=msk, subreg_reference=None)
            ctd_ct = calc_centroids_labeled_buffered(msk_reference=vert, subreg_reference=subreg, subreg_id=50)

            reg_segmentation.ridged_registration_by_dict_from_ctd(
                [d, msk],
                [ct, vert],
                ctd_mr,
                ctd_ct,
                generate_file_names=generate_file_path_reg,
                axcodes_to=("R", "I", "P"),
                # voxel_spacing=(-1, 1, 1),
                snap_shot_folder="/media/data/robert/datasets/spinegan_T2w_1p_reg2/",
            )
            exit()


if __name__ == "__main__":
    # register(n_jobs=1)
    main()
    pass
