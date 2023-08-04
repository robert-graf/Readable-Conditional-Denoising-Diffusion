import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz

from numpy.typing import NDArray
from BIDS import BIDS_Global_info, Subject_Container, BIDS_FILE, to_nii, NII, load_centroids
from math import ceil as c, floor as f


def setup_reg_affine(
    dixon: BIDS_FILE,
    ct: BIDS_FILE,
    ctd: BIDS_FILE,
    key: str,
    sequ: str,
    axis: int,
    filter=None,
):

    # rescale_and_reorient
    center = load_centroids(ctd)
    ct_nii = to_nii(ct)
    center.zoom = ct_nii.zoom
    center.shape = ct_nii.shape

    mi_arr: NII = to_nii(dixon).rescale_and_reorient_(("R", "I", "P"), voxel_spacing=(-1, 1, 1))
    ct_arr: NII = ct_nii.rescale_and_reorient_(("R", "I", "P"), voxel_spacing=mi_arr.zoom)
    center = center.reorient(("R", "I", "P")).rescale_(ct_arr.zoom)
    x1 = 0
    y1 = 0
    z1 = 0
    for _, (x, y, z) in center.items():
        x1 += x
        y1 += x
        z1 += x
    x1 /= len(center)
    y1 /= len(center)
    z1 /= len(center)
    x2, y2, z2 = mi_arr.shape
    x1 = max(f(x1 - x2 / 2), 0)
    y1 = max(f(y1 - y2 / 2), 0)
    z1 = max(f(z1 - z2 / 2), 0)
    slicing = (
        slice(x1, x1 + x2),
        slice(y1, y1 + y2),
        slice(z1, z1 + z2),
    )
    # print(slicing)
    # print(ct_arr.shape, mi_arr.shape)

    ct_arr.apply_crop_slice_(slicing)
    if ct_arr.shape != mi_arr.shape:
        offx = (mi_arr.shape[0] - ct_arr.shape[0]) // 2
        offy = (mi_arr.shape[1] - ct_arr.shape[1]) // 2
        offz = (mi_arr.shape[2] - ct_arr.shape[2]) // 2
        slicing = (
            slice(offx, -offx if offx != 0 else None),
            slice(offy, -offy if offy != 0 else None),
            slice(offz, -offz if offz != 0 else None),
        )
        mi_arr.apply_crop_slice_(slicing)

    # print(center, ct_arr.zoom)
    # sg_arr: NII = to_nii(seg, True).reorient_to_(("R", "I", "P"))

    arr_dic = {"CT": ct_arr.get_array(), "T2": mi_arr.get_array()}  # , "SG": sg_arr.get_array()}
    affine_dic = {"CT": ct_arr.affine, "T2": mi_arr.affine}  # "SG": sg_arr.affine}

    interpolation_lvl = {"CT": 3, "T2": 3}  # "SG": 0
    normalize = {
        "CT": lambda x, volumes: (x + 1000) / 2000,
        "T2": lambda x, volumes: x / max(0.00001, np.max(volumes)),
        "SG": lambda x, volumes: x,
    }
    return arr_dic, interpolation_lvl, normalize, filter, affine_dic


if __name__ == "__main__":
    import pandas as pd

    look_up = {}
    global_info = BIDS_Global_info(
        ["/media/data/robert/datasets/spinegan_T2w/raw"],
        ["sourcedata", "rawdata", "rawdata_ct", "rawdata_dixon", "derivatives"],  #
    )
    for subj_name, subject in global_info.enumerate_subjects():
        subject: Subject_Container
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
            dixon = bids_file["dixon"]
            e = bids_file["msk"][0].get("e")

            for d in dixon:
                assert "e" in d.info, d
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
                assert "ct" in bids_file, bids_file
                assert "msk_vert" in bids_file, bids_file
                assert len(bids_file["ct"]) == 1, bids_file
                assert len(bids_file["msk_vert"]) == 1, bids_file
                ct = bids_file["ct"][0]
                ctd = bids_file["ctd_subreg"][0]
                # vert = bids_file["msk_vert"][0]
                look_up[(d.get("sub"), f"{d.get('sequ')}_{ct.get('sequ')}")] = (d, ct, ctd)

    print(look_up.keys())
    png = False
    for j in range(1 if png else 12):

        for search_path in ["/media/data/robert/datasets/spinegan_T2w/"]:

            try:
                search_path = Path(search_path)
                df = pd.read_excel(str(Path(search_path, "info.xlsx")))
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
                    out_path = "/media/data/robert/datasets/spinegan_T2w_no_reg/img/deform"
                elif j == 10:
                    out_path = f"/media/data/robert/datasets/spinegan_T2w_no_reg/val/"
                elif j == 11:
                    out_path = f"/media/data/robert/datasets/spinegan_T2w_no_reg/test/"
                else:
                    out_path = f"/media/data/robert/datasets/spinegan_T2w_no_reg/train/{j}/"
                print(out_path)

                for key, sequ in list(zip(df["Name"], df["Sequ"])):
                    key = f"spinegan{int(key):04d}"
                    d, ct, ctd = look_up[(key, sequ)]

                    try:
                        arr_dic, interpolation_lvl, normalize, filter, _ = setup_reg_affine(
                            d, ct, ctd, key, sequ, 2, filter=None
                        )
                    except KeyError:
                        print("file not found")
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
