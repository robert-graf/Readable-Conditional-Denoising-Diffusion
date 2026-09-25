import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz


def setup_bailiang(start_path: Path, key: str, axis: int, filter=None, only_oversampling=False):
    if key.startswith("fxclass"):
        ct_arr, ct_affine = load_nii(Path(start_path, key, "reg_ct.nii.gz"))
        mi_arr, mi_affine = load_nii(Path(start_path, key, "T1.nii.gz"))
        sg_arr, sg_affine = load_nii(Path(start_path, key, "seg_T1.nii.gz"))
    else:
        folder, split = key.split("_")
        ct_arr, ct_affine = load_nii(next(Path(start_path, folder).glob(f"*_{split}_reg_ct.nii.gz")))
        mi_arr, mi_affine = load_nii(next(Path(start_path, folder).glob(f"*_{split}_dixon.nii.gz")))
        sg_arr, sg_affine = load_nii(next(Path(start_path, folder).glob(f"*_{split}_seg_dixon.nii.gz")))

    print(ct_arr.shape)
    arr_dic = {"CT": ct_arr, "T1": mi_arr, "SG": sg_arr}
    affine_dic = {"CT": ct_affine, "T1": mi_affine, "SG": sg_affine}
    # axis
    for key, value in arr_dic.items():
        value: np.ndarray
        value = value.transpose()
        arr_dic[key] = value.swapaxes(0, axis).swapaxes(1, 2)[:, ::-1, ::-1].copy()

    interpolation_lvl = {"CT": 3, "T1": 3, "SG": 0}
    normalize = {
        "CT": lambda x, volumes: (x + 1000) / 2000,
        "T1": lambda x, volumes: x / max(0.00001, np.max(volumes)),
        "SG": lambda x, volumes: x,
    }
    if only_oversampling:
        # Remove parts of the segmentation, so only images with processus spinosus are used
        from scipy.ndimage.measurements import center_of_mass

        tmp = arr_dic["SG"]
        arr_dic["SG"] = np.zeros_like(arr_dic["SG"])

        for i in np.unique(tmp):
            msk_temp = np.zeros(tmp.shape, dtype=bool)
            msk_temp[tmp == i] = True
            ctr_mass = center_of_mass(msk_temp)
            x = int(ctr_mass[0])
            arr_dic["SG"][x] = tmp[x]

    return arr_dic, interpolation_lvl, normalize, filter, affine_dic


if __name__ == "__main__":
    import pandas as pd

    png = False
    for j in range(1 if png else 10):
        for search_path in [
            "/media/data/robert/datasets/2022_09_08_wopathfx_reg_bailiang/registered/",
            "/media/data/robert/datasets/2022_09_09_NRad_reg_bailiang/registered",
        ]:
            try:
                search_path = Path(search_path)
                print(search_path)
                df = pd.read_excel(str(Path(search_path.parent, "info.xlsx")))
                df = df[df["Qualität"] == 0]
                # print(df[["Name", "Qualität"]])
                if png:
                    out_path = "/media/data/robert/datasets/bailiang/img/deform"
                elif j != 10:
                    out_path = f"/media/data/robert/datasets/bailiang/train/{j}/"
                else:
                    out_path = "/media/data/robert/datasets/bailiang/val/"
                for key in list(df["Name"]):
                    print(key)
                    arr_dic, interpolation_lvl, normalize, filter, _ = setup_bailiang(search_path, key, 0, filter="SG")
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
                            crop3D="T1",
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
                            deform=j != 0,
                            crop3D="T1",
                            crackup=0.8,
                        )

            except Exception as e:
                raise e
    print("#### OVERSAMPLING ####")
    for j in range(10, 11 if png else 20):
        print(j)
        for search_path in [
            "/media/data/robert/datasets/2022_09_08_wopathfx_reg_bailiang/registered/",
            "/media/data/robert/datasets/2022_09_09_NRad_reg_bailiang/registered",
        ]:
            try:
                search_path = Path(search_path)
                print(search_path)
                df = pd.read_excel(str(Path(search_path.parent, "info.xlsx")))
                df = df[df["Qualität"] == 0]
                # print(df[["Name", "Qualität"]])
                if png:
                    out_path = "/media/data/robert/datasets/bailiang/img/oversampling"
                else:
                    out_path = f"/media/data/robert/datasets/bailiang/train/{j}/"
                # else:
                #    out_path = "/media/data/robert/datasets/bailiang/val/"
                for key in list(df["Name"]):
                    print(key)
                    arr_dic, interpolation_lvl, normalize, filter, _ = setup_bailiang(
                        search_path, key, 0, filter="SG", only_oversampling=True
                    )
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
                            crop3D="T1",
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
                            deform=j != 0,
                            crop3D="T1",
                            crackup=0.8,
                        )

            except Exception as e:
                raise e
