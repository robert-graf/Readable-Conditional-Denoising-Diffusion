import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz

from numpy.typing import NDArray


def setup_reg_affine(start_path: Path, key: str, axis: int, filter=None, only_oversampling=False):
    if key.startswith("fxclass"):
        ct_arr, ct_affine = load_nii(Path(start_path, key, "target-1_from-2", f"sub-{key}_sequ-2_reg-1_ct.nii.gz"))
        mi_arr, mi_affine = load_nii(Path(start_path, key, "target-1_from-2", f"sub-{key}_sequ-1_reg-2_T1c.nii.gz"))
        sg_arr, sg_affine = load_nii(
            Path(start_path, key, "target-1_from-2", f"sub-{key}_sequ-2_reg-1_seg-vert_msk.nii.gz")
        )
    else:
        assert False

        # folder, split = key.split("_")
        # ct_arr, ct_affine = load_nii(next(Path(start_path, folder).glob(f"*_{split}_reg_ct.nii.gz")))
        # mi_arr, mi_affine = load_nii(next(Path(start_path, folder).glob(f"*_{split}_dixon.nii.gz")))
        # sg_arr, sg_affine = load_nii(next(Path(start_path, folder).glob(f"*_{split}_seg_dixon.nii.gz")))

    print(ct_arr.shape)
    arr_dic = {"CT": ct_arr, "T1": mi_arr, "SG": sg_arr}
    affine_dic = {"CT": ct_affine, "T1": mi_affine, "SG": sg_affine}

    # axis
    for key, value in arr_dic.items():
        # value: np.ndarray
        # value = value.transpose()
        arr_dic[key] = value.swapaxes(0, axis)[:, ::-1, ::-1].copy()  # .swapaxes(1, 2)  # [:, ::-1, ::-1].copy()

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
    for j in range(1 if png else 12):
        for search_path in ["/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/ML_translation/registration/"]:

            try:
                search_path = Path(search_path)
                print(search_path)
                df = pd.read_excel(str(Path(search_path.parent, "info.xlsx")))
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
                    out_path = "/media/data/robert/datasets/fx_T1w_v2/img/deform"
                elif j == 10:
                    out_path = f"/media/data/robert/datasets/fx_T1w_v2/val/"
                elif j == 11:
                    out_path = f"/media/data/robert/datasets/fx_T1w_v2/test/"
                else:
                    out_path = f"/media/data/robert/datasets/fx_T1w_v2/train/{j}/"
                for key in list(df["Name"]):

                    print(key)
                    try:
                        arr_dic, interpolation_lvl, normalize, filter, _ = setup_reg_affine(
                            search_path, key, 2, filter="SG"
                        )
                    except FileNotFoundError:
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
                            deform=j < 9,
                            crop3D="T1",
                            crackup=0.8,
                        )

            except Exception as e:
                raise e
