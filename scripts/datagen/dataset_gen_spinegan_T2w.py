import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz

from numpy.typing import NDArray


def setup_reg_affine(start_path: Path, key: str, sequ: str, axis: int, filter=None, only_oversampling=False):

    s = sequ.split("_")
    folder = f"target-{s[0]}_from-{s[1]}"
    dixon = None
    ct = None
    seg = None

    for p in Path(start_path, "registration", key, folder).iterdir():
        if p.name.endswith("real_dixon.nii.gz"):
            dixon = p
        elif p.name.endswith("ct.nii.gz"):
            ct = p
        elif p.name.endswith("seg-vert_msk.nii.gz"):
            seg = p
    #
    ct_arr, ct_affine = load_nii(ct)
    mi_arr, mi_affine = load_nii(dixon)
    sg_arr, sg_affine = load_nii(seg)

    arr_dic = {"CT": ct_arr, "T2": mi_arr, "SG": sg_arr}
    affine_dic = {"CT": ct_affine, "T2": mi_affine, "SG": sg_affine}

    # axis
    for key, value in arr_dic.items():
        # value: np.ndarray
        arr_dic[key] = value.swapaxes(1, 2)
        pass
        # arr_dic[key] = value.swapaxes(0, axis)[:, ::-1, ::-1].copy()  # .swapaxes(1, 2)  # [:, ::-1, ::-1].copy()

    interpolation_lvl = {"CT": 3, "T2": 3, "SG": 0}
    normalize = {
        "CT": lambda x, volumes: (x + 1000) / 2000,
        "T2": lambda x, volumes: x / max(0.00001, np.max(volumes)),
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
        for search_path in ["/media/data/robert/datasets/spinegan_T2w/"]:

            try:
                search_path = Path(search_path)
                print(search_path)
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
                    out_path = "/media/data/robert/datasets/spinegan_T2w/img/deform"
                elif j == 10:
                    out_path = f"/media/data/robert/datasets/spinegan_T2w/val/"
                elif j == 11:
                    out_path = f"/media/data/robert/datasets/spinegan_T2w/test/"
                else:
                    out_path = f"/media/data/robert/datasets/spinegan_T2w/train/{j}/"
                for key, sequ in list(zip(df["Name"], df["Sequ"])):
                    key = f"spinegan{int(key):04d}"
                    try:
                        arr_dic, interpolation_lvl, normalize, filter, _ = setup_reg_affine(
                            search_path, key, sequ, 2, filter="SG"
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
