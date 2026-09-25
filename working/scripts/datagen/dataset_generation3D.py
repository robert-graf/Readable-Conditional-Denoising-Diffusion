from glob import glob
import os
import numpy as np
import nibabel as nib
import time
from math import floor, ceil
from pathlib import Path

try:
    from utils.datagen.dataset_generation import save_nii, load_nii, pad, get_random_deform_parameter
except:
    from dataset_generation import save_nii, load_nii, pad, get_random_deform_parameter


def deformed_np(arr_dic: dict, interpolation_lvl: dict, sigma=None, points=None, crackup=1):
    if sigma == None or points == None:
        sigma, points = get_random_deform_parameter(crackup=crackup)

    print("deform parm", sigma, points)
    t = time.time()
    keys = list(arr_dic.keys())
    # Deform
    import elasticdeform

    print("interpolation function", interpolation_lvl)
    out = elasticdeform.deform_random_grid(
        [pad(arr_dic[key]) for key in keys], sigma=sigma, points=points, order=[interpolation_lvl[key] for key in keys]
    )

    out = [i[10:-10, 10:-10, 10:-10] for i in out]
    print(round(time.time() - t))
    return dict(zip(keys, out))


def TGD_3D_deform(start_path: str, axis: int = 0, tgd_mode=True, **kwargs):
    if tgd_mode:
        ct_arr, ct_affine = load_nii(start_path.replace("[$FOLDER$]", "ct"))
        mi_arr, mi_affine = load_nii(start_path.replace("[$FOLDER$]", "mri"))
    else:
        ct_arr, ct_affine = load_nii(start_path[0])
        mi_arr, mi_affine = load_nii(start_path[1])
    arr_dic = {"CT": ct_arr, "T1": mi_arr}
    affine_dic = {"CT": ct_affine, "T1": mi_affine}

    interpolation_lvl = {"CT": 3, "T1": 3}

    arr_dic = deformed_np(arr_dic, interpolation_lvl, **kwargs)
    for key, value in arr_dic.items():
        arr_dic[key] = np.round(value, decimals=5)
    # axis
    if axis != 0:
        for key, value in arr_dic.items():
            arr_dic[key] = value.swapaxes(0, axis)
    return (
        arr_dic,
        affine_dic,
    )  # normalize, filter, affine_dic


def TGD_3D_deform_save(start_path: Path, out_path: Path, out_name: str, axis: int = 0, **kwargs):
    (arr_dic, affine_dic) = TGD_3D_deform(start_path, axis, **kwargs)
    for key, arr in arr_dic.items():
        if key == "CT":
            p = out_path.replace("[$FOLDER$]", "ct")
        else:
            p = out_path.replace("[$FOLDER$]", "mri")
        save_nii(arr, affine_dic[key], p, out_name, verbose=True)


def wopathfx_3D_deform_save(start_path: Path, out_path: Path, out_name: str, axis: int = 0, **kwargs):
    start_path_list = [None, None]
    for f in glob(start_path):
        f: str
        if f.endswith("_ct.nii.gz"):
            start_path_list[0] = f
        elif f.endswith("_T1.nii.gz"):
            start_path_list[1] = f
    if start_path_list[0] is None or start_path_list[1] is None:
        return
    print(start_path_list)

    (arr_dic, affine_dic) = TGD_3D_deform(start_path_list, axis, tgd_mode=False, **kwargs)
    for key, arr in arr_dic.items():
        if key == "CT":
            p = out_path.replace("[$FOLDER$]", "ct")
        else:
            p = out_path.replace("[$FOLDER$]", "mri")
        save_nii(arr, affine_dic[key], p, out_name, verbose=True)


if __name__ == "__main__":
    # for j in range(25):
    #    for i in range(500):
    #        png = False
    #        try:
    #            search_path = f"/media/data/robert/datasets/TGD_3D/[$FOLDER$]/train/{i}.nii.gz"
    #            if os.path.exists(search_path.replace("[$FOLDER$]", "ct")):
    #                print(i)
    #                # def make_nii_to_slice(png:bool,start_path:Path, out_path:Path, out_name:str, axis: int = 2 ,**kwargs):
    #                out_path = f"/media/data/robert/datasets/TGD_3D/[$FOLDER$]/train/{j}/"
    #                out_name = f"{i:04d}"
    #                TGD_3D_deform_save(search_path, out_path, out_name)
    #                # print(filter)
    #        except Exception as e:
    #            print(i, e)
    #            # raise e
    #            pass

    # for j in range(30):
    #    for i in range(500):
    #        png = False
    #        try:
    #            search_path = f"/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/rawdata_/fxclass{i:04d}/sorted/*.nii.gz"
    #
    #            # def make_nii_to_slice(png:bool,start_path:Path, out_path:Path, out_name:str, axis: int = 2 ,**kwargs):
    #            out_path = f"/media/data/robert/datasets/wopathfx3d/[$FOLDER$]/train/{j}/"
    #            out_name = f"fxclass{i:04d}"
    #            wopathfx_3D_deform_save(search_path, out_path, out_name)
    #            # print(filter)
    #        except Exception as e:
    #            print(i, e)
    #            raise e
    #            pass
    for i in range(500):
        png = False
        try:
            search_path = (
                f"/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/rawdata_/fxclass{i:04d}/sorted/*.nii.gz"
            )

            # def make_nii_to_slice(png:bool,start_path:Path, out_path:Path, out_name:str, axis: int = 2 ,**kwargs):
            out_path = f"/media/data/robert/datasets/wopathfx3d/[$FOLDER$]/test/{0}/"
            out_name = f"fxclass{i:04d}"
            wopathfx_3D_deform_save(search_path, out_path, out_name)
            # print(filter)
        except Exception as e:
            print(i, e)
            raise e
            pass
