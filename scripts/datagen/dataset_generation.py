import os
import numpy as np
import nibabel as nib
import time
from math import floor, ceil
from pathlib import Path
from numpy.typing import NDArray


def save_nii(arr, affine, path, name, verbose=False):
    if not os.path.isdir(path):
        os.makedirs(path)
    img = nib.Nifti1Image(arr, affine)
    f = os.path.join(path, f"{name}.nii.gz")
    # save
    if verbose:
        print("Save to:", f)
    nib.save(img, f)
    return f


def load_nii(path, name=None):
    path = str(path)
    if name is None:
        f = path
    else:
        f = os.path.join(path, f"{name}.nii.gz")
        if not os.path.exists(f):
            f = os.path.join(path, f"{name}.nii")
        if not os.path.exists(f):
            f = os.path.join(path, name)

    # save
    nii = nib.load(f)

    import nibabel.orientations as nio

    ornt_fr = nio.io_orientation(nii.affine)
    ornt_to = nio.ornt2axcodes(ornt_fr)

    print("zooms", nii.header.get_zooms())
    print("ornt", ornt_to)
    return nii.get_fdata(), nii.affine


def pad(arr):
    return np.pad(arr, 10, mode="reflect")


def deformed_np(arr_dic: dict[str, NDArray], interpolation_lvl: dict[str, int], sigma=None, points=None, crackup=1):
    if sigma is None or points is None:
        sigma, points = get_random_deform_parameter(crackup=crackup)

    print("deform parm", sigma, points)
    t = time.time()
    keys = list(arr_dic.keys())
    # Deform
    import elasticdeform

    print("interpolation function", interpolation_lvl)
    assert sigma is not None

    out: list[NDArray] = elasticdeform.deform_random_grid(
        [pad(arr_dic[key]) for key in keys], sigma=sigma, points=points, order=[interpolation_lvl[key] for key in keys]  # type: ignore
    )

    out2 = [i[10:-10, 10:-10, 10:-10] for i in out]
    print(round(time.time() - t))
    return dict(zip(keys, out2))


def crop_slice(msk: np.ndarray, min_value=0):
    shp = msk.shape
    cor_msk = np.where(msk > min_value)
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0] if c_min[0] > 0 else 0
    y0 = c_min[1] if c_min[1] > 0 else 0
    z0 = c_min[2] if c_min[2] > 0 else 0
    x1 = c_max[0] if c_max[0] < shp[0] else shp[0]
    y1 = c_max[1] if c_max[1] < shp[1] else shp[1]
    z1 = c_max[2] if c_max[2] < shp[2] else shp[2]
    ex_slice = tuple([slice(x0, x1), slice(y0, y1), slice(z0, z1)])
    print(shp, ex_slice)
    return ex_slice


def make_3d_np_to_2d(
    arr_dic: dict[str, NDArray],
    normalize: dict,
    interpolation_lvl: dict,
    filter=None,
    deform=True,
    crop3D=None,
    **kwargs,
) -> list[tuple[int, dict[str, NDArray]]]:
    out = []
    # print(filter)
    if crop3D is not None:
        ex_slice = crop_slice(arr_dic[crop3D])
        for key, value in arr_dic.items():
            arr_dic[key] = value[ex_slice]  # type: ignore

        # crop_slice()
    # polynomial 3D deformation
    if deform:
        # TODO FIXME PADDING
        arr_dic = deformed_np(arr_dic, interpolation_lvl, **kwargs)
    for i in range(arr_dic[list(arr_dic.keys())[0]].shape[0]):
        slice_dic = {}
        for key, value in arr_dic.items():
            # [-1000, 1000] -> [0,1]
            slice = normalize[key](value[i], value)
            if interpolation_lvl[key] != 1:
                slice = np.clip(slice, a_min=0, a_max=1)
                slice = np.round(slice, decimals=5)

            slice_dic[key] = slice
            # print(key,i)
        # filter out when there is no spine segmentation
        if filter is not None:
            seg = np.copy(slice_dic[filter])
            seg[seg > 0] = 1
            if seg.sum() < 10:
                print("skip slice", i)
                continue
        out.append((i, slice_dic))
    return out


def make_np_to_PNG(
    path: str | Path, image_name: str, arr_dic: dict, filter: str | None = None, single_png=False, **kwargs
):
    path = str(path)
    image_list = make_3d_np_to_2d(arr_dic, filter=filter, **kwargs)
    Path(path).mkdir(exist_ok=True, parents=True)
    from PIL import Image

    for id, slice_dic in image_list:
        if single_png:
            arr_list = []
            for key, value in slice_dic.items():
                arr_list.append(value)
            value = np.concatenate(arr_list, axis=1)
            im = Image.fromarray(value * 255)
            im = im.convert("L")
            print(os.path.join(path, f"{image_name}_{id}.png"))
            im.save(os.path.join(path, f"{image_name}_{id}.png"))

        else:
            for key, value in slice_dic.items():
                if key is not filter:
                    im = Image.fromarray(value * 255)
                    im = im.convert("L")
                    print(os.path.join(path, f"{image_name}_{key}_{id}.png"))
                    im.save(os.path.join(path, f"{image_name}_{key}_{id}.png"))


def make_np_to_npz(path, image_name, arr_dic, **kwargs):
    image_list = make_3d_np_to_2d(arr_dic, **kwargs)

    Path(path).mkdir(exist_ok=True, parents=True)
    for id, arr_dic in image_list:
        np.savez_compressed(os.path.join(path, f"{image_name}_{id}.npz"), **arr_dic)


def fetch_nii_files(start_path, search_key):
    from pathlib import Path

    checkpoints = list(Path(start_path).rglob(f"*{search_key}*.nii.gz"))
    assert (
        len(checkpoints) == 1
    ), f"Expected to find one file but found {len(checkpoints)}. From f'{start_path}*{search_key}*.nii.gz'; to {checkpoints}; "
    return checkpoints[0]


def setup_wopathfx(start_path: str, axis: int, filter=None, **kwargs):
    ct_arr, ct_affine = load_nii(fetch_nii_files(start_path, "sorted/*_ct"))
    mi_arr, mi_affine = load_nii(fetch_nii_files(start_path, "_T1"))
    sg_arr, sg_affine = load_nii(fetch_nii_files(start_path, "vert_msk"))

    print(ct_arr.shape)
    arr_dic = {"CT": ct_arr, "T1": mi_arr, "SG": sg_arr}
    affine_dic = {"CT": ct_affine, "T1": mi_affine, "SG": sg_affine}
    # axis
    if axis != 0:
        for key, value in arr_dic.items():
            arr_dic[key] = value.swapaxes(0, axis)

    interpolation_lvl = {"CT": 3, "T1": 3, "SG": 0}
    normalize = {
        "CT": lambda x, volumes: (x + 1000) / 2000,
        "T1": lambda x, volumes: x / max(0.00001, np.max(volumes)),
        "SG": lambda x, volumes: x,
    }

    # filter = 'SG'
    # filter = None
    return arr_dic, interpolation_lvl, normalize, filter, affine_dic


def setup_TGD(start_path: str, key: str, axis: int, filter=None):
    ct_arr, ct_affine = load_nii(start_path.replace("[$FOLDER$]", "ct"))
    mi_arr, mi_affine = load_nii(start_path.replace("[$FOLDER$]", "mri"))
    sg_arr, sg_affine = load_nii(start_path.replace("[$FOLDER$]", "seg"))

    print(ct_arr.shape)
    arr_dic = {"CT": ct_arr, "T1": mi_arr, "SG": sg_arr}
    affine_dic = {"CT": ct_affine, "T1": mi_affine, "SG": sg_affine}
    # axis
    if axis != 0:
        for key, value in arr_dic.items():
            arr_dic[key] = value.swapaxes(0, axis)

    interpolation_lvl = {"CT": 3, "T1": 3, "SG": 0}
    normalize = {
        "CT": lambda x, volumes: (x + 1000) / 2000,
        "T1": lambda x, volumes: x / max(0.00001, np.max(volumes)),
        "SG": lambda x, volumes: x,
    }

    # filter = 'SG'
    # filter = None
    return arr_dic, interpolation_lvl, normalize, filter, affine_dic


def make_nii_to_deformed_nii(start_path: str, out_path: str, out_name: str, axis: int = 2, **kwargs):
    arr_dic, interpolation_lvl, normalize, filter, affine_dic = setup_wopathfx(start_path, axis, **kwargs)
    image_list = make_3d_np_to_2d(
        arr_dic=arr_dic, interpolation_lvl=interpolation_lvl, normalize=normalize, filter=filter, **kwargs
    )
    # print(image_list)
    for key, affine in affine_dic.items():
        print("glue together:", key)
        arr = np.stack([i[1][key] for i in image_list], axis=0)
        print("save", arr.shape)
        # arr[5,...] = 5
        if axis != 0:
            arr = arr.swapaxes(0, axis)
        save_nii(arr, affine, out_path, f"{out_name}_{key}_deformed")


def make_nii_to_slice(png: bool, start_path: str, out_path: str, out_name: str, axis: int = 2, **kwargs):
    arr_dic, interpolation_lvl, normalize, filter, _ = setup_wopathfx(start_path, axis, **kwargs)
    # print(filter)
    if png:
        make_np_to_PNG(
            out_path,
            out_name,
            arr_dic=arr_dic,
            interpolation_lvl=interpolation_lvl,
            normalize=normalize,
            filter=filter,
            **kwargs,
        )
    else:
        make_np_to_npz(
            out_path,
            out_name,
            arr_dic=arr_dic,
            interpolation_lvl=interpolation_lvl,
            normalize=normalize,
            filter=filter,
            **kwargs,
        )


def get_random_deform_parameter(crackup=1):
    import math

    sigma = 2 + np.random.uniform() * 2.5  # 1,5 - 4.5
    min_points = 3
    max_points = 17
    if sigma < 2:
        max_points = 17
    elif sigma < 1.7:
        max_points = 16
    elif sigma < 2.1:
        max_points = 15
    elif sigma < 2.3:
        max_points = 14
    elif sigma < 2.5:
        max_points = 13
    elif sigma < 2.6:
        max_points = 12
    elif sigma < 2.7:
        max_points = 11
    elif sigma < 2.8:
        max_points = 10
    elif sigma < 3:
        max_points = 9
    elif sigma < 3.5:
        max_points = 8
    elif sigma < 4.0:
        max_points = 7
    elif sigma < 4.3:
        max_points = 6
    else:
        max_points = 5
    points = np.random.randint(max_points - min_points + 1) + min_points
    # Stronger
    sigma *= crackup
    points *= crackup
    points = round(points)
    return (sigma, points)


if __name__ == "__main__":
    # for i in (1,2,3,4,5):
    #    make_nii_to_deformed_nii(f'/media/data/robert/test/2022_06_21_T1_CT_wopathfx/dataset/rawdata/fxclass000{i}/',
    #                            '/media/data/robert/test', f'fxclass000{i}')
    # make_nii_to_slice(False,'/media/data/robert/test/2022_06_21_T1_CT_wopathfx/dataset/rawdata/fxclass0002/',
    #                '/media/data/robert/test/pngs/', 'fxclass0002')
    j = 0
    for i in range(500):
        try:
            search_path = f"/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/rawdata/fxclass{i:04d}/"
            if os.path.exists(search_path):
                print("create fxclass", i, "deformation is", 0 == 0, "folder id is", 0)
                make_nii_to_slice(
                    False,
                    search_path,
                    f"/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/traindata/val/{0}/",
                    f"fxclass{i:04d}",
                    deform=True,
                )
        except Exception as e:
            print(i, e)
            pass

    for j in range(1, 10):

        for i in range(500):
            try:
                search_path = f"/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/rawdata/fxclass{i:04d}/"
                if os.path.exists(search_path):
                    print("create fxclass", i, "deformation is", j == 0, "folder id is", j)
                    make_nii_to_slice(
                        False,
                        search_path,
                        f"/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/traindata/train/{j}/",
                        f"fxclass{i:04d}",
                        deform=j != 0,
                    )
            except Exception as e:
                print(i, e)
                pass

    for j in range(10):
        break
        for i in range(500):
            try:
                search_path = f"/media/data/robert/test/2022_06_21_T1_CT_wopathfx/dataset/rawdata/fxclass{i:04d}/"
                if os.path.exists(search_path):
                    print("create fxclass", i, "deformation is", j == 0, "folder id is", j)
                    make_nii_to_slice(
                        True,
                        search_path,
                        f"/media/data/robert/test/2022_06_21_T1_CT_wopathfx/dataset/traindata/train_png/{j}/",
                        f"fxclass{i:04d}",
                        deform=j != 0,
                    )
            except Exception as e:
                print(i, e)
                pass

    for j in range(10):
        break
        for i in range(500):
            png = False
            try:
                search_path = f"/media/spine_data/MR_CT_forGAN/dataset_paper_TGD/TGD/[$FOLDER$]/train/{i}.nii.gz"
                if os.path.exists(search_path.replace("[$FOLDER$]", "ct")):
                    print(i)

                    # def make_nii_to_slice(png:bool,start_path:Path, out_path:Path, out_name:str, axis: int = 2 ,**kwargs):
                    out_path = f"/media/data/robert/datasets/TGD_2D_dash/train/{j}"
                    out_name = f"{i:04d}"
                    arr_dic, interpolation_lvl, normalize, filter, _ = setup_TGD(search_path, f"{i}", 2, filter="SG")
                    # print(filter)
                    if png:
                        make_np_to_PNG(
                            out_path,
                            out_name,
                            arr_dic=arr_dic,
                            interpolation_lvl=interpolation_lvl,
                            normalize=normalize,
                            filter=filter,
                            deform=False,
                        )
                    else:
                        make_np_to_npz(
                            out_path,
                            out_name,
                            arr_dic=arr_dic,
                            interpolation_lvl=interpolation_lvl,
                            normalize=normalize,
                            filter=filter,
                            deform=False,
                        )

            except Exception as e:
                print(i, e)
                pass
    for i in range(500):
        png = False
        break
        try:
            search_path = f"/media/spine_data/MR_CT_forGAN/dataset_paper_TGD/TGD/[$FOLDER$]/train/{i}.nii.gz"
            if os.path.exists(search_path.replace("[$FOLDER$]", "ct")):
                print(i)

                # def make_nii_to_slice(png:bool,start_path:Path, out_path:Path, out_name:str, axis: int = 2 ,**kwargs):
                out_path = f"/media/data/robert/datasets/TGD_2D_dash/val/{j}"
                out_name = f"{i:04d}"
                arr_dic, interpolation_lvl, normalize, filter, _ = setup_TGD(search_path, f"{i}", 2, filter="SG")
                # print(filter)
                if png:
                    make_np_to_PNG(
                        out_path,
                        out_name,
                        arr_dic=arr_dic,
                        interpolation_lvl=interpolation_lvl,
                        normalize=normalize,
                        filter=filter,
                        deform=False,
                    )
                else:
                    make_np_to_npz(
                        out_path,
                        out_name,
                        arr_dic=arr_dic,
                        interpolation_lvl=interpolation_lvl,
                        normalize=normalize,
                        filter=filter,
                        deform=False,
                    )

        except Exception as e:
            print(i, e)
            pass
