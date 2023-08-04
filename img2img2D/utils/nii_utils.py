from __future__ import annotations
import numbers
from pathlib import Path
import typing
import numpy as np
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
from scipy.ndimage import center_of_mass
from scipy.ndimage import binary_erosion, generate_binary_structure
import json

from typing import TYPE_CHECKING, Optional, TypedDict, Union, Tuple

# import bids_files

if TYPE_CHECKING:
    # from bids_files import BIDS_FILE
    from typing import TypeGuard

    Image_Reference = Union[nib.Nifti1Image, Path, str]
    Centroid_Reference = Union[Path, str, dict[str, str | Path]]

# fmt: off
v_idx2name = {
     1: "C1",     2: "C2",     3: "C3",     4: "C4",     5: "C5",     6: "C6",     7: "C7", 
     8: "T1",     9: "T2",    10: "T3",    11: "T4",    12: "T5",    13: "T6",    14: "T7",    15: "T8",    16: "T9",    17: "T10",   18: "T11",   19: "T12",
    20: "L1",    21: "L2",    22: "L3",    23: "L4",    24: "L5",    25: "L6",    
    26: "Sacrum",27: "Cocc",  28: "T13",
}  
# fmt: on


def crop_slice(msk, dist=20):
    shp = msk.dataobj.shape
    zms = msk.header.get_zooms()
    d = np.around(dist / np.asarray(zms)).astype(int)
    msk_bin = np.asanyarray(msk.dataobj, dtype=bool)
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0] - d[0] if (c_min[0] - d[0]) > 0 else 0
    y0 = c_min[1] - d[1] if (c_min[1] - d[1]) > 0 else 0
    z0 = c_min[2] - d[2] if (c_min[2] - d[2]) > 0 else 0
    x1 = c_max[0] + d[0] if (c_max[0] + d[0]) < shp[0] else shp[0]
    y1 = c_max[1] + d[1] if (c_max[1] + d[1]) < shp[1] else shp[1]
    z1 = c_max[2] + d[2] if (c_max[2] + d[2]) < shp[2] else shp[2]
    ex_slice = tuple([slice(x0, x1), slice(y0, y1), slice(z0, z1)])
    origin_shift = tuple([x0, y0, z0])
    return ex_slice, origin_shift


def crop_centroids(ctd_list: Centroid_List, o_shift):
    for idx, v in enumerate(ctd_list[1:]):
        assert is_Point3D_tuple(v)
        v = list(v)
        v[1] = v[1] - o_shift[0]
        v[2] = v[2] - o_shift[1]
        v[3] = v[3] - o_shift[2]
        ctd_list[idx] = tuple(v)  # type: ignore
    return ctd_list


def crop_centroids_cdl(ctd_list: Centroid_DictList, o_shift):
    for v in ctd_list[1:]:  # type: ignore
        v: Point3D
        v["X"] = v["X"] - o_shift[0]
        v["Y"] = v["Y"] - o_shift[1]
        v["Z"] = v["Z"] - o_shift[2]
    return ctd_list


#########################
# Resample and reorient #


def nii2arr(img: nib.Nifti1Image) -> np.ndarray:
    return np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)


def reorient_same_as(img: nib.Nifti1Image, img_as: nib.Nifti1Image, verbose=False, return_arr=False) -> nib.Nifti1Image:
    axcodes_to = nio.ornt2axcodes(nio.io_orientation(img_as.affine))
    return reorient_to(img, axcodes_to, verbose, return_arr)


def reorient_to(img, axcodes_to=("P", "I", "R"), verbose=False, return_arr=False) -> nib.Nifti1Image:
    # Note: nibabel axes codes describe the direction not origin of axes
    # direction PIR+ = origin ASL
    aff = img.affine
    arr = nii2arr(img)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    new_aff = np.matmul(aff, aff_trans)
    new_img = nib.Nifti1Image(arr, new_aff)
    if verbose:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    if return_arr:
        return new_img, arr.copy()  # type: ignore
    return new_img


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3, c_val=-1024, verbose=False) -> nib.Nifti1Image:
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(
        np.rint([shp[0] * zms[0] / voxel_spacing[0], shp[1] * zms[1] / voxel_spacing[1], shp[2] * zms[2] / voxel_spacing[2]]).astype(int)
    )
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)  # type: ignore
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=c_val)
    if verbose:
        print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_nib_4d(img, voxel_spacing=(1, 1, 1, 1), order=3):
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(
        np.rint([shp[0] * zms[0] / voxel_spacing[0], shp[1] * zms[1] / voxel_spacing[1], shp[2] * zms[2] / voxel_spacing[2]]).astype(int)
    )
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)  # type: ignore
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_mask_to(msk, to_img):
    to_img.header["bitpix"] = 8
    to_img.header["datatype"] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)  # type: ignore
    print("[*] Mask resampled to image size:", new_msk.header.get_data_shape())
    return new_msk


def get_plane(img_path: Image_Reference):
    plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
    img = to_nii(img_path)
    axc = np.array(nio.aff2axcodes(img.affine))
    zms = np.around(get_header(img).get_zooms(), 1)
    ix_max = np.array(zms == np.amax(zms))
    num_max = np.count_nonzero(ix_max)
    if num_max == 2:
        plane = plane_dict[axc[~ix_max][0]]
    elif num_max == 1:
        plane = plane_dict[axc[ix_max][0]]
    else:
        plane = "iso"
    return plane


######################
# Handling centroids #

Point3D = TypedDict("Point3D", X=float, Y=float, Z=float, label=int)
Orientation = TypedDict("Orientation", direction=Tuple[str, str, str])
Centroid_List = list[typing.Union[Tuple[str, str, str], Tuple[int, float, float, float]]]
Centroid_DictList = list[Union[Point3D, Orientation]]
Centroid_Dict = dict[int, Tuple[float, float, float]]


def is_Point3D(obj) -> TypeGuard[Point3D]:
    return "label" in obj and "X" in obj and "Y" in obj and "Z" in obj


def is_Orientation(obj) -> TypeGuard[Orientation]:
    return "direction" in obj


def is_Orientation_tuple(obj) -> TypeGuard[Tuple[str, str, str]]:
    return isinstance(obj, tuple) and len(obj) == 3 and isinstance(obj[0], str) and isinstance(obj[1], str) and isinstance(obj[2], str)


def is_Point3D_tuple(obj) -> TypeGuard[Tuple[int, float, float, float]]:
    return len(obj) == 4 and all([isinstance(i, numbers.Number) for i in obj])


def get_header(nii: nib.Nifti1Image) -> nib.Nifti1Header:
    return nii.header  # type: ignore


def load_centroids(ctd_path: Centroid_Reference) -> Centroid_List:
    if isinstance(ctd_path, dict):
        ctd_path = ctd_path["ctd"]
    # if isinstance(ctd_path, bids_files.BIDS_FILE):
    #    dict_list = ctd_path.open_json()
    # else:
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    # print(dict_list)
    for d in dict_list:
        if "direction" in d:
            ctd_list.append(tuple(d["direction"]))
        elif "nan" in str(d):  # skipping NaN centroids
            continue
        else:
            ctd_list.append([d["label"], d["X"], d["Y"], d["Z"]])
    return ctd_list


def centroids_to_dict_list(ctd_list: Centroid_List) -> Centroid_DictList:
    dict_list = []
    for v in ctd_list:
        if any("nan" in str(v_item) for v_item in v):
            continue  # skipping invalid NaN values
        v_dict = {}
        if is_Orientation_tuple(v):
            v_dict["direction"] = v
        elif is_Point3D_tuple(v):
            v_dict["label"] = int(v[0])
            v_dict["X"] = v[1]
            v_dict["Y"] = v[2]
            v_dict["Z"] = v[3]
        else:
            assert False, str(v) + "is not a valid data-structure"
        dict_list.append(v_dict)
    return dict_list


def centroid_dictList_to_centroid_list(ctd_dict_list: Centroid_DictList) -> Centroid_List:
    ctd_list: Centroid_List = []
    for v in ctd_dict_list:
        if is_Orientation(v):
            assert len(ctd_list) == 0
            ctd_list.append(v["direction"])
        elif is_Point3D(v):
            assert len(ctd_list) != 0
            ctd_list.append((v["label"], v["X"], v["Y"], v["Z"]))
        else:
            assert False, str(v) + "is not a valid object for ctd_dict_list"
    return ctd_list


def to_dict(ctd_list: Centroid_DictList | Centroid_List) -> Centroid_Dict:
    v_dict = {}
    for v in ctd_list:
        if any("nan" in str(v_item) for v_item in v):
            continue  # skipping invalid NaN values
        try:
            if is_Orientation_tuple(v):
                continue
            elif is_Point3D_tuple(v):
                v_dict[v[0]] = [v[1], v[2], v[3]]
            elif is_Orientation(v):
                continue
            elif is_Point3D(v):
                v_dict[v["label"]] = [v["X"], v["Y"], v["Z"]]
        except Exception as e:
            print(v)
            raise e

    return v_dict


def save_centroids(ctd_list: Centroid_List, out_path: Path | str) -> None:
    out_path = str(out_path)
    if len(ctd_list) < 2:
        print("[#] Centroids empty, not saved:", out_path)
        return
    json_object = centroids_to_dict_list(ctd_list)
    # Problem with python 3 and int64 serialization.
    def convert(o):
        if isinstance(o, np.int64):  # type: ignore
            return int(o)
        raise TypeError

    with open(out_path, "w") as f:
        json.dump(json_object, f, default=convert)
    print("[*] Centroids saved:", out_path)


def calc_centroids_from_file(path: Path, out_path: Path) -> Centroid_DictList:
    print("[*] Generate ctd json from _msk.nii.gz")
    msk_nii = nib.load(str(path))
    ctd_list = calc_centroids(msk_nii, decimals=2)
    ctd_dict = centroids_to_dict_list(ctd_list)
    save_json(ctd_dict, str(out_path))
    return ctd_dict


def to_nii_optional(img_bids: Optional[Image_Reference]) -> Optional[nib.Nifti1Image]:
    if img_bids is None:
        return None
    return to_nii(img_bids)


def to_nii(img_bids: Image_Reference) -> nib.Nifti1Image:
    assert img_bids is not None

    # if isinstance(img_bids, bids_files.BIDS_FILE):
    #    return img_bids.open_nii()
    # el
    if isinstance(img_bids, Path):
        return nib.load(str(img_bids))
    elif isinstance(img_bids, str):
        return nib.load(img_bids)
    else:
        return img_bids


def calc_centroids_from_subreg_vert(
    vert_msk, subreg, decimals=1, world=False, subreg_id=50, axcodes_to=None, verb=False, fixed_offset=0
) -> Centroid_List:
    # Centroids are in voxel coordinates!
    # world=True: centroids are in world coordinates
    vert_msk = to_nii(vert_msk)
    subreg = to_nii(subreg)

    if axcodes_to is not None:
        # Like: ("P","I","R")
        vert_msk = reorient_to(vert_msk, verbose=verb, axcodes_to=axcodes_to)
        assert not isinstance(vert_msk, tuple)
        subreg = reorient_to(subreg, verbose=verb, axcodes_to=axcodes_to)
        assert not isinstance(subreg, tuple)
    vert = nii2arr(vert_msk)
    subreg = nii2arr(subreg)

    vert[subreg != subreg_id] = 0
    msk_data = vert
    nii = nip.Nifti1Image(msk_data, vert_msk.affine, vert_msk.header)
    ctd_list = calc_centroids(nii, decimals=decimals, world=world, fixed_offset=fixed_offset)
    return ctd_list  # rescale_centroids(ctd_list,nii,voxel_spacing=voxel_spacing)


def calc_centroids(msk: nib.Nifti1Image, decimals=1, world=False, fixed_offset=0) -> Centroid_List:
    # Centroids are in voxel coordinates!
    # world=True: centroids are in world coordinates

    # if isinstance(msk, bids_files.BIDS_FILE):
    #    msk = msk.open_nii()

    msk_data = np.asanyarray(msk.dataobj, dtype=msk.dataobj.dtype)
    axc = nio.aff2axcodes(msk.affine)
    ctd_list: Centroid_List = [axc]
    verts = np.unique(msk_data)[1:]
    verts = verts[~np.isnan(verts)]  # remove NaN values
    for i in verts:
        msk_temp = np.zeros(msk_data.shape, dtype=bool)
        msk_temp[msk_data == i] = True
        ctr_mass = center_of_mass(msk_temp)
        if world:
            ctr_mass = msk.affine[:3, :3].dot(ctr_mass) + msk.affine[:3, 3]
            ctr_mass = ctr_mass.tolist()
        ctd_list.append(tuple([int(i + fixed_offset)] + [round(x, decimals) for x in ctr_mass]))  # type: ignore
    return ctd_list


def reorient_centroids_to(ctd_list: Centroid_List, img, decimals=1, verb=False) -> Centroid_List:
    # reorient centroids to image orientation
    # todo: reorient to given axcodes (careful if img ornt != ctd ornt)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present")
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list: Centroid_List = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append(tuple([v, *ctd]))
    if verb:
        print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list


def rescale_centroids(ctd_list: Centroid_List, img, voxel_spacing=(1, 1, 1)) -> Centroid_List:
    # rescale centroid coordinates to new spacing in current x-y-z-orientation
    ornt_img = nio.io_orientation(img.affine)
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)  # type: ignore
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] * zms[0] / voxel_spacing[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] * zms[1] / voxel_spacing[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] * zms[2] / voxel_spacing[2], decimals=1)
    out_list: Centroid_List = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append(tuple([v, *ctd]))
    print("[*] Rescaled centroid coordinates to spacing (x, y, z) =", voxel_spacing, "mm")
    return out_list


def rescale_dockerctd_to(ctd_list: Centroid_List, img):
    # rescale and reorient docker output centroid coordinates to img coordinates
    ornt_img = nio.io_orientation(img.affine)
    ctd_list.insert(0, ("I", "P", "L"))
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)  # type: ignore
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] / zms[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] / zms[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] / zms[2], decimals=1)
    out_list: Centroid_List = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append(tuple([v, *ctd]))
    if not np.array_equal(ornt_img, ornt_ctd):
        out_list = reorient_centroids_to(out_list, img)
    return out_list


def rescale_aslctd_to(ctd_list: Centroid_List, img):
    # rescale and reorient 1mm ASL centroids coordinates to img coordinates
    ornt_img = nio.io_orientation(img.affine)
    if str(ctd_list[0][0]).isnumeric():  # either insert direction or change wrongly indicated direction
        ctd_list.insert(0, ("P", "I", "R"))
    else:
        ctd_list[0] = ("P", "I", "R")
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)  # type: ignore
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] / zms[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] / zms[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] / zms[2], decimals=1)
    out_list: Centroid_List = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append(tuple([v, *ctd]))
    if not np.array_equal(ornt_img, ornt_ctd):
        out_list = reorient_centroids_to(out_list, img)
    return out_list


def nii_2_axcode(nii: Image_Reference) -> Orientation:
    tmp_nii = to_nii(nii)
    axcodes_to: Orientation = nio.aff2axcodes(tmp_nii.affine)  # type: ignore
    return axcodes_to


#########################
# Nifti mask processing #


def erode_msk(msk, mm=5):
    # msk_data[np.isnan(msk_data)] = 0 # remove NaN
    struct = generate_binary_structure(3, 1)
    msk_i = resample_nib(msk, (1, 1, 1), 0)
    msk_i_data = np.asanyarray(msk_i.dataobj, dtype=msk_i.dataobj.dtype)
    msk_ibe_data = binary_erosion(msk_i_data, structure=struct, iterations=mm)
    msk_i_data[~msk_ibe_data] = 0  # type: ignore
    msk_ie = nib.Nifti1Image(msk_i_data.astype(np.uint8), msk_i.affine)
    msk_e = resample_mask_to(msk_ie, msk)
    print("[*] Mask eroded by", mm, "mm")
    return msk_e


def map_labels(img, label_map):
    # label_map is a dictionary mapping from(str)->to(int)
    # eg: label_map = {'40': 40, '50': 41, '51': 42, '52': 43, '53': 44}
    data = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    for v in np.unique(data):
        if v > 0 and str(int(v)) in label_map.keys():  # int needed to match non-integer data-types
            print("changing label", v, "to", label_map[str(int(v))])
            data[data == v] = label_map[str(int(v))]
    print("[*] N =", len(label_map), "labels reassigned")
    return nib.Nifti1Image(data.astype(np.uint8), img.affine)


def get_subreg_msk(msk, srg, s_label=40):
    # reduce vertebral mask to subregion
    msk_data = np.asanyarray(msk.dataobj, dtype=np.uint8)
    srg_data = np.asanyarray(srg.dataobj, dtype=np.uint8)
    out = msk_data.copy()
    out[srg_data != s_label] = 0
    # print("[*] Subregion mask of label", s_label, "created")
    return nib.Nifti1Image(out, msk.affine)


def reduce_labels(msk1, labels=[20, 21, 22, 23]):
    msk1_data = np.asanyarray(msk1.dataobj, dtype=msk1.dataobj.dtype)
    for v in np.unique(msk1_data)[1:]:
        if v not in labels:
            msk1_data[msk1_data == v] = 0
    return nib.Nifti1Image(msk1_data, msk1.affine)


def save_json(out_dict, out_path):
    # NB: dict keys that are not of a basic type (str, int, float, bool, None)
    # cannot be serialized and need to be converted be default=function
    def convert(o):
        if isinstance(o, np.int64):  # type: ignore
            return int(o)
        if isinstance(o, np.uint16):  # type: ignore
            return int(o)
        print("Unknown type in save_json:", type(o))
        raise TypeError

    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=4, default=convert)
    print("[*] Json saved:", Path(out_path).name)
