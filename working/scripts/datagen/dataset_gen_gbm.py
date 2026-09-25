from genericpath import isdir
import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz

image_types = ["T1", "T2", "T1GD", "FLAIR"]


def setup_gbm(patient_folder: Path):
    arr_dic: dict[str, np.ndarray] = {}
    normalize = {}
    interpolation_lvl = {}
    for t in image_types:
        arr, affine = load_nii(Path(patient_folder, f"{patient_folder.name}_{t}.nii.gz"))
        arr_dic[t] = arr.swapaxes(0, 2)
        normalize[t] = lambda x, volumes: x / max(0.00001, np.max(volumes))
        interpolation_lvl[t] = 3
        # value = value.transpose()
        # arr_dic[key] = value.swapaxes(0, axis).swapaxes(1, 2)[:, ::-1, ::-1].copy()
    # Brain_mask (replacement)
    arr_dic["SG"] = arr_dic["T1"].copy()
    for i in range(arr_dic["SG"].shape[0]):
        slice_ = arr_dic["SG"][i]
        slice_[slice_ != 0] = 1
        if slice_.sum() <= 1000:
            arr_dic["SG"][i] *= 0

    normalize["SG"] = normalize["T1"]
    interpolation_lvl["SG"] = interpolation_lvl["T1"]
    return arr_dic, interpolation_lvl, normalize, "SG"


if __name__ == "__main__":
    train_c = 0
    val_c = 0
    test_c = 0
    idx = 0
    png = False
    for search_path in ["/media/data/robert/datasets/gbm/"]:
        try:
            search_path = Path("/media/data/robert/datasets/gbm/images_structural/")
            # print(df[["Name", "Qualität"]])

            for patient_folder in sorted(search_path.iterdir()):
                if patient_folder is None:
                    continue
                elif not patient_folder.is_dir():
                    continue
                name = patient_folder.name
                seg_path = Path(search_path.parent, "images_segm").glob(name + "*")
                if next(seg_path, None) is not None:
                    print(name, "--> test set")
                    test_c += 1
                    continue
                del seg_path
                print(len(list(patient_folder.iterdir())))
                if len(list(patient_folder.iterdir())) < 4:
                    continue
                idx += 1

                is_validation = idx % 12 == 0

                if png:
                    out_path = "/media/data/robert/datasets/gbm/img/org"

                elif not is_validation:
                    out_path = f"/media/data/robert/datasets/gbm/train/"
                    train_c += 1
                else:
                    val_c += 1
                    out_path = "/media/data/robert/datasets/gbm/val/"

                print(name, "val" if is_validation else "train")

                arr_dic, interpolation_lvl, normalize, filter = setup_gbm(patient_folder)
                print(filter)
                if png:
                    make_np_to_PNG(
                        out_path,
                        name,
                        arr_dic=arr_dic,
                        interpolation_lvl=interpolation_lvl,
                        normalize=normalize,
                        filter=filter,
                        deform=False,
                        crackup=0.8,
                        single_png=True,
                        # crop3D="SG",
                    )
                else:
                    make_np_to_npz(
                        out_path,
                        name,
                        arr_dic=arr_dic,
                        interpolation_lvl=interpolation_lvl,
                        normalize=normalize,
                        filter=filter,
                        deform=False,
                        crackup=0.8,
                    )

        except Exception as e:
            raise e
        print(train_c, val_c, test_c)
