import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz

from numpy.typing import NDArray
from BIDS import (
    BIDS_Global_info,
    Subject_Container,
    BIDS_FILE,
    to_nii,
    load_centroids,
    calc_centroids_labeled_buffered,
    Location,
)
from BIDS.registration.ridged_points import reg_segmentation
from joblib import Parallel, delayed

import pandas as pd
from functools import partial


def setup_reg_affine(start_path: Path, key: str, sequ: str, axis: int, filter=None):

    folder = sequ
    dixon = None
    ct = None
    seg = None

    for p in Path(start_path, "registration", key, folder).iterdir():
        if p.name.endswith("dixon.nii.gz"):
            dixon = p
        elif p.name.endswith("ct.nii.gz"):
            ct = p
        elif p.name.endswith("seg-vert_msk.nii.gz"):
            seg = p
    #
    ct_arr = to_nii(ct).reorient_(("R", "I", "P"))
    mi_arr = to_nii(dixon).reorient_(("R", "I", "P"))
    sg_arr = to_nii(seg).reorient_(("R", "I", "P"))
    arr_dic = {"CT": ct_arr.get_array(), "T2": mi_arr.get_array(), "SG": sg_arr.get_array()}
    affine_dic = {"CT": ct_arr.affine, "T2": mi_arr.affine, "SG": sg_arr.affine}

    interpolation_lvl = {"CT": 3, "T2": 3, "SG": 0}
    normalize = {
        "CT": lambda x, volumes: (x + 1000) / 2000,
        "T2": lambda x, volumes: x / max(0.00001, np.max(volumes)),
        "SG": lambda x, volumes: x,
    }

    return arr_dic, interpolation_lvl, normalize, filter, affine_dic


def main(n_jobs=12, png=False, reg_ds_path="/media/data/robert/datasets/spinegan_T2w_all_reg_iso/"):

    Parallel(n_jobs=n_jobs)(delayed(__main)(png, j, reg_ds_path) for j in range(1 if png else 12))


def __main(png, j, reg_ds_path="/media/data/robert/datasets/spinegan_T2w_all_reg_iso/"):
    for search_path in [reg_ds_path]:

        try:
            search_path = Path(search_path)
            # print(search_path)
            if j < 10:
                split = "train"
            elif j == 10:
                split = "val"
            elif j == 11:
                split = "test"
            else:
                assert False
            df = pd.read_csv(str(Path(reg_ds_path, split + ".csv")))
            if png:
                out_path = Path(reg_ds_path, "deform")
            elif j == 10:
                out_path = Path(reg_ds_path, "val")
            elif j == 11:
                out_path = Path(reg_ds_path, "test")
            else:
                out_path = Path(reg_ds_path, f"train/{j}/")
            for sub_path in list(df["name"]):
                key, sequ = str(sub_path).split("/", 1)
                try:
                    arr_dic, interpolation_lvl, normalize, filter, _ = setup_reg_affine(
                        search_path, key, sequ, 2, filter="SG"
                    )
                except ValueError as e:
                    print("file not found", e)
                    continue
                except FileNotFoundError as e:
                    print("file not found", e)
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


# Save registered file
def generate_file_path_reg(out, file: BIDS_FILE, other_file_id, folder, keys=["sub", "ses", "sequ", "reg", "acq"]):
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
        dataset_path=f"/media/data/robert/datasets/{out}/",
        file_type="nii.gz",
        parent="registration",
        path="{sub}/" + folder,
        info=info,
        from_info=True,
    )
    return out_file


def register(n_jobs=10):
    global_info = BIDS_Global_info(
        [
            # "/media/data/robert/datasets/spinegan_T2w/sourcedata",
            # "/media/data/robert/datasets/spinegan_T2w/translation/result_exp_syndiff_t2w/",
            "/media/data/robert/datasets/spinegan_T2w_all_reg_2/translation/result_exp_syndiff_t2w/",
            "/media/data/robert/datasets/spine_transition_new/",
        ],
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
    query.filter("seg", "vert")
    # sub-spinegan0001_ses-20220527_sequ-302_e-1_seg-subreg_ctd.json
    # print(query)
    # query.filter("label", "procspin")
    d = None
    for bids_file in query.loop_dict():
        # Do something with the files.
        # print(bids_file)
        print(bids_file.get_key_len())
        dixon = bids_file["dixon"]
        msk = bids_file["msk_vert"][0]
        # procspin = bids_file["msk_procspin"][0] if "msk_procspin" in bids_file else None
        e = msk.get("e")
        vert_dixon = bids_file["msk_vert"][0]
        subreg_dixon = bids_file["msk_subreg"][0]
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
        query.flatten()
        query.filter_non_existence("e")
        query.unflatten()
        for bids_file in query.loop_dict():
            try:
                assert len(bids_file["ct"]) == 1, bids_file["ct"]
                assert len(bids_file["msk_vert"]) == 1, bids_file["msk_vert"]
                # assert len(bids_file["msk"]) == 1, bids_file
                ct = bids_file["ct"][0]
                vert = bids_file["msk_vert"][0]
                subreg = bids_file["msk_subreg"][0]
                print(bids_file.get_key_len())
                ctd_mr = calc_centroids_labeled_buffered(
                    vert_dixon,
                    subreg_dixon,
                    subreg_id=list(range(40, 51)),
                )

                ctd_ct = calc_centroids_labeled_buffered(
                    msk_reference=vert, subreg_reference=subreg, subreg_id=list(range(40, 51)), override=False
                )
                if ctd_ct.shape is None:
                    print("override")
                    ctd_ct = calc_centroids_labeled_buffered(
                        msk_reference=vert, subreg_reference=subreg, subreg_id=list(range(40, 51)), override=True
                    )

                reg_segmentation.ridged_registration_by_dict_from_ctd(
                    [d, msk],
                    [ct, vert],
                    ctd_mr,
                    ctd_ct,
                    generate_file_names=partial(generate_file_path_reg, "spinegan_T2w_all_reg_2"),
                    axcodes_to=("R", "I", "P"),
                    voxel_spacing=(-1, 1, 1),
                    snap_shot_folder="/media/data/robert/datasets/spinegan_T2w_all_reg_2/",
                )
            except ValueError as e:
                print(e)
            # except IndexError as e:
            #    print(e)
            except Exception as e:
                print(e)
                raise e


if __name__ == "__main__":
    main()
    exit()
    register(n_jobs=6)

    from BIDS.snapshot2D import Snapshot_Frame, create_snapshot
    from BIDS import NII

    data_set_name = "spinegan_T2w_all_reg_iso"
    for i in Path(f"/media/data/robert/datasets/{data_set_name}/registration").iterdir():
        for session in i.glob("*"):
            if len(list(session.iterdir())) != 0:
                ct = next(session.glob("*_ct.nii.gz"))
                seg = next(session.glob("*_seg-vert_msk.nii.gz"))
                dixon_f = next(session.glob("*_dixon.nii.gz"))
                s = dixon_f.parent.parent.name + "_" + dixon_f.parent.name
                ct = NII.load(ct, False)
                dixon = NII.load(dixon_f, False)
                arr = dixon.get_array()

                f_arr = ct.get_array()
                f_arr[f_arr != 0] = 2
                infect_list = []

                def infect(x, y, z):
                    if any([x < 0, y < 0, z < 0, x == f_arr.shape[0], y == f_arr.shape[1], z == f_arr.shape[2]]):
                        return
                    if f_arr[x, y, z] == 0:
                        f_arr[x, y, z] = 1
                        for a, b, c in [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ]:
                            infect_list.append((x + a, y + b, z + c))

                    else:
                        pass

                infect(0, 0, 0)
                infect(f_arr.shape[0] - 1, 0, 0)
                infect(0, f_arr.shape[1] - 1, 0)
                infect(f_arr.shape[0] - 1, f_arr.shape[1] - 1, 0)

                infect(0, 0, f_arr.shape[2] - 1)
                infect(f_arr.shape[0] - 1, 0, f_arr.shape[2] - 1)
                infect(0, f_arr.shape[1] - 1, f_arr.shape[2] - 1)
                infect(f_arr.shape[0] - 1, f_arr.shape[1] - 1, f_arr.shape[2] - 1)
                while len(infect_list) != 0:
                    infect(*infect_list.pop())
                f_arr[f_arr == 0] = 2
                f_arr -= 1
                dixon.set_array_(f_arr * arr)
                dixon.save(dixon_f)
                frames = [Snapshot_Frame(ct, seg, mode="CT"), Snapshot_Frame(dixon, seg, mode="MRI")]
                Path(f"/media/data/robert/datasets/{data_set_name}/snapshot-registration2/").mkdir(exist_ok=True)
                try:
                    create_snapshot(
                        f"/media/data/robert/datasets/{data_set_name}/snapshot-registration2/{s}.jpg", frames=frames
                    )
                except Exception as e:
                    print("[!]", e)

        # create_snapshot(){'lr': 2e-05, 'batch_size': 8, 'max_epochs': 500, 'num_cpu': 16, 'gpus': [3], 'new': False}

    # main()
    pass
