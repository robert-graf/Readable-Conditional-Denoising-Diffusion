from __future__ import annotations
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

import torch
from diffusion import Diffusion
from dataloader.dataloader_nii_seg import nii_Dataset, nii_Datasets

import subprocess

from loader.arguments import get_latest_Checkpoint

###### Model
name = "bailiang_256"
version: str = "*"
ddim = True
guidance_w = 1
test = False
skip_generation = False
skip_segmentation = False
dont_segment_org_ct = False
model = None
# spine_data/CT_fxclass/dataset-fxclass
root = "/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/"

result_folder = "ML_translation"
if not skip_generation:
    checkpoint = get_latest_Checkpoint(name, log_dir_name="logs_diffusion", best=False, version=version)
    assert checkpoint is not None
    model = Diffusion.load_from_checkpoint(checkpoint, strict=False)
    model.cuda()

#### Dataset
dataset = nii_Datasets(root=root + "dataset_org/rawdata/fxclass*/*/*T1*")

Path(f"{root}/{result_folder}/rawdata/").mkdir(exist_ok=True, parents=True)
####### Loop
file_names = []
with torch.no_grad():
    for i, nii_ds in enumerate(dataset):  # type: ignore

        continue
        # if not test and i == 0:
        #    continue
        ## remember file names for dice
        if not Path(nii_ds.file).stem.endswith("_T1.nii"):
            continue
        sub = Path(nii_ds.file).stem.replace("_", "-").split("-")[1]
        print(sub)
        # {Path(nii_ds.file).stem.replace(".nii","")}
        file_ct_ml = f"{root}/{result_folder}/rawdata/{sub}/sub-{sub}_sequ-1_ct.nii.gz"
        file_t1 = f"{root}/{result_folder}/rawdata/{sub}/sub-{sub}_sequ-1_T1c.nii.gz"
        file_ct = f"{root}/{result_folder}/rawdata/{sub}/sub-{sub}_sequ-2_ct_.nii.gz"  #
        if not dont_segment_org_ct:
            file_ct.replace("ct_.nii.gz", "ct.nii.gz")
        file_ct_subreg = f"{root}/{result_folder}/derivatives/{sub}/sub-{sub}_sequ-2_seg-subreg_msk.nii.gz"  #
        file_ct_vert = f"{root}/{result_folder}/derivatives/{sub}/sub-{sub}_sequ-2_seg-vert_msk.nii.gz"  #
        Path(file_ct_ml).parent.mkdir(exist_ok=True)
        Path(file_ct_subreg).parent.mkdir(exist_ok=True, parents=True)
        file_names.append((sub, file_ct_ml, file_t1, file_ct, file_ct_subreg, file_ct_vert))
        import shutil

        f = Path(nii_ds.file).parent.parent
        try:
            src = next(f.glob("*/*_ct.nii.gz"))
            shutil.copyfile(src, file_ct) if not Path(file_ct).exists() else None
            if dont_segment_org_ct:
                del_me = Path(file_ct.replace("ct_.nii.gz", "ct.nii.gz"))
                del_me.unlink() if del_me.exists() else None

            src = next(f.glob("*/*_T1.nii.gz"))
            shutil.copyfile(src, file_t1) if not Path(file_t1).exists() else None
        except StopIteration:
            continue
        f = Path(str(f).replace("rawdata", "derivatives"))
        try:
            src = next(f.glob("*/*_seg-vert_msk.nii.gz"))
            shutil.copyfile(src, file_ct_vert) if not Path(file_ct_vert).exists() else None
        except StopIteration:
            pass
        try:
            src = next(f.glob("*/*_seg-subreg_msk.nii.gz"))
            shutil.copyfile(src, file_ct_subreg) if not Path(file_ct_subreg).exists() else None
        except StopIteration:
            pass
        if Path(file_ct_ml).exists():
            continue
        if skip_generation:
            if test:
                break
            continue
        assert model is not None
        nii_ds: nii_Dataset
        from_idx = 0
        to_idx = 1000

        # a = (nii_ds.nii.shape[-1] - 256) // 2
        # b = (nii_ds.nii.shape[-2] - 256) // 2
        # crop = (b, a, 256, 256)
        # if a == 0 and b == 0:
        crop = 256

        arr = nii_ds.get_copy(crop=crop)  # [from_idx:to_idx]

        # Normalize
        arr /= arr.max()
        # seg /= seg.max()
        cond = (arr * 2 - 1).unsqueeze(1).cuda()

        if cond.shape[0] <= 55:
            if not ddim:
                out = model.forward(cond.shape[0], x_conditional=cond, guidance_w=guidance_w)
            else:
                out = model.forward_ddim(cond.shape[0], range(0, 1000, 20), x_conditional=cond, w=guidance_w, eta=1.0)
                out = out[0]
        else:
            c_a, c_b = torch.split(cond, cond.shape[0] // 2 + 1, dim=0)
            if not ddim:
                out1 = model.forward(c_a.shape[0], x_conditional=c_a, guidance_w=guidance_w)
                out2 = model.forward(c_b.shape[0], x_conditional=c_b, guidance_w=guidance_w)
            else:
                out1 = model.forward_ddim(c_a.shape[0], range(0, 1000, 20), x_conditional=c_a, w=guidance_w, eta=1.0)[0]
                out2 = model.forward_ddim(c_b.shape[0], range(0, 1000, 20), x_conditional=c_b, w=guidance_w, eta=1.0)[0]
            out = torch.cat((out1, out2), dim=0)

        assert isinstance(out, torch.Tensor)
        out_save = nii_ds.insert_sub_corp(
            out.squeeze(1) * 2000 - 1000, offset=crop[:2] if not isinstance(crop, int) else None, fill_value=-1000
        ).numpy()

        ### Save CT image
        nii_ds.save(out_save, file_ct_ml, revert_to_size=True)  #
        del out_save
        del arr
        del cond

        if test:
            break

#### Segment
if not skip_segmentation:
    process = subprocess.Popen(
        [
            "docker",
            "run",
            "-v",
            f"{root}/{result_folder}:/data",
            "anjanys/anduin:parallel",
            "/process_spine_batch_parallelised",
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    print(str(stdout))
##### img

from BIDS.wrapper.snapshot_modular import Snapshot_Frame, create_snapshot

for sub, file_ct_ml, file_t1, file_ct, file_ct_subreg, file_ct_vert in file_names:
    file_seg_ml = str(Path(file_ct_ml)).replace("rawdata", "derivatives").replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")
    a = [
        Snapshot_Frame(image=file_t1, segmentation=file_seg_ml, centroids=None, coronal=True, axial=True, mode="MRI"),
        Snapshot_Frame(image=file_ct_ml, segmentation=file_seg_ml, centroids=None, coronal=True, axial=True, mode="CT"),
        Snapshot_Frame(image=file_ct, segmentation=file_ct_vert, centroids=None, coronal=True, axial=True, mode="CT"),
    ]
    folder = Path((Path(file_ct_ml).parent.parent.parent), "snapshot")
    folder.mkdir(exist_ok=True)
    try:
        create_snapshot(f"{str(folder)}/{sub}.jpg", a, crop=True)
    except Exception:
        print("failed snapshot")
