from __future__ import annotations
from pathlib import Path
import pickle
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))


from dataloader.Wrapper_datasets import Wrapper_Image2Image

from torch.utils.data import DataLoader

# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_pix2pix --new -bs 16
# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_cut_sa-unet --new -bs 16
# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_pcut_sa-unet --new -bs 16

# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_cut --new --factor 1 -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_cut_sa-unet --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_pix2pix_sa-unet --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_pix2pix --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --cut -en paper_T1_pcut_sa-unet --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --syndiff -en exp_syndiff --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --ddim --eta 0 -t 20 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --ddim --eta 1 -t 20 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --ddim --eta 0 -t 50 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --ddim --eta 1 -t 50 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --ddpm               -en paper_T1_diffusion --new -bs 24 --factor 10


# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_cut --new --factor 1 -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_cut_sa-unet --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pix2pix_sa-unet --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pix2pix --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pcut_sa-unet --new -bs 8 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --syndiff -en exp_syndiff_t2w --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 0 -t 20 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 1 -t 20 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 0 -t 50 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 1 -t 50 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality_fid.py --root /media/data/robert/datasets/spinegan_T2w --ddpm               -en paper_T2_diffusion --new -bs 24 --factor 10


import torch
from dataloader.dataloader_mri2ct import Wopathfx_Dataset


from scipy import ndimage
import numpy as np
import reload_any

parser = reload_any.get_option_reload()
parser = reload_any.get_option_test(parser)
opt = reload_any.get_option(parser)
print(opt)
exp_name = opt.exp_name
version: str = opt.version
ddim = opt.ddim
cut = opt.cut
guidance_w = opt.guidance_w
test = opt.test
root = opt.root
do_reload = not opt.new

###### Model
model, checkpoint = reload_any.get_model(opt)

pickle_file = Path(checkpoint).parent.parent
pickle_file = Path(pickle_file, f"quality/fid_{opt.get_result_folder_name()}.pkl")

#### Dataset
general_wrapper_info = {
    "image_dropout": 0,
    "size": 256,
    "inpainting": None,
    "compute_mean": False,
}
assert model is not None
assert model.opt is not None
con_type = model.opt.condition_types + ["SG"]  # type: ignore
if len(con_type) <= 2:
    if opt.root == "/media/data/robert/datasets/spinegan_T2w":
        con_type = ["CT", "T2", "SG"]
    else:
        con_type = ["CT", "T1", "SG"]

ds = Wopathfx_Dataset(size=256, gauss=True, root=root, train="test", condition_types=con_type)  # type: ignore
dataset = Wrapper_Image2Image(ds, **general_wrapper_info)
m_opt = model.opt
test_loader = DataLoader(dataset, batch_size=opt.bs, num_workers=model.opt.num_cpu, shuffle=False, drop_last=False)
####### Loop
from torchmetrics.image.fid import FrechetInceptionDistance
import random

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
fid = FrechetInceptionDistance()
fid.cuda()
fid.reset()


def to_fid(img: torch.Tensor):
    img = torch.cat([img, img, img], dim=1) * 255
    return img.to(torch.uint8).clone()


metric_list = {}
if pickle_file.exists():
    with open(pickle_file, "rb") as f:
        metric_list = pickle.load(f)
pickle_file.parent.mkdir(exist_ok=True)
l = len(test_loader)
image_list = []

image_gt = []
image_pred = []
with torch.no_grad():
    for iteration in range(opt.factor):
        for i, train_batch in enumerate(test_loader, start=1):  # type: ignore
            print(f"{i:3}/{l:3} in loop {iteration}", "         ", end="\r")
            # inception score
            #### Get batch ####
            cond: torch.Tensor = train_batch["condition"]
            seg = (cond[:, [-1]] + 1) / 2
            seg_np: np.ndarray = seg.numpy()
            for i in range(seg_np.shape[0]):
                seg_np[i][0] = ndimage.binary_dilation(seg_np[i][0], iterations=10).astype(seg_np.dtype)
            seg = torch.from_numpy(seg_np)
            cond = cond[:, :-1]

            target: torch.Tensor = train_batch["target"]
            assert cond is not None
            assert target is not None
            cond = cond.cuda()
            out = reload_any.get_image(cond, model, opt)
            out = out.cuda()
            target = (target.to(out.device) + 1) / 2

            assert out.min() >= 0, f"[{out.min()} - {out.max()}]"
            assert out.min() <= 1, f"[{out.min()} - {out.max()}]"
            assert target.min() >= 0, f"[{target.min()} - {target.max()}]"
            assert target.min() <= 1, f"[{target.min()} - {target.max()}]"

            assert isinstance(out, torch.Tensor)
            seg = seg.to(out.device)
            target_s = target * seg
            out_s = out * seg

            out = out.clamp(0, 1)
            fid.update(to_fid(target_s), real=True)
            fid.update(to_fid(out_s), real=False)
            # image_gt.append(target.detach())
            # image_pred.append(out.detach())
            # print(fid.compute().item())
            # if 0 == iteration:
            #    image_list.append(
            #        torch.cat(
            #            [
            #                (cond.detach().cpu() + 1) / 2,
            #               out.detach().cpu(),
            #               target.detach().cpu(),
            #            ],
            #            2,
            #        ).cpu()
            #    )

            if test:
                break
# print(len(image_gt), len(image_pred), "Längen")
# fid.reset()
# fid.update(to_fid(torch.cat(image_gt, 0)), real=True)
# fid.update(to_fid(torch.cat(image_pred, 0)), real=False)
fid_score = fid.compute().item()
fid.reset()
print("FID-score", fid_score, opt.exp_name)
metric_list["FID"] = fid_score
with open(pickle_file, "wb") as filehandler:
    pickle.dump(metric_list, filehandler)

import torchvision

# grid = torchvision.utils.make_grid(
#    torch.cat(image_list, dim=0).cpu(),
#    nrow=len(image_list),
# )
# from torchvision.utils import save_image
#
# print(grid.shape)
# save_image(grid, "img2.jpg")
