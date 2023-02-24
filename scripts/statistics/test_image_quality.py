from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))


import pickle

import numpy as np
import torch

###### Model
from scipy import ndimage
from torch import Tensor
from torch.utils.data import DataLoader

from dataloader.dataloader_mri2ct import Wopathfx_Dataset
from dataloader.Wrapper_datasets import Wrapper_Image2Image


from scripts import reload_any

# CUDA_VISIBLE_DEVICES=1
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_cut --new --factor 1 -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_cut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_pix2pix --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_pix2pix_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --cut -en paper_T1_pcut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --syndiff -en exp_syndiff --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 0 -t 20 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 1 -t 20 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 0 -t 50 -en paper_T1_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --ddim --eta 1 -t 50 -en paper_T1_diffusion --new -bs 24 --factor 10

# python3 scripts/statistics/test_image_quality.py --ddpm               -en paper_T1_diffusion --new -bs 24 --factor 10

# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_cut --new --factor 1 -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_cut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pix2pix --new -bs 16 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pix2pix_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --cut -en paper_T2_pcut_sa-unet --new -bs 4 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --syndiff -en exp_syndiff_t2w --new -bs 4 --factor 10

# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 0 -t 20 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 1 -t 20 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 0 -t 50 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddim --eta 1 -t 50 -en paper_T2_diffusion --new -bs 24 --factor 10
# python3 scripts/statistics/test_image_quality.py --root /media/data/robert/datasets/spinegan_T2w --ddpm               -en paper_T2_diffusion --new -bs 24 --factor 10


def test_image_quality(opt: reload_any.Reload_Any_Option):
    print(opt)
    exp_name = opt.exp_name
    version: str = opt.version
    ddim = opt.ddim
    cut = opt.cut
    guidance_w = opt.guidance_w
    test = opt.test
    root = opt.root
    do_reload = not opt.new

    model, checkpoint = reload_any.get_model(opt)

    pickle_file = Path(checkpoint).parent.parent
    pickle_file = Path(pickle_file, f"quality/qa_{opt.get_result_folder_name()}.pkl")

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

    ds = Wopathfx_Dataset(size=256, gauss=False, root=root, train="test", condition_types=con_type)  # type: ignore
    dataset = Wrapper_Image2Image(ds, **general_wrapper_info)
    m_opt = model.opt
    test_loader = DataLoader(dataset, batch_size=opt.bs, num_workers=m_opt.num_cpu, shuffle=False, drop_last=False)  # type: ignore
    import random

    from piq import DISTS, vif_p

    ####### Loop
    from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    metric: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
        "L1": lambda x, y: torch.mean(torch.abs(x - y)),
        "MSE": torch.nn.functional.mse_loss,
        "PSNR": peak_signal_noise_ratio,
        "SSIM": structural_similarity_index_measure,  # type: ignore
        # https://piq.readthedocs.io/en/latest/overview.html
        "VIFp": vif_p,
        "DISTS": DISTS(),
    }
    batchable = ["SSIM", "VIFp", "DISTS"]
    # batchable = []
    metric_list: dict[str, list[float]] = {}
    if do_reload:
        if pickle_file.exists():
            with open(pickle_file, "rb") as f:
                metric_list = pickle.load(f)

    pickle_file.parent.mkdir(exist_ok=True)
    shortest_key = ""
    shortest_key_count = 100000000
    for x in metric:
        if x not in metric_list:
            metric_list[x] = []
            shortest_key = x
            shortest_key_count = 0
        else:
            if shortest_key_count >= len(metric_list[x]):
                shortest_key = x
                shortest_key_count = len(metric_list[x])

    # print("shortest_key", shortest_key, shortest_key_count)
    ds_size = len(test_loader)
    bs = opt.bs
    image_list = []

    for iteration in range(opt.factor):
        for j, train_batch in enumerate(test_loader, start=1):  # type: ignore

            if len(metric_list["L1"]) >= bs * j + bs * ds_size * iteration:
                continue
            # inception score
            # Structural consitency score
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

            # out = ds.gauss_filter(out)
            out.cuda()

            assert out.min() >= -1, f"[{out.min()} - {out.max()}]"
            assert out.min() <= 1, f"[{out.min()} - {out.max()}]"

            target = (target.to(out.device) + 1) / 2
            seg = seg.to(out.device)
            target_s = target * seg
            out_s = out * seg
            assert out_s.min() >= -1, f"[{out_s.min()} - {out.max()}]"
            assert out_s.min() <= 1, f"[{out_s.min()} - {out.max()}]"

            for key, metric_fun in metric.items():
                if key in batchable:
                    # m = metric_fun(target_s, out_s)
                    # metric_list[key].append(m.detach().cpu().item())
                    for i in range(out.shape[0]):
                        m = metric_fun(out_s[i].unsqueeze(0), target_s[i].unsqueeze(0))
                        metric_list[key].append(m.detach().cpu().item())
                else:
                    for i in range(out.shape[0]):
                        m = metric_fun(out_s[i], target_s[i])
                        metric_list[key].append(m.detach().cpu().item())
            print(f"\rCompute metrics on id {j}/{ds_size} in loop {iteration}                 \t\t", end="\r")
            if 0 == iteration:
                image_list.append(
                    torch.cat(
                        [
                            (cond.detach().cpu() + 1) / 2,
                            # out.detach().cpu(),
                            # target.detach().cpu(),
                            out_s.detach().cpu(),
                            target_s.detach().cpu(),
                        ],
                        2,
                    ).cpu()
                )

            if j % 100 == 99:
                with open(pickle_file, "wb") as filehandler:
                    pickle.dump(metric_list, filehandler)
            if test and j == 100:
                break

    with open(pickle_file, "wb") as filehandler:
        pickle.dump(metric_list, filehandler)

    for key, value_list in metric_list.items():
        l = np.array(value_list)
        print(f"{key}\t {l.mean():.4f} ± {l.var():.4f}                \t\t{l.shape}")
    print("\n")
    for i, (key, value_list) in enumerate(metric_list.items()):
        l = np.array(value_list)
        if i == 0 or i == 1:
            print(f"{l.mean():.4f}", end="\t")
        else:
            print(f"{l.mean():.3f}", end="\t")
    print("\n")

    import torchvision

    grid = torchvision.utils.make_grid(
        torch.cat(image_list, dim=0).cpu(),
        nrow=len(image_list),
    )
    from torchvision.utils import save_image

    # print(grid.shape)
    save_image(grid, "img1.jpg")


if __name__ == "__main__":
    parser = reload_any.get_option_reload()
    parser = reload_any.get_option_test(parser)
    opt = reload_any.get_option(parser)
    test_image_quality(opt)
