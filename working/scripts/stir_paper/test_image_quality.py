from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from BIDS import NII

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
import pickle
import random

import numpy as np
import torch
from piq import DISTS, vif_p
from torch import Tensor
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

batch_able = ["SSIM", "VIFp", "DISTS"]

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


@torch.no_grad()
def eval_2_images(out_nii: NII, target_nii: NII, metric_list: dict[str, list[float]]):
    out_nii = out_nii.normalize_mri(quantile=0.95)
    target_nii = target_nii.normalize_mri(quantile=0.95)
    arr = target_nii.get_array()  # .astype(float)
    target = Tensor(arr.astype(float))  # (Tensor(arr).to(device) + 1) / 2
    out = Tensor(out_nii.get_array())  # * seg
    assert out.min() >= -1, f"[{out.min()} - {out_nii.max()}]"
    assert out.min() <= 1, f"[{out.min()} - {out_nii.max()}]"
    # print(f"[{out.min()} - {out_nii.max()}]")
    # print(f"[{target_nii.min()} - {target_nii.max()}]")
    for key, metric_fun in metric.items():
        if key not in metric_list:
            metric_list[key] = []
        # m = metric_fun(target_s, out_s)
        # metric_list[key].append(m.detach().cpu().item())
        if key in batch_able:
            m = metric_fun(out.unsqueeze(1), target.unsqueeze(1))
        else:
            m = metric_fun(out, target)

        metric_list[key].append(m.detach().cpu().item())


def save_metric_list(metric_list: dict[str, list[float]], path: Path):
    with open(path, "wb") as filehandler:
        pickle.dump(metric_list, filehandler)


from BIDS import Logger_Interface, No_Logger


from BIDS import Logger_Interface, No_Logger


def print_metric_list(metric_list: dict[str, list[float]], logger: Logger_Interface = No_Logger()):
    for key, value_list in metric_list.items():
        l = np.array(value_list)
        logger.print(f"{key}\t {l.mean():.4f} ± {l.var():.8f}                \t\t{l.shape}", ignore_prefix=True)
    logger.print("\n", ignore_prefix=True)
    for i, (key, value_list) in enumerate(metric_list.items()):
        if i == 0:
            logger.print(f"n = {len(value_list)}\n", ignore_prefix=True)
        l = np.array(value_list)
        if i == 0 or i == 1:
            logger.print(f"{l.mean():.4f}", end="\t", ignore_prefix=True)
        else:
            logger.print(f"{l.mean():.3f}", end="\t", ignore_prefix=True)

    logger.print("\n")


if __name__ == "__main__":
    parser = reload_any.get_option_reload()
    parser = reload_any.get_option_test(parser)
    opt = reload_any.get_option(parser)
    test_image_quality(opt)
