from __future__ import annotations

import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))


from scripts.segmentation.SegTestCases import forbidden_keys
from scripts.statistics.test_image_quality import test_image_quality
from scripts import reload_any
from scripts.segmentation.NAKO2Seg_run_all import get_all_models


def run_all(opt: reload_any.Reload_Any_Option):
    run_ds(True, opt)
    run_ds(False, opt)


def run_ds(T1: bool, opt: reload_any.Reload_Any_Option):
    if T1:
        root = "/media/data/robert/datasets/fx_T1w/"
    else:
        root = "/media/data/robert/datasets/spinegan_T2w"
    opts = {k: v for k, v in opt.__dict__.items() if k not in forbidden_keys}

    opt_list = get_all_models(root, T1, opts)
    for opt in opt_list:
        opt.test = False

        _, checkpoint = reload_any.get_model(opt, no_reload=True, verbose=False)
        pickle_file = Path(checkpoint).parent.parent
        pickle_file = Path(pickle_file, f"quality/qa_{opt.get_result_folder_name()}.pkl")
        if pickle_file.exists():
            print("skip", opt.exp_name, ", exist!")
            continue
        test_image_quality(opt)


if __name__ == "__main__":
    parser = reload_any.get_option_reload()
    opt = reload_any.get_option(parser)
    opt.factor = 10
    opt.new = True
    opt.bs = 24
    run_all(opt)
    # python3 scripts/statistics/test_image_quality_all.py
    # python3 scripts/segmentation/SegTestCases.py  -en All -ds All --translationType TOP
