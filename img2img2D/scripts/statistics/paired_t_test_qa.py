from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
import scipy.stats as stats
import math
from loader.arguments import get_latest_Checkpoint

from scripts import reload_any
from scripts.segmentation.NAKO2Seg_run_all import get_all_models
import numpy as np

parser = reload_any.get_option_reload(True)
opt = reload_any.get_option(parser)
# opt.exp_name
from scripts.segmentation.SegTestCases import forbidden_keys as forbidden

opts = {k: v for k, v in opt.__dict__.items() if k not in forbidden}
models_reload_opt = get_all_models(opt.root, opt.T1, opts)
import pickle

model_outs: dict[str, dict[str, list[float]]] = {}
model_outs_fid: dict[str, float] = {}

for opt in models_reload_opt:
    checkpoint = get_latest_Checkpoint(opt.exp_name, version=opt.version, log_dir_name=opt.log_dir_name, best=False)

    if checkpoint is None:
        try:
            # print(opt)
            # print("########################################################################")
            _, checkpoint = reload_any.get_model(opt, no_reload=True, verbose=False)
        except FileNotFoundError:
            continue
    checkpoint = Path(checkpoint)
    path = checkpoint.parent.parent / "quality" / ("qa_" + opt.get_result_folder_name() + ".pkl")
    path_fid = checkpoint.parent.parent / "quality" / f"fid_{opt.get_result_folder_name()}.pkl"
    print(path.name)
    # path = checkpoint.parent.parent / "quality" / ("qa_" + checkpoint.stem + ".pkl")  # TODO
    if not path.exists():
        continue
    name = (
        opt.get_result_folder_name()
        .replace("result_", "")
        .replace("paper_", "")
        .replace("diffusion_", "")
        .replace("exp_", "")
        .replace("T2_", "")
        .replace("T1_", "")
        .replace("ablation_image_", "i")
        .replace("image_", "i")
    )
    with open(path, "rb") as filehandler:
        data = pickle.load(filehandler)
        model_outs[name] = data
    if path_fid.exists():
        with open(path_fid, "rb") as filehandler:
            data = pickle.load(filehandler)
            model_outs_fid[name] = data["FID"]


keys = model_outs[list(model_outs.keys())[0]].keys()
print(f"{'name':30}", end="")
for key in keys:
    print(f"{key:15}", end="")
print(f"{'fid':7}", end="")

print()
for key_model, values_dict in model_outs.items():

    print(f"{key_model:30}", end="")
    for key_loss, loss_list in values_dict.items():
        # loss_list = loss_list[: len(loss_list) // 10]  ################################### Use only the first instanceg
        loss_list = np.array(loss_list)
        if key_loss == "L1" or key_loss == "MSE":
            out = f"{loss_list.mean():.4f}±{loss_list.std():.4f}"
        elif key_loss == "PSNR":
            out = f"{loss_list.mean():.2f}±{loss_list.std():.2f}"
        else:
            out = f"{loss_list.mean():.3f}±{loss_list.std():.3f}"
        # {len(loss_list)/10}
        print(f"{out:15}", end="")
    if key_model in model_outs_fid:
        print(f"{model_outs_fid[key_model]:.3f}", end="")
    print()
for key in keys:

    print("\n", "\n", key, "T1" if opt.T1 else "T2")
    print(f"{'name':21}", end="")
    for names in model_outs.keys():
        names = names[:10] if len(names) > 10 else names
        print(f"{names:10}", end="")
    print()
    for key1, values1 in model_outs.items():
        print(f"{key1:21}", end="")
        for key2, values2 in model_outs.items():

            if key1 == key2:
                print(f"{'.':10}", end="")
                continue

            l1 = values1[key]
            l2 = values2[key]
            l1 = l1[: len(l2) // 10]  ################################### Use only the first instanceg
            l2 = l2[: len(l2) // 10]  ################################### Use only the first instanceg

            if len(l1) != len(l2):
                out = f"{len(l1)} != {len(l2)}"
            else:
                out = f"{stats.ttest_rel(l1, l2).pvalue:.4f}"
            print(f"{out:10}", end="")

        print()
