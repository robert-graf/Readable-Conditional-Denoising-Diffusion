from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
import scipy.stats as stats
import pandas
from BIDS import v_idx2name
import math
from scripts.segmentation.NAKO2Seg_run_all import get_all_models


def compute_significance(out=Path("/media/data/robert/test/T1w/")):
    print("##############################################################")

    if not out.exists():
        print("Does not exit", out)
        return
    print(out)
    model_outs = {}

    models = get_all_models(str(out), T1="T1w" in str(out))
    model_outs = {}
    # for class_folder in out.iterdir(): #<-- get all computed results (not sorted)
    for a in models:
        class_folder = a.get_result_folder_name()
        xls = out / class_folder / "dice_scores.xlsx"
        if not xls.exists():
            continue
        data = pandas.read_excel(xls)

        model_outs[
            class_folder.replace("result_", "")
            .replace("paper_", "")
            .replace("diffusion_", "")
            .replace("exp_", "")
            .replace("T2_", "")
            .replace("T1_", "")
            .replace("ablation_image_", "i")
            .replace("image_", "i")
            .replace("eta", "e")
        ] = data

    # model_outs = {key: model_outs[key] for key in sorted(model_outs)}
    for key, values in model_outs.items():
        # print(values)
        global_dice: pandas.Series = values.loc[:, "global"]
        # d = values.loc[:, "T12"]
        # print(d.isnull() == global_dice.isnull())
        # print(d)
        individual_dice = []

        for name in v_idx2name.values():
            if name in values:
                dice: pandas.Series = values.loc[:, name]
                individual_dice.append(dice)
        a = pandas.concat(individual_dice)
        # ±{global_dice.std():.2f}#±{a.std():.2f}
        print(f"{key:20}  {global_dice.count()}  {global_dice.mean():.2f}  {a.mean():.2f}")
        # print(i, j, f"{stats.ttest_rel(l, l2).pvalue:.4f}")
    # python3 scripts/statistics/test_image_quality_all.py
    # python3 scripts/segmentation/SegTestCases.py  -en All -ds T2w --translationType TOP

    print("\nstat")
    print(f"{'':20}", end="")
    for key in model_outs.keys():
        print(f"{key[:7]:8}", end="")
    print()
    for key1, values1 in model_outs.items():
        print(f"{key1:23}", end="")
        for key2, values2 in model_outs.items():
            if key1 == key2:
                print(f"{'-':8}", end="")
                continue
            global_dice_1: pandas.Series = values1.loc[:, "global"]
            global_dice_2: pandas.Series = values2.loc[:, "global"]
            global_dice_1 = global_dice_1.fillna(0)
            global_dice_2 = global_dice_2.fillna(0)
            s = f"{stats.ttest_rel(global_dice_1.to_list(), global_dice_2.to_list()).pvalue:.4f}"
            print(f"{s:8}", end="")
        print()
    print("\nvertebra lvl")
    print(f"{'':23}", end="")
    for key in model_outs.keys():
        print(f"{key[:7]:8}", end="")
    print()
    for key1, values1 in model_outs.items():
        print(f"{key1:23}", end="")
        for key2, values2 in model_outs.items():
            if key1 == key2:
                print(f"{'-':8}", end="")
                continue
            global_dice_1: pandas.Series = values1.loc[:, "global"]
            global_dice_2: pandas.Series = values2.loc[:, "global"]
            individual_dice = []
            for name in v_idx2name.values():
                if name in values1:
                    dice: pandas.Series = values1.loc[:, name]
                    individual_dice.append(dice)
            individual_dice1 = pandas.concat(individual_dice)

            individual_dice = []
            for name in v_idx2name.values():
                if name in values2:
                    dice: pandas.Series = values2.loc[:, name]
                    individual_dice.append(dice)
            individual_dice2 = pandas.concat(individual_dice)
            l1 = individual_dice1.to_list()
            l2 = individual_dice2.to_list()
            l1_out = []
            l2_out = []
            for i, (v1, v2) in enumerate(zip(l1, l2)):
                if math.isnan(v1) and math.isnan(v2):
                    continue
                if math.isnan(v1):
                    l1_out.append(0)
                    l2_out.append(v2)
                elif math.isnan(v2):
                    l1_out.append(v1)
                    l2_out.append(0)
                else:
                    l1_out.append(v1)
                    l2_out.append(v2)

            s = f"{stats.ttest_rel(l1_out, l2_out).pvalue:.4f}"
            print(f"{s:8}", end="")
        print()


if __name__ == "__main__":
    compute_significance(out=Path("/media/data/robert/test/T1w/"))
    compute_significance(out=Path("/media/data/robert/test/T2w/"))
    compute_significance(out=Path("/media/data/robert/test/MRSpineSeg_Challenge_ours/"))
    compute_significance(out=Path("/media/data/robert/test/MRSpineSeg_Challenge_split1/"))
