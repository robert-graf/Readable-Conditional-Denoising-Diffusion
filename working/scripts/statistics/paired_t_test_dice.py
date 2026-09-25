from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
import scipy.stats as stats
import pandas
from BIDS import v_idx2name
import math
from scripts.segmentation.MRI2Seg_run_all import get_all_models


def compute_significance(out=Path("/media/data/robert/test/T1w/"), file_name="dice_scores", ablation_registration=False):
    print("##############################################################")

    if not out.exists():
        print("Does not exit", out)
        return
    print(out)
    model_outs = {}

    models = get_all_models(str(out), T1=("T1w" in str(out)), ablation_registration=ablation_registration)
    model_outs = {}
    # for class_folder in out.iterdir(): #<-- get all computed results (not sorted)
    for a in models:
        class_folder = a.get_result_folder_name()
        if "3D" in class_folder:
            if file_name == "dice_score_posterior":
                fi = "dice_score3D_subreg"
            else:
                fi = "dice_score3D"
        else:
            fi = file_name
        print(class_folder)
        xls = out / class_folder / f"{fi}.xlsx"
        if not xls.exists():
            print(xls)
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
            .replace("diffusion3D_CT", "3D")
            .replace("dif_spinegan3D_CT", "3D")
            .replace("_img", "i")
        ] = data

    # model_outs = {key: model_outs[key] for key in sorted(model_outs)}
    for key, values in model_outs.items():
        # print(values)
        global_dice: pandas.Series = values.loc[:, "global"]
        # d = values.loc[:, "T12"]
        # print(d.isnull() == global_dice.isnull())
        # print(d)
        individual_dice = []
        group1 = []
        group2 = []
        group3 = []
        g1 = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        g2 = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13"]
        g3 = ["L1", "L2", "L3", "L4", "L5", "L6"]
        for name in v_idx2name.values():
            if name in values:
                dice: pandas.Series = values.loc[:, name]
                individual_dice.append(dice)
                if name in g1:
                    group1.append(dice)
                if name in g2:
                    group2.append(dice)
                if name in g3:
                    group3.append(dice)

        a = pandas.concat(individual_dice)
        group1 = pandas.concat(group1)
        group2 = pandas.concat(group2)
        group3 = pandas.concat(group3)
        # ±{global_dice.std():.2f}#±{a.std():.2f}
        print(
            f"{key:20}  {global_dice.count()}  {global_dice.mean():.2f}  {a.mean():.2f}  {group1.mean():.2f}  {group2.mean():.2f}  {group3.mean():.2f}"
        )
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
            individual_dice1: pandas.DataFrame = pandas.concat(individual_dice)

            individual_dice = []
            for name in v_idx2name.values():
                if name in values2:
                    dice: pandas.Series = values2.loc[:, name]
                    individual_dice.append(dice)
            individual_dice2: pandas.DataFrame = pandas.concat(individual_dice)
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
    # compute_significance(out=Path("/media/data/robert/test/T2w/"), ablation_registration=True)

    # compute_significance(out=Path("/media/data/robert/test/T1w/"))
    # compute_significance(out=Path("/media/data/robert/test/T2w/"))
    # compute_significance(out=Path("/media/data/robert/test/MRSpineSeg_Challenge_ours/"))
    # compute_significance(out=Path("/media/data/robert/test/MRSpineSeg_Challenge_split1/"))

    compute_significance(out=Path("/media/data/robert/test/T1w/"), file_name="dice_score_posterior")
    compute_significance(out=Path("/media/data/robert/test/T2w/"), file_name="dice_score_posterior")
    compute_significance(out=Path("/media/data/robert/test/MRSpineSeg_Challenge_ours/"), file_name="dice_score_posterior")
    # compute_significance(out=Path("/media/data/robert/test/MRSpineSeg_Challenge_split1/"))
