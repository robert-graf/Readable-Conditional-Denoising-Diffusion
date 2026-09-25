import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
import scipy.stats as stats
import pandas

# from BIDS import v_idx2name
import math
from scripts.segmentation.MRI2Seg_run_all import get_all_models


def add_p_value_annotation(fig: go.Figure, array_columns, subplot=None, _format=dict(interline=0.07, text_height=1.07, color="black")):
    """Original Source: https://stackoverflow.com/questions/67505252/plotly-box-p-value-significant-annotation
    Adds notations giving the p-value between two box plot data (t-test two-sided comparison)

    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    """
    # Specify in what y_range to plot for each pair of columns
    shift_down = 0.9
    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        # y_range[i] = [1.01 + i * _format["interline"] - shift_down, 1.02 + i * _format["interline"] - shift_down]
        y_range[i] = [1.02 + i * _format["interline"] - shift_down, 1.01 + i * _format["interline"] - shift_down]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ""
        else:
            subplot_str = str(subplot)
        indices = []  # Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict["data"]):
            # print(index, data['xaxis'], 'x' + subplot_str)
            if data["xaxis"] == "x" + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        print((indices))
    else:
        subplot_str = ""

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        # print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        # print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        # Get the p-value
        pvalue = stats.ttest_rel(
            fig_dict["data"][data_pair[0]]["y"],
            fig_dict["data"][data_pair[1]]["y"],
            # equal_var=False,
        )[1]
        if pvalue >= 0.05:
            symbol = "ns"
        elif pvalue >= 0.01:
            symbol = "*"
        elif pvalue >= 0.001:
            symbol = "**"
        else:
            symbol = "***"
        # Vertical line
        fig.add_shape(
            type="line",
            xref="x" + subplot_str,
            yref="y" + subplot_str + " domain",
            x0=column_pair[0],
            y0=y_range[index][0],
            x1=column_pair[0],
            y1=y_range[index][1],
            line=dict(
                color=_format["color"],
                width=2,
            ),
        )
        # Horizontal line
        fig.add_shape(
            type="line",
            xref="x" + subplot_str,
            yref="y" + subplot_str + " domain",
            x0=column_pair[0],
            y0=y_range[index][1],
            x1=column_pair[1],
            y1=y_range[index][1],
            line=dict(
                color=_format["color"],
                width=2,
            ),
        )
        # Vertical line
        fig.add_shape(
            type="line",
            xref="x" + subplot_str,
            yref="y" + subplot_str + " domain",
            x0=column_pair[1],
            y0=y_range[index][0],
            x1=column_pair[1],
            y1=y_range[index][1],
            line=dict(
                color=_format["color"],
                width=2,
            ),
        )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(
            dict(
                font=dict(color=_format["color"], size=14),
                x=(column_pair[0] + column_pair[1]) / 2,
                y=(y_range[index][0] + y_range[index][1]) / 2,  # _format["interline"] / 2
                showarrow=False,
                text=symbol,
                textangle=0,
                xref="x" + subplot_str,
                yref="y" + subplot_str + " domain",
            ),
            yanchor="bottom",
        )
    return fig


def load_files(out=Path("/media/data/robert/test/T1w/"), file_name="dice_scores"):
    print("##############################################################")
    out = Path(out)
    if not out.exists():
        print("Does not exit", out)
        return
    print(out)
    model_outs = {}

    models = get_all_models(str(out), T1=("T1w" in str(out)))
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
    return model_outs


lvl_th = [0.05, 0.01, 0.001]


def plot_global(model_outs: dict, lvl=3) -> None:
    fig = go.Figure()
    for key, values in model_outs.items():
        global_dice: pandas.Series = values.loc[:, "global"]
        fig.add_trace(go.Box(y=global_dice.to_list(), name=key, boxpoints="all"))

    anno = []
    keys = list(model_outs.keys())
    for x in range(len(model_outs)):
        for y in range(x):
            global_dice_1: pandas.Series = model_outs[keys[x]].loc[:, "global"]
            global_dice_2: pandas.Series = model_outs[keys[y]].loc[:, "global"]
            global_dice_1 = global_dice_1.fillna(0)
            global_dice_2 = global_dice_2.fillna(0)
            pvalue = stats.ttest_rel(
                global_dice_1.to_list(),
                global_dice_2.to_list(),
            ).pvalue
            # pvalue = stats.ttest_ind(
            #    fig_dict["data"][data_pair[0]]["y"],
            #    fig_dict["data"][data_pair[1]]["y"],
            #    equal_var=False,
            # )[1]

            if pvalue >= lvl_th[lvl]:
                anno.append((x, y))
    fig = add_p_value_annotation(fig, anno, _format=dict(interline=0.03, text_height=1.03, color="black"))
    fig.show()


if __name__ == "__main__":
    models_out = load_files(out=Path("/media/data/robert/test/T2w/"))
    lvl = 0

    # models_out = load_files(out=Path("/media/data/robert/_test/MRSpineSeg_Challenge_ours/"))
    # models_out = load_files(out=Path("/media/data/robert/_test/MRSpineSeg_Challenge_split1/"))
    # models_out = load_files(out=Path("/media/data/robert/_test/MRSpineSeg_Challenge_ours/"), file_name="dice_score_posterior")
    # models_out = load_files(out=Path("/media/data/robert/_test/MRSpineSeg_Challenge_ours/"), file_name="dice_score_posterior")
    # lvl = 2
    try:
        del models_out["cut_sa-unet"]
    except Exception:
        pass
    plot_global(models_out, lvl)
    # tips = px.data.tips()

    # fig = go.Figure()
    # for day in ["Thur", "Fri", "Sat", "Sun"]:
    #    fig.add_trace(go.Box(y=tips[tips["day"] == day].total_bill, name=day, boxpoints="outliers"))
    # fig = add_p_value_annotation(fig, [[0, 1], [0, 2], [0, 3]])
    # fig.show()
