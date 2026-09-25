import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
import json

with open("/media/data/robert/datasets/MRSpineSeg_Challenge/result/seg_test.json", "r") as f:
    d = json.load(f)

import plotly.express as px
import pandas as pd
from pandas import DataFrame

dic = {"w": [], "fun": [], "eta": [], "t": [], "dice": []}

df = pd.DataFrame(dic)
for w_key, d1 in d.items():
    for f_key, d2 in d1.items():
        for eta_key, d3 in d2.items():
            for t_key, d4 in d3.items():
                new_row = {"w": w_key, "fun": f_key, "eta": eta_key, "t": t_key, "dice": d4}
                df = df.append(new_row, ignore_index=True)
                # df = pd.concat([new_row, pd.DataFrame.from_records(df)])
# append row to the dataframe

fig = px.scatter(
    df,
    x="w",
    y="dice",
    color="t",
    # size=10,
    hover_data=["w", "fun", "eta", "t", "dice"],
    symbol="fun",
)
fig.update_traces(marker_size=10)
fig.show()
