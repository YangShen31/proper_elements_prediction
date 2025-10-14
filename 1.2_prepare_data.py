# %%
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
# %%
nesvorny_df = pd.read_csv("data/nesvorny_catalog_dataset.csv", index_col=0, dtype={"Des'n": str})
linear_prediction_path = Path("data/linear_predictions")
# %%
g0 = []
s0 = []
prope_h = []
propsini_h = []
for i, row in tqdm(nesvorny_df.iterrows(), total=nesvorny_df.size):
    linear_prediction = np.load(linear_prediction_path / f"linear_prediction_results_{row["Des'n"]}.npz")
    g0.append(linear_prediction['g'])
    s0.append(linear_prediction['s'])
    prope_h.append(np.abs(linear_prediction['u']).item())
    propsini_h.append(np.abs(linear_prediction['v']).item())
# %%