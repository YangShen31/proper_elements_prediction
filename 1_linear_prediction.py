# %%
import time
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

from tqdm import tqdm

import rebound as rb
import celmech as cm

from linear_theory import linear_theory_prediction, make_simpler_secular_theory
# %%
# prediction_path = Path("data/linear_predictions")
# prediction_path.mkdir(parents=True, exist_ok=True)
# %%
simpler_secular_theory = make_simpler_secular_theory()

df = pd.read_fwf('data/MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
df = df[df['Epoch'] == 'K239D'] # take only ones at common epoch--almost all of them
for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
	df[c] = pd.to_numeric(df[c])

labels = pd.read_fwf('data/proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

nesvorny_df = pd.merge(df, labels, on="Des'n", how="inner")
nesvorny_df.to_csv("data/nesvorny_catalog_dataset.csv")
# %%
def ecc_inc_prediction(r):
	idx, row = r
	# find linear estimates using proper a
	# u0, v0, g0, s0 = linear_theory_prediction(row['e'], row['Incl.'] * np.pi / 180, row['Peri.'] * np.pi / 180, row['Node'] * np.pi / 180, row['propa'], simpler_secular_theory)
	# find linear elements using oscullating a (allows for faster inference)
	u0, v0, g0, s0 = linear_theory_prediction(row['e'], row['Incl.'] * np.pi / 180, row['Peri.'] * np.pi / 180, row['Node'] * np.pi / 180, row['a'], simpler_secular_theory)

	# np.savez(prediction_path / f"linear_prediction_results_{row["Des'n"]}", u=u0, v=v0, g=g0, s=s0)
	return row["Des'n"], u0, v0, g0, s0
# %%
start_t = time.process_time()
ncpus = 40
with Pool(ncpus) as p:
	table = list(tqdm(p.imap(ecc_inc_prediction, nesvorny_df.iterrows()), total=len(nesvorny_df)))
eval_t = (time.process_time() - start_t) * ncpus
print(f"Linear Theory Time: {eval_t:.2f} sec for {len(nesvorny_df)} asteroids. {eval_t/len(nesvorny_df):.4} sec / asteroid")
# Linear Theory Time: 9531.33 sec for 1249051 asteroids. 0.007631 sec / asteroid
# %%
df = pd.DataFrame(table, columns=["Des'n", "u0", "v0", "g0", "s0"])
df.to_csv("data/linear_theory.csv")
# %%