# %%
import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

import rebound as rb
import celmech as cm

from linear_theory import linear_theory_prediction, make_simpler_secular_theory
# %%
prediction_path = Path("data/linear_predictions")
prediction_path.mkdir(parents=True, exist_ok=True)
# %%
simpler_secular_theory = make_simpler_secular_theory()

df = pd.read_fwf('data/MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
df = df[df['Epoch'] == 'K239D'] # take only ones at common epoch--almost all of them
for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
	df[c] = pd.to_numeric(df[c])

labels = pd.read_fwf('data/proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

nesvorny_df = pd.merge(df, labels, on="Des'n", how="inner")
nesvorny_df.to_csv("data/nesvorny_catalog_dataset.csv")

def ecc_inc_prediction(r):
	idx, row = r
	sim = rb.Simulation('data/planets-epoch-2460200.5.bin')
	sim.add(
		a=row['a'],
		e=row['e'],
		inc=row['Incl.'] * np.pi / 180,
		Omega=row['Node'] * np.pi / 180,
		omega=row['Peri.'] * np.pi / 180,
		primary=sim.particles[0]
	)
	sim.particles[5].m
	for i in range(4): # this removes the four terrestrial planets
		sim.remove(index=1)
	sim.move_to_com()
	cm.nbody_simulation_utilities.align_simulation(sim)
	sim.integrator = 'whfast'
	sim.dt = np.min([p.P for p in sim.particles[1:]]) / 25.
	sim.ri_whfast.safe_mode = 0
	a = sim.particles[5].a
	e = sim.particles[5].e
	inc = sim.particles[5].inc
	omega = sim.particles[5].omega
	Omega = sim.particles[5].Omega

	u0, v0, g0, s0 = linear_theory_prediction(e, inc, omega, Omega, a, row['propa'], simpler_secular_theory)

	np.savez(prediction_path / f"linear_prediction_results_{row["Des'n"]}", u=u0, v=v0, g=g0, s=s0)
# %%
with Pool(40) as p:
	table = p.map(ecc_inc_prediction, nesvorny_df.iterrows())
