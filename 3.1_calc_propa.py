# %%
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from itertools import islice

import pandas as pd
import numpy as np

import rebound as rb
from celmech.nbody_simulation_utilities import get_simarchive_integration_results
import assist
from utils import ecliptic_to_icrf, icrf_to_ecliptic, ecliptic_xyz_to_elements, sun_sim
# %%
model_results = pd.read_csv("data/model_results.csv", index_col=0, dtype={"Des'n": str})

prediction_path = Path("data/calc_propa")
prediction_path.mkdir(parents=True, exist_ok=True)
ephem = assist.Ephem("data/assist/linux_m13000p17000.441", "data/assist/sb441-n16.bsp")

conver = 58.132440867049
epoch = 2460200.5
# %%
num_to_run = len(model_results)
# model_results = model_results.sample(num_to_run, random_state=42)
# %%
sun = sun_sim.particles[0]
def propa_calc(r):
	idx, row = r
	# Convert Nesvorny orbital elements into xyz, vxyz
	sim = rb.Simulation()
	ps = sim.particles
	sim.add(primary=sun, 
		 a=row['a'], 
		 e=row['e'], 
		 inc=row['Incl.']*np.pi/180, 
		 Omega=row['Node']*np.pi/180, 
		 omega=row['Peri.']*np.pi/180, 
		 M=row['M']*np.pi/180)
	p = sim.particles[-1]
	p = ecliptic_to_icrf(sim.particles[-1])
	p.vxyz = np.array(p.vxyz) / conver

	# ASSIST integration
	Nout = int(1e3*np.pi)
	sim = rb.Simulation()
	ex = assist.Extras(sim, ephem)
	sim.t = epoch - ephem.jd_ref
	sim.ri_ias15.adaptive_mode = 2
	
    # We are doing the unit conversion ourself (using the conver variable)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="Particle is being adding from a simulation that uses different units.")
		sim.add(p)
	
	p = icrf_to_ecliptic(p)
	orb = ecliptic_xyz_to_elements(p)
	times = np.linspace(sim.t, 3e5 + sim.t, Nout)
	a = np.zeros(Nout)
	for i, time in enumerate(times):
		sim.integrate(time)
		p = sim.particles[-1]
		p = icrf_to_ecliptic(p)
		p.vxyz = np.array(p.vxyz) * conver
		orbit = ecliptic_xyz_to_elements(p)
		p.vxyz = np.array(p.vxyz) / conver
		p = ecliptic_to_icrf(p)
		a[i] = orbit.a
	
	return row["Des'n"], np.mean(a), row["propa"]

r = propa_calc(next(model_results.iterrows()))
r
# %%
def propa_calc_safe(row):
	try:
		return propa_calc(row)
	except Exception as e:
		print(e)
		return None

start = time.time()
with Pool(40) as p:
	raw_results = list(
		tqdm(
			p.imap(propa_calc_safe, islice(model_results.iterrows(), num_to_run)),
			total=num_to_run
		)
	)
end = time.time()
print("Prop a Calc Time: %.2f seconds" % (end - start))
results = [r for r in raw_results if r is not None]

df_results = pd.DataFrame(results, columns=["Des'n", "propa_cal", "propa_nes"])
df_results.to_csv("data/proper_a_integration_results_all.csv")
# %%
# calcas = np.array([x[1] for x in results])
# trueas = np.array(model_results.head(num_to_run)["propa"])
# desns = [x[0] for x in results]

# print(np.mean((calcas - trueas)**2))
# prop_a_df = pd.DataFrame({"Des'n": desns, "nesvorny_propa": trueas, "calc_propa": calcas})
# prop_a_df.to_csv("data/propa_calc.csv")
# # %%
# prop_a_df = pd.read_csv("data/propa_calc.csv")
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colors import LogNorm

# cmap = mpl.cm.magma
# norm = LogNorm(vmin=10)
# # plt.hist2d(prop_a_df["nesvorny_propa"], prop_a_df["calc_propa"], bins=200, norm=norm, cmap=cmap)
# plt.scatter(prop_a_df["nesvorny_propa"], prop_a_df["calc_propa"])

# minval = min(prop_a_df["nesvorny_propa"].min(), prop_a_df["calc_propa"].min())
# maxval = max(prop_a_df["nesvorny_propa"].max(), prop_a_df["calc_propa"].max())
# plt.plot([minval, maxval], [minval, maxval], ls = 'dashed', linewidth=2, color = "grey")

# plt.xlim(0, 5)
# plt.ylim(0, 5)

# plt.show()
# # %%
