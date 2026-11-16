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
from utils import ecliptic_to_icrf, icrf_to_ecliptic, ecliptic_xyz_to_elements, UNIT_CONVER, sun_sim
# %%
nesvorny_elements = pd.read_csv("data/nesvorny_catalog_dataset.csv", index_col=0, dtype={"Des'n": str})

prediction_path = Path("data/pred_propa")
prediction_path.mkdir(parents=True, exist_ok=True)
ephem = assist.Ephem("data/assist/linux_m13000p17000.441", "data/assist/sb441-n16.bsp")

conver = 58.132440867049
epoch = 2460200.5
# %%
num_to_run = 10000 #len(nesvorny_elements)
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
	p.vxyz = np.array(p.vxyz) / UNIT_CONVER

	# ASSIST integration
	Nout = int(1e3*np.pi)
	sim = rb.Simulation()
	ex = assist.Extras(sim, ephem)
	sim.t = epoch - ephem.jd_ref
	sim.ri_ias15.adaptive_mode = 2
	
    # We are doing the unit conversion ourself (using the conver variable)
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message=r"Particle added is from a simulation that uses different units\.",
			category=RuntimeWarning,
		)
		sim.add(p)
	
	p = icrf_to_ecliptic(p)
	times = np.linspace(sim.t, 3e5 + sim.t, Nout)
	a = np.zeros(Nout)

	# Get the solar ephemeris for Sun's position at different times
	def sun_state_at_time(t):
		sun = ephem.get_particle("sun", t)
		sun = icrf_to_ecliptic(sun)
		return np.array([
			sun.x, sun.y, sun.z,
			sun.vx * UNIT_CONVER, sun.vy * UNIT_CONVER, sun.vz * UNIT_CONVER
		])

	sun_states = np.array([sun_state_at_time(t) for t in times])
	
	for i, time in enumerate(times):
		sim.integrate(time)
		p = sim.particles[-1]
		p = icrf_to_ecliptic(p)
		p.vxyz = np.array(p.vxyz) * conver
		sx, sy, sz, svx, svy, svz = sun_states[i]
		sun_now = rb.Particle(x=sx, y=sy, z=sz,
							vx=svx, vy=svy, vz=svz, m=1.0)
		orbit = ecliptic_xyz_to_elements(p, sun_now)
		p.vxyz = np.array(p.vxyz) / conver
		p = ecliptic_to_icrf(p)
		a[i] = orbit.a

	des = row["Des'n"]

	return des, np.mean(a), row["propa"]

r = propa_calc(next(nesvorny_elements.iterrows()))
r
# %%
start_t = time.process_time()
ncpus = 40
with Pool(ncpus) as p:
	raw_results = list(
		tqdm(
			p.imap(propa_calc, islice(nesvorny_elements.iterrows(), num_to_run)),
			total=num_to_run
		)
	)
eval_t = (time.process_time() - start_t) * ncpus
print(f"Proper semi-major axis calculation time: {eval_t:.2f} sec for {len(nesvorny_elements)} asteroids. {eval_t/len(nesvorny_elements):.4} sec / asteroid")
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
