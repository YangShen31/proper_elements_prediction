# %%
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from itertools import islice

import pandas as pd
import numpy as np

import rebound as rb
from celmech.nbody_simulation_utilities import get_simarchive_integration_results
import assist
from utils import ecliptic_to_icrf, icrf_to_ecliptic, ecliptic_xyz_to_elements
# %%
model_results = pd.read_csv("data/model_results.csv", index_col=0, dtype={"Des'n": str})

prediction_path = Path("data/pred_propa")
prediction_path.mkdir(parents=True, exist_ok=True)
ephem = assist.Ephem("data/assist/linux_m13000p17000.441", "data/assist/sb441-n16.bsp")

conver = 58.132440867049
epoch = 2460200.5
# %%
num_to_run = len(model_results)
# %%
sun_sim = rb.Simulation()
sun_sim.add("sun", plane="ecliptic", date="JD%f"%epoch)
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
	np.savez(prediction_path / f"propa_prediction_results_{row["Des'n"]}", u = np.mean(a))

	des = row["Des'n"]

	return des

# %%
with Pool(40) as p:
	for _ in tqdm(p.imap_unordered(propa_calc, model_results.iterrows()), total=len(model_results),
		desc="Processing asteroids"
	):
		pass
