# %%
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rebound as rb
import celmech as cm
import assist

# %%
ephem = assist.Ephem("../data/assist/linux_m13000p17000.441", "../data/assist/sb441-n16.bsp")
# %%
outer_only = True
print(f"Outer Planets Only: {outer_only}")
# %%
integration_path = Path("../data") / "uncertainty_integrations"
integration_path.mkdir(parents=True, exist_ok=True)
# %%
merged_df = pd.read_csv("../data/uncertainty_asteroids_sampled.csv", index_col=0)
start_time = 2460200.5

# %%
if not outer_only:
    planets_list = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune"
	]

planets_list = [
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune"
]
    

# %%
def add_planets(sim):
	for name in planets_list:
		planet = ephem.get_particle(name, start_time - ephem.jd_ref)
		sim.add(planet)
# %%
def run_sim(r):
	idx, row = r
	sim_file = integration_path / f"asteroid_integration_{row["Des'n"]}-{idx}.sa"
	print(sim_file.resolve())
	
	# Integrate backwards to Nesvorny epoch
	sim = rb.Simulation()
	ex = assist.Extras(sim, ephem)
	# sim.move_to_hel()
	# sim.add(x=row['x'], y=row['y'], z=row['z'], 
	# 	 vx=row['vx']/(np.pi*2), vy=row['vy']/(np.pi*2), vz=row['vz']/(np.pi*2))
	sim.add(x=row['x'], y=row['y'], z=row['z'], vx=row['vx'], vy=row['vy'], vz=row['vz'])
	print(sim.particles[-1].orbit(rb.Particle(m=1)).a)
	# sim.move_to_com()
	sim.integrate(start_time - ephem.jd_ref)

	print(np.sqrt(row['x']**2 + row['y']**2 + row['z']**2))

	# Create a new simulation and integrate forward
	ps = sim.particles
	ps_asteroid = ps[-1]
	ps_final = {
		'x': ps_asteroid.x,
		'y': ps_asteroid.y,
		'z': ps_asteroid.z,
		'vx': ps_asteroid.vx,
		'vy': ps_asteroid.vy,
		'vz': ps_asteroid.vz
	}

	sim2 = rb.Simulation()
	ex2 = assist.Extras(sim2, ephem)
	add_planets(sim2)
	sim2.move_to_hel()
	sim2.t = start_time - ephem.jd_ref

	sim2.add(x=ps_final['x'], y=ps_final['y'], z=ps_final['z'], 
		vx=ps_final['vx'], vy=ps_final['vy'], vz=ps_final['vz'])
	
	ps2 = sim2.particles
	sim2.integrator = 'whfast'
	sim2.dt = ps2[1].P/100.
	sim2.ri_whfast.safe_mode = 0

	sim2.save_to_file(str(sim_file), step=int(5e3*(np.pi*2)/sim.dt), delete_file=True)
	sim2.integrate(50e6, exact_finish_time=0)

# %%
with Pool(1) as p:
      p.map(run_sim, merged_df.iterrows())


