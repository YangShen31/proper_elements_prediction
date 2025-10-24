# %%
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rebound as rb

import sys
sys.path.insert(0, 'SBDynT/src')
import sbdynt as sbd

outer_only = True
print(f"Outer Planets Only: {outer_only}")
# %%
integration_path = Path("../data") / "uncertainty_integrations"
integration_path.mkdir(parents=True, exist_ok=True)
# %%
merged_df = pd.read_csv("../data/uncertainty_asteroids_sampled.csv", index_col=0)
start_time = 2460200.5
# %%
sim = rb.Simulation()

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='sun',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='sun')
if not outer_only:
    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='mercury',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='mercury')

    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='venus',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='venus')

    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='earth',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='earth')

    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='mars',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='mars')

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='jupiter',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='jupiter')

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='saturn',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='saturn')

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='uranus',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='uranus')

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='neptune',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2), hash='neptune')

if outer_only:
    assert len(sim.particles) == 5, "Error adding planets"
else:
    assert len(sim.particles) == 9, "Error adding planets"

sim.save_to_file(str(integration_path/"planets.bin"))
# %%
def run_sim(r):
    idx, row = r

    sim = rb.Simulation(str(integration_path/'planets.bin'))
    rel_int_time = (row["epoch"]-start_time) / ((1/(2.*np.pi))*365.25)
    sim.integrate(rel_int_time)
    sim.move_to_hel()

    sim.add(x=row['x'], y=row['y'], z=row['z'], vx=row['vx']/(np.pi*2), vy=row['vy']/(np.pi*2), vz=row['vz']/(np.pi*2))
    sim.move_to_com()

    sim.exit_max_distance = 50.

    ps = sim.particles
    ps[-1].a

    sim.integrator='whfast'
    sim.dt = ps[1].P/100.
    sim.ri_whfast.safe_mode = 0

    Tfin_approx = 3e7*ps[-1].P
    total_steps = np.ceil(Tfin_approx / sim.dt)
    Tfin = total_steps * sim.dt + sim.dt
    Nout = 25_000

    sim_file = integration_path / f"asteroid_integration_{row["Des'n"]}-{idx}.sa"
    sim.save_to_file(str(sim_file), step=int(np.floor(total_steps/Nout)), delete_file=True)
    sim.integrate(Tfin, exact_finish_time=0)
# %%
with Pool(40) as p:
      p.map(run_sim, merged_df.iterrows())
# %%
