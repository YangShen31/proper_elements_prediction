# %%
import time
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

from tqdm import tqdm

import rebound as rb
import reboundx
from reboundx import constants as rbx_constants
import celmech as cm

from linear_theory import linear_theory_prediction, make_simpler_secular_theory
# %%
# Load the Nesvorny dataset and convert it to a CSV file
df = pd.read_fwf('data/MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
df = df[df['Epoch'] == 'K239D'] # take only ones at common epoch--almost all of them
for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
	df[c] = pd.to_numeric(df[c])

labels = pd.read_fwf('data/proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

nesvorny_df = pd.merge(df, labels, on="Des'n", how="inner")
nesvorny_df.to_csv("data/nesvorny_catalog_dataset.csv")
# %%
# The linear theory expects elements in the invariable frame (angular momentum points in the Z direction)
# This is not the frame that Nesvorny elements are measured in, so we create a simulation to measure the 
# total angular momentum of the solar system, and calculate the correct rotation into the invariable frame

start_time = 2460200.5
ref_sim = rb.Simulation()

ref_sim.add("Sun", date="JD%f" % start_time)
ref_sim.add("Jupiter", date="JD%f" % start_time)
ref_sim.add("Saturn", date="JD%f" % start_time)
ref_sim.add("Uranus", date="JD%f" % start_time)
ref_sim.add("Neptune", date="JD%f" % start_time)

ref_sim.move_to_com()

L = ref_sim.angular_momentum()
x_new = rb.Vec3d(L.y, -L.x, 0.0)
# create a rotation we can apply later
rot = rb.Rotation.to_new_axes(newz=L,newx=x_new)
# %%
# load linear theory harmonics from file
simpler_secular_theory = make_simpler_secular_theory()

def ecc_inc_prediction(r):
	idx, row = r

	# Get the orbital elements of the particle in the invariable frame
	sim = rb.Simulation()
	sun = ref_sim.particles[0]
	sim.add(
        m=sun.m,
        x=sun.x,
        y=sun.y,
        z=sun.z,
        vx=sun.vx,
        vy=sun.vy,
        vz=sun.vz
    )

	sim.add(m=0,
		a=row["a"],
		e=row["e"],
		inc=np.radians(row["Incl."]),
		Omega=np.radians(row["Node"]),
		omega=np.radians(row["Peri."]),
		primary=sim.particles[0],
		date="JD%f" % start_time)
	
	sim.rotate(rot)

	p = sim.particles[-1]
	orb = p.orbit(primary=sim.particles[0])

	# Calculate linear theory
	u0, v0, g0, s0 = linear_theory_prediction(orb.e, orb.inc, orb.pomega, orb.Omega, orb.a, simpler_secular_theory)

	# From now on, we will use the invariable frame elements as the oscullating elements
	return row["Des'n"], u0, v0, g0, s0, orb.a, orb.e, np.degrees(orb.inc), np.degrees(orb.Omega), np.degrees(orb.pomega)
# %%
start_t = time.process_time()
ncpus = 40
with Pool(ncpus) as p:
	table = list(tqdm(p.imap(ecc_inc_prediction, nesvorny_df.iterrows()), total=len(nesvorny_df)))
eval_t = (time.process_time() - start_t) * ncpus
print(f"Linear Theory Time: {eval_t:.2f} sec for {len(nesvorny_df)} asteroids. {eval_t/len(nesvorny_df):.4} sec / asteroid")
# Linear Theory Time: 11992.52 sec for 1249051 asteroids. 0.009601 sec / asteroid
# %%
# u0,v0 are the complex proper elements in the invariable frame
# g0,s0 are the proper frequencies
# prope_linear, propi_linear are the proper elements in the original frame
df_all = pd.DataFrame(table, columns=["Des'n", "u0", "v0", "g0", "s0", "a", "e", "Incl.", "Node", "Peri."])
df_all.to_csv("data/linear_theory.csv")
