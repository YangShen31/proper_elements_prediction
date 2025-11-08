# %%
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from itertools import islice

import pandas as pd
import numpy as np

import rebound as rb
from celmech.nbody_simulation_utilities import get_simarchive_integration_results
# %%
model_results = pd.read_csv("data/model_results.csv", index_col=0, dtype={"Des'n": str})

prediction_path = Path("data/calc_propa")
prediction_path.mkdir(parents=True, exist_ok=True)
# %%
num_to_run = 100_000
# model_results = model_results.sample(num_to_run, random_state=42)
# %%
def propa_calc(r):
    idx, row = r

    # sim = rb.Simulation('data/planets-epoch-2460200.5.bin')
    # sim.remove(hash="mercury")
    # sim.remove(hash="venus")
    # sim.remove(hash="earth")
    # sim.remove(hash="mars")

    # sim.N_active = sim.N
    # # add some asteroids
    # sim.add(a=row['a'], e=row['e'], inc=row['Incl.']*np.pi/180, Omega=row['Node']*np.pi/180, omega=row['Peri.']*np.pi/180, M=row['M'], primary=sim.particles[0])
    # sim.move_to_com()

    # # We set the timestep of almost exactly 6.5 days. The sqrt(42) ensures we have a transcendental number.d
    # sim.dt = np.sqrt(42)*(np.pi*2)/365.25
    # sim.integrator = 'whfast'

    # fpath = str(prediction_path / f"linear_prediction_results_{row["Des'n"]}.bin")
    # sim.save_to_file(fpath, step=int(15*(np.pi*2)/sim.dt), delete_file=True)
    # sim.integrate(1e4*np.pi*2, exact_finish_time=0)

    result = get_simarchive_integration_results(str(prediction_path / f"linear_prediction_results_{row["Des'n"]}.bin"), coordinates='heliocentric')
    return row["Des'n"], result['a'][-1].mean()
    # return result

r = propa_calc(next(model_results.iterrows()))
# %%
with Pool(40) as p:
    results = list(
        tqdm(
            p.imap(propa_calc, islice(model_results.iterrows(), num_to_run)),
            total=num_to_run
        )
    )
# %%
calcas = np.array([x[1] for x in results])
trueas = np.array(model_results.head(num_to_run)["propa"])
desns = [x[0] for x in results]

print(np.mean((calcas - trueas)**2))
prop_a_df = pd.DataFrame({"Des'n": desns, "nesvorny_propa": trueas, "calc_propa": calcas})
prop_a_df.to_csv("data/propa_calc.csv")
# %%
prop_a_df = pd.read_csv("data/propa_calc.csv")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

cmap = mpl.cm.magma
norm = LogNorm(vmin=10)
# plt.hist2d(prop_a_df["nesvorny_propa"], prop_a_df["calc_propa"], bins=200, norm=norm, cmap=cmap)
plt.scatter(prop_a_df["nesvorny_propa"], prop_a_df["calc_propa"])

minval = min(prop_a_df["nesvorny_propa"].min(), prop_a_df["calc_propa"].min())
maxval = max(prop_a_df["nesvorny_propa"].max(), prop_a_df["calc_propa"].max())
plt.plot([minval, maxval], [minval, maxval], ls = 'dashed', linewidth=2, color = "grey")

plt.xlim(0, 5)
plt.ylim(0, 5)

plt.show()
# %%
