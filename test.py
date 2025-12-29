# %%
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import rebound as rb
from celmech.nbody_simulation_utilities import get_simarchive_integration_results

%config InlineBackend.figure_format = 'retina'
# %%
model_results = pd.read_csv("data/model_results.csv", index_col=0, dtype={"Des'n": str})
nesvorny_elements = pd.read_csv("data/nesvorny_catalog_dataset.csv", index_col=0, dtype={"Des'n": str})
df = pd.merge(model_results, nesvorny_elements[["Des'n", 'a', 'da', 'de', 'dsini']], on="Des'n", how="inner")

prediction_path = Path("data/test_plots")
prediction_path.mkdir(parents=True, exist_ok=True)
# %%
column_names = ['propa', 'prope', 'propsini', 'g', 's', 'H', 'NumOpps', 'PackedName', 'UnpackedName']
df_Velleda = pd.read_csv("data/family_tables/inner_126_velleda_fam3.csv", header=None, names=column_names)
def convert_id(val):
	val_str = str(val)
	if val_str.isdigit():
		return int(val_str)
	return val_str

df_Velleda["PackedName"] = df_Velleda["PackedName"].apply(convert_id)

family_df = nesvorny_elements[nesvorny_elements["Des'n"].isin(df_Velleda['PackedName'])]
family_df
# %%
in_family = ["B1424", "B4029"] # prope ~ 0.071, e ~ 0.064-0.71

out_family_candidates = model_results[(model_results["e"]<0.0641) & (model_results["e"]> 0.0640)]
for i,c in out_family_candidates.iterrows():
    if not c["Des'n"] in family_df["Des'n"].tolist():
        out_family = [c["Des'n"]]

print(in_family, out_family)
# %%
sim = rb.Simulation('data/planets-epoch-2460200.5.bin')

sim.remove(hash="mercury")
sim.remove(hash="venus")
sim.remove(hash="earth")
sim.remove(hash="mars")

for idx,row in model_results[model_results["Des'n"].isin(in_family+out_family)].iterrows():
    sim.add(a=row['a'], e=row['e'], inc=row['Incl.']*np.pi/180, Omega=row['Node']*np.pi/180, omega=row['Peri.']*np.pi/180, M=row['M'], primary=sim.particles[0])

sim.move_to_com()

sim.dt = np.sqrt(42)*(np.pi*2)/365.25
sim.integrator = 'whfast'

fpath = str(prediction_path / f"family_inout.bin")
sim.save_to_file(fpath, step=int(1e2*(np.pi*2)/sim.dt), delete_file=True)
sim.integrate(2e4*np.pi*2, exact_finish_time=0)
# %%
result = get_simarchive_integration_results(str(prediction_path / f"family_inout.bin"), coordinates='heliocentric')
# %%
matplotlib.rcParams.update({'font.size': 22, 'lines.linewidth': 2})

fig, axs = plt.subplots(1, 2, figsize=(15,5), sharex=True, sharey=True)
axs[0].plot(result['time']/(2*np.pi), result["e"][-3], c='tab:blue')
axs[0].plot(result['time']/(2*np.pi), result["e"][-2], c='tab:cyan')
axs[0].plot(result['time']/(2*np.pi), result["e"][-1], c='gray', linestyle="--")
# axs[0].set_xlabel("Time [yr]")
axs[0].set_ylabel("$e$")
axs[0].set_title("Eccentricity")

osc_e = []
for idx,row in model_results[model_results["Des'n"].isin(in_family+out_family)].iterrows():
      osc_e.append(row['prope'])
axs[1].axhline(osc_e[-3], c='tab:blue', label="Asteroid 1")
axs[1].axhline(osc_e[-2], c='tab:cyan', label="Asteroid 2")
axs[1].axhline(osc_e[-1], c='gray', linestyle="--", label="Asteroid 3")
axs[1].legend()
axs[1].set_title("Proper Eccentricity")

ax = fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_xlabel("Time [yr]")
# %%
fig, axs_simple = plt.subplots(1, 2, figsize=(15,5), sharex=True, sharey=True)
axs_simple[0].plot(result['time']/(2*np.pi), result["e"][-3], c='tab:blue')
# axs[0].plot(result['time']/(2*np.pi), result["e"][-2], c='tab:cyan')
# axs[0].plot(result['time']/(2*np.pi), result["e"][-1], c='gray', linestyle="--")
axs_simple[0].set_xlabel("Time [yr]")
axs_simple[0].set_ylabel("$e$")
axs_simple[0].set_title("Eccentricity")
axs_simple[0].set_ylim(*axs[0].get_ylim())
# %%
