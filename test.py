# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import rebound as rb
import assist

%config InlineBackend.figure_format = 'retina'
# %%
nesvorny_data = pd.read_csv("data/nesvorny_catalog_dataset.csv", index_col=0)
ephem = assist.Ephem("data/assist/linux_m13000p17000.441", "data/assist/sb441-n16.bsp")
# %%
epoch = 2460200.5
epoch_n = 2460200.5 # Nesvorny epoch
# %%
def get_e(row):
    Nout = int(500)
    sim = rb.Simulation()
    sim.add("Sun", date="JD%f"%epoch_n)
    sim.add("Mercury", date="JD%f"%epoch_n)
    sim.add("Venus", date="JD%f"%epoch_n)
    sim.add("Earth", date="JD%f"%epoch_n)
    sim.add("Mars", date="JD%f"%epoch_n)
    sim.add("Jupiter", date="JD%f"%epoch_n)
    sim.add("Saturn", date="JD%f"%epoch_n)
    sim.add("Uranus", date="JD%f"%epoch_n)
    sim.add("Neptune", date="JD%f"%epoch_n)

    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = 6.5*np.pi*2/365.25

    sim.add(a=row['a'], 
            e=row['e'], 
            inc=row['Incl.']*np.pi/180, 
            Omega=row['Node']*np.pi/180, 
            omega=row['Peri.']*np.pi/180, 
            M=row['M']*np.pi/180)

    times = np.linspace(sim.t, 80e3*np.pi*2 + sim.t, Nout)
    e = np.zeros(Nout)

    for i, time in enumerate(times):
        sim.integrate(time)
        p = sim.particles[-1]
        orbit = p.orbit()
        e[i] = orbit.e
    return e, times
# %%
des = ["K10B40G", "K19N18R", "m4495", "K15P14X", "99528", "K17C22O", "B5159"]
row = nesvorny_data[nesvorny_data["Des'n"] == des[-2]].iloc[0]
row
# %%
e, times = get_e(row)

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6,2))

axs[0].plot(times/(np.pi*2), e)
axs[0].set_xlim(5000, 75000)
axs[0].set_ylabel("$e$")
axs[0].set_title("Eccentricity")

axs[1].axhline(row['prope'])
axs[1].set_title("Proper Eccentricity")

fig.text(0.5, -0.06, 'Time [yr]', ha='center')
# %%
fragment_idx_base = [375]
fragment_idx = fragment_idx_base
# fragment_idx_rand = np.random.rand(10)
fragment_idx_rand = np.array([0.92133582, 0.78800129, 0.12554724, 0.59192609, 0.53162654,
       0.34461229, 0.81344192, 0.2996385 , 0.1975619 , 0.38077154])
fragment_idx = (fragment_idx_base[0] + 500 * (fragment_idx_rand - 0.5)).astype(int) % 499
# multiply by 0, then 100, then 500    ^^^  to get animation of the dots spreading out

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6,2))

axs[0].plot(times/(np.pi*2), e, zorder=-1, c='tab:blue')
axs[0].set_xlim(5000, 75000)
axs[0].set_ylim(0.06, 0.25)
axs[0].set_ylabel("$e$")
axs[0].set_title("Osculating Eccentricity")
axs[0].set_xticks([])

axs[0].scatter(times[fragment_idx]/(np.pi*2), e[fragment_idx], c='black')

axs[1].axhline(row['prope'], c='tab:blue', zorder=-1)
axs[1].set_title("Proper Eccentricity")
axs[1].scatter(times[fragment_idx]/(np.pi*2), np.ones(len(fragment_idx))*row['prope'], c='black')

# fig.text(0.5, -0.06, 'Time [yr]', ha='center')
# %%
row1 = nesvorny_data[nesvorny_data["Des'n"] == des[0]].iloc[0]
e1, times = get_e(row1)

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6,2))

axs[0].plot(times/(np.pi*2), e1)
axs[0].set_xlim(5000, 75000)
axs[0].set_ylabel("$e$")
axs[0].set_title("Eccentricity")

axs[1].axhline(row['prope'])
axs[1].set_title("Proper Eccentricity")

fig.text(0.5, -0.06, 'Time [yr]', ha='center')
# %%
fragment_idx_base = [375]
fragment_idx = fragment_idx_base
# fragment_idx_rand = np.random.rand(10)
fragment_idx_rand = np.array([0.92133582, 0.78800129, 0.12554724, 0.59192609, 0.53162654,
       0.34461229, 0.81344192, 0.2996385 , 0.1975619 , 0.38077154])
fragment_idx = (fragment_idx_base[0] + 500 * (fragment_idx_rand - 0.5)).astype(int) % 499

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6,2))

axs[0].plot(times/(np.pi*2), e, zorder=-1, c='tab:blue')
axs[0].set_xlim(5000, 75000)
axs[0].set_ylim(0.06, 0.25)
axs[0].set_ylabel("$e$")
axs[0].set_title("Osculating Eccentricity")
axs[0].set_xticks([])

axs[0].scatter(times[fragment_idx]/(np.pi*2), e[fragment_idx], c='black')

axs[1].axhline(row['prope'], c='tab:blue', zorder=-1)
axs[1].set_title("Proper Eccentricity")
axs[1].scatter(times[fragment_idx]/(np.pi*2), np.ones(len(fragment_idx))*row['prope'], c='black')

fragment_idx_base = [375]
fragment_idx = fragment_idx_base
# fragment_idx_rand = np.random.rand(10)
fragment_idx_rand = np.array([0.65264815, 0.71442867, 0.91297066, 0.45948945, 0.97008024,
       0.3226722 , 0.49777986, 0.92008058, 0.18583027, 0.70138386])
fragment_idx = (fragment_idx_base[0] + 500 * (fragment_idx_rand - 0.5)).astype(int) % 499


axs[0].plot(times/(np.pi*2), e1, zorder=-1, c='tab:orange')
axs[0].scatter(times[fragment_idx]/(np.pi*2), e1[fragment_idx], c='black')

axs[1].axhline(row1['prope'], c='tab:orange', zorder=-1)
axs[1].scatter(times[fragment_idx]/(np.pi*2), np.ones(len(fragment_idx))*row1['prope'], c='black')
# %%
