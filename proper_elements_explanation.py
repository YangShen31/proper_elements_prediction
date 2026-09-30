# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector, BboxConnectorPatch,
                                                   BboxPatch)

import pandas as pd
import numpy as np

import rebound as rb
import assist

plt.rcParams.update({'font.size': 8})
%config InlineBackend.figure_format = 'retina'
# %%
nesvorny_data = pd.read_csv("data/nesvorny_catalog_dataset.csv", index_col=0)
ephem = assist.Ephem("data/assist/linux_m13000p17000.441", "data/assist/sb441-n16.bsp")
# %%
epoch = 2460200.5
epoch_n = 2460200.5 # Nesvorny epoch

TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
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

    time_span = 80e3*np.pi*2
    times = np.linspace(sim.t-time_span/2, sim.t+time_span/2, Nout)
    e = np.zeros(Nout)

    for i, time in enumerate(times):
        sim.integrate(time)
        p = sim.particles[-1]
        orbit = p.orbit()
        e[i] = orbit.e
    return e, times
# %%
# from: highi_1312_vassar_fam3
des = ["K10B40G", "K16NH8S", "K14T98G"]
# %%
ecc = []
t = np.array([])
for d in des:
    row = nesvorny_data[nesvorny_data["Des'n"] == d].iloc[0]
    e, t = get_e(row)
    ecc.append(e)
# %%
time = t/(np.pi*2)/1e3
fig, axs = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(7,2))

colors = ["tab:purple", "tab:pink", "tab:brown"]
downsample=2

### Plot the osculating and proper elements over time
for i in range(len(des)):
    row = nesvorny_data[nesvorny_data["Des'n"] == des[i]].iloc[0]
    # osc
    axs[0].plot(time[::downsample], ecc[i][::downsample], c=colors[i], linewidth=0.8, zorder=0)
    # proper
    axs[0].axhline(row["prope"], c=colors[i], label=row["Des'n"], linestyle=(i, (3, 2)), zorder=10)
    # dot
    axs[0].scatter(time[time.shape[0]//2], ecc[i][time.shape[0]//2],
                   c=colors[i], s=25, edgecolors='black', linewidth=0.75,
                   zorder=20)

    # proper
    axs[1].axhline(row["prope"], c=colors[i], label=row["Des'n"], linestyle=(i, (3, 2)))

    # plot sin with amplitude de and frequency $g$
    # axs[1].plot(t/(np.pi*2), row['de'] * np.sin(t * row['g']/TO_ARCSEC_PER_YEAR))

### Axis labels and legend
axs[0].set_ylabel("Eccentricity")
axs[0].set_xlim(time[0],time[-1])
axs[0].xaxis.set_major_locator(MultipleLocator(20))
axs[0].text(0.5, -0.2, 'Present epoch', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)

axs[1].legend(labelspacing=0.3)
ymin, ymax = 0.160, 0.164
axs[1].set_ylim(ymin, ymax)
axs[1].yaxis.tick_right()
axs[1].yaxis.set_major_locator(MultipleLocator(0.002))

fig.text(0.5, 0.05, 'Time [kyr]', ha='center')

fig.tight_layout()

### Black rectangle on left plot
rect = patches.Rectangle(
    (time[0], ymin),
    time[-1]-time[0],
    ymax-ymin,
    linewidth=0.4,
    edgecolor='k',
    facecolor='none',
    zorder=10
)
axs[0].add_patch(rect)

## Lines connecting the left and right axes
# From: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html
# bbox on the left plot (the whole x axis, and a portion of the y axis defined by ymin and ymax)
bbox = Bbox.from_extents(0, ymin, 1, ymax)
bbox1 = TransformedBbox(bbox, axs[0].get_yaxis_transform())
# bbox on the right plot (the whole plot)
bbox2 = axs[1].bbox

# settings for how to render the bboxes and lines
prop_lines = {}
prop_patches = {
    **prop_lines,
    "alpha": prop_lines.get("alpha", 1) * 0,
    "clip_on": False,
}

# which corners to connect (top right (1) to top left (2) and bottom right (4) to bottom left (3))
loc1a, loc2a, loc1b, loc2b = 1, 2, 4, 3

# connect the bboxes with lines
c1 = BboxConnector(
    bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
c2 = BboxConnector(
    bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

# different background color for the bboxes (disabled)
bbox_patch1 = BboxPatch(bbox1, **prop_patches)
bbox_patch2 = BboxPatch(bbox2, **prop_patches)

p = BboxConnectorPatch(bbox1, bbox2,
                        loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                        **prop_patches)

# render everything
axs[0].add_patch(bbox_patch1)
axs[1].add_patch(bbox_patch2)
axs[1].add_patch(c1)
axs[1].add_patch(c2)
axs[1].add_patch(p)

### Render and save
plt.savefig("plots/proper_elements_explanation.pdf", bbox_inches="tight")
plt.show()
# %%
