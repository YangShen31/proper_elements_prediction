# %%
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import rebound as rb
from reboundx import constants as rbx_constants
from celmech.secular import LaplaceLagrangeSystem
from celmech.nbody_simulation_utilities import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
# %%
def load_sim(path, asteroid_mass=1e-11):
    rb_sim = rb.Simulationarchive(path)
    results = get_simarchive_integration_results(str(path), coordinates='heliocentric')
    
    masses = np.array(list(map(lambda ps: ps.m, rb_sim[0].particles))[1:-1] + [asteroid_mass], dtype=np.float64)
    m = masses[..., None].repeat(results['a'].shape[1], axis=-1)
    G = 1
    beta = ((1 * m) / (1 + m))
    mu = G * (1 + m)
    results['Lambda'] = beta * np.sqrt(mu * results['a'])
    
    M = results['l'] - results['pomega']
    results['lambda'] = M + results['pomega']

    results['x'] = np.sqrt(results['Lambda']) * np.sqrt(1 - np.sqrt(1-results['e']**2)) * np.exp(1j * results['pomega'])
    results['y'] = np.sqrt(2 * results['Lambda']) * np.power(1-results['e']**2, 1/4) * np.sin(results['inc']/2) * np.exp(1j * results['Omega'])

    # coordinate pairs are:
    # - Lambda, Lambda
    # - x, -i * x_bar
    # - y, -i * y_bar
    return rb_sim, results
dataset_path = Path('../data') / 'uncertainty_integrations'
TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
# %%
sims = list(dataset_path.glob('*.sa'))
desns = list(map(lambda p: p.stem.split('_')[-1], sims))
desns = sorted(desns)
# %%
asteroid_elements = []
for desn in tqdm(desns):
    # %%
    sim_pth = str(dataset_path / f"asteroid_integration_{desn}.sa")
    sim, sim_data = load_sim(sim_pth)
    
    fs_arcsec_per_yr = (TO_ARCSEC_PER_YEAR / np.gradient(sim_data['time']).mean()) * 2 * np.pi
    print("sample rate (\"/yr):", fs_arcsec_per_yr)
    print("dt (yr):", np.gradient(sim_data['time']).mean() / (2 * np.pi))
    print("total samples:", len(sim_data['time']))
    # %%
    initial_sim = sim[0]
    # initial_sim.particles[5].m = 1e-11
    initial_sim.remove(5)
    # %%
    lsys = LaplaceLagrangeSystem.from_Simulation(initial_sim)
    lsys.add_general_relativity_correction(rbx_constants.C) # add GR correction

    # calculate rotation matricies and reorder entries to attempt preserve mode order
    ecc_rotation_matrix_T, ecc_eigval = lsys.diagonalize_eccentricity()
    ecc_eigval = np.diag(ecc_eigval)
    ecc_eigguess = np.diag(lsys.Neccentricity_matrix)
    _, ecc_order = linear_sum_assignment(np.abs(ecc_eigval[..., None] - ecc_eigguess[None, ...]).T)
    ecc_rotation_matrix_T = ecc_rotation_matrix_T[:,ecc_order]
    full_ecc_rotation_matrix_T = np.eye(5)
    full_ecc_rotation_matrix_T[0:4, 0:4] = ecc_rotation_matrix_T

    inc_rotation_matrix_T, inc_eigval = lsys.diagonalize_inclination()
    inc_eigval = np.diag(inc_eigval)
    inc_eigguess = np.diag(lsys.Ninclination_matrix)
    _, inc_order = linear_sum_assignment(np.abs(inc_eigval[..., None] - inc_eigguess[None, ...]).T)
    inc_rotation_matrix_T = inc_rotation_matrix_T[:,inc_order]
    full_inc_rotation_matrix_T = np.eye(5)
    full_inc_rotation_matrix_T[0:4, 0:4] = inc_rotation_matrix_T
    # %%
    Phi = (np.linalg.inv(full_ecc_rotation_matrix_T) @ sim_data['x'])
    Psi = (np.linalg.inv(full_inc_rotation_matrix_T) @ sim_data['y'])

    planets = ("Jupiter","Saturn","Uranus","Neptune")
    planet_ecc_fmft = dict()
    planet_inc_fmft = dict()
    for i,pl in enumerate(planets + ("Asteroid",)):
        planet_ecc_fmft[pl] = fmft(sim_data['time'],Phi[i],14)
        planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
        planet_e_freqs_arcsec_per_yr = planet_e_freqs * TO_ARCSEC_PER_YEAR

        planet_inc_fmft[pl] = fmft(sim_data['time'],Psi[i],14)
        planet_inc_freqs = np.array(list(planet_inc_fmft[pl].keys()))
        planet_inc_freqs_arcsec_per_yr = planet_inc_freqs * TO_ARCSEC_PER_YEAR

        print("")
        print(pl)
        print("g")
        print("-------")
        for g in planet_e_freqs[:6]:
            print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(planet_ecc_fmft[pl][g])))
        print("s")
        print("-------")
        for s in planet_inc_freqs[:6]:
            print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(planet_inc_fmft[pl][s])))
    # %%
    g_vec = np.zeros(5)
    s_vec = np.zeros(5)

    g_vec[0] = list(planet_ecc_fmft['Jupiter'].keys())[0]
    g_vec[1] = list(planet_ecc_fmft['Saturn'].keys())[0]
    g_vec[2] = list(planet_ecc_fmft['Uranus'].keys())[0]
    g_vec[3] = list(planet_ecc_fmft['Neptune'].keys())[0]

    for g in list(planet_ecc_fmft['Asteroid'].keys()):
        if np.min(np.abs(g_vec[:4] - g)*TO_ARCSEC_PER_YEAR) > 0.1:
            g_vec[4] = g
            break

    s_vec[0] = list(planet_inc_fmft['Jupiter'].keys())[0]
    s_vec[1] = list(planet_inc_fmft['Saturn'].keys())[0]
    s_vec[2] = list(planet_inc_fmft['Uranus'].keys())[0]
    s_vec[3] = list(planet_inc_fmft['Neptune'].keys())[0]

    for s in list(planet_inc_fmft['Asteroid'].keys()):
        if np.min(np.abs(s_vec[:4] - s)*TO_ARCSEC_PER_YEAR) > 0.1:
            s_vec[4] = s
            break

    g_amp = np.zeros(5, dtype=np.complex128)
    s_amp = np.zeros(5, dtype=np.complex128)

    g_amp[0] = planet_ecc_fmft['Jupiter'][g_vec[0]]
    g_amp[1] = planet_ecc_fmft['Saturn'][g_vec[1]]
    g_amp[2] = planet_ecc_fmft['Uranus'][g_vec[2]]
    g_amp[3] = planet_ecc_fmft['Neptune'][g_vec[3]]
    g_amp[4] = planet_ecc_fmft['Asteroid'][g_vec[4]]

    s_amp[0] = planet_inc_fmft['Jupiter'][s_vec[0]]
    s_amp[1] = planet_inc_fmft['Saturn'][s_vec[1]]
    s_amp[2] = planet_inc_fmft['Uranus'][s_vec[2]]
    s_amp[3] = planet_inc_fmft['Neptune'][s_vec[3]]
    s_amp[4] = planet_inc_fmft['Asteroid'][s_vec[4]]

    omega_vec = np.concat([g_vec, s_vec])
    omega_amp = np.concat([g_amp, s_amp])

    print(omega_vec * TO_ARCSEC_PER_YEAR)
    print(omega_amp)
    # %%
    g = omega_vec[4] * TO_ARCSEC_PER_YEAR
    s = omega_vec[9] * TO_ARCSEC_PER_YEAR

    x_amp = np.abs(omega_amp[4]) / full_ecc_rotation_matrix_T[4,4]
    y_amp = np.abs(omega_amp[9]) / full_inc_rotation_matrix_T[4,4]

    a = sim_data['a'][4].mean()
    Lambda = sim_data['Lambda'][4].mean()
    e = (x_amp * np.sqrt(2*Lambda - x_amp**2)) / Lambda
    sini = np.sin(2 * np.arcsin(y_amp / (np.sqrt(2*Lambda) * np.power(1-e**2, 1/4))))

    a_elements = [desn, float(g), float(s), float(a), float(e), float(sini)]
    print(a_elements)
    asteroid_elements.append(a_elements)
# %%
df = pd.DataFrame(asteroid_elements, columns=["Des'n", "g", "s", "propa", "prope", "propsini"])
df.to_csv("../data/uncertainty_asteroid_elements_proper.csv")
# %%
