import linear_theory.test_particle_secular_hamiltonian
from linear_theory.test_particle_secular_hamiltonian import linear_theory_prediction, SyntheticSecularTheory

import sys
sys.modules['test_particle_secular_hamiltonian'] = linear_theory.test_particle_secular_hamiltonian

import pickle
import os
import numpy as np

def make_simpler_secular_theory():
    try:
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/solar_system_synthetic_solution.bin","rb") as fi:
            solar_system_synthetic_theory=pickle.load(fi)
    except Exception as e:
        print("Cannot find solar_system_synthetic_solution.bin. Please run OuterSolarSystemSyntheticSecularSolution.ipynb first")
        raise Exception()

    truncate_dictionary = lambda d,tol: {key:val for key,val in d.items() if np.abs(val)>tol}
    simpler_secular_theory = SyntheticSecularTheory(
        solar_system_synthetic_theory.masses,
        solar_system_synthetic_theory.semi_major_axes,
        solar_system_synthetic_theory.omega_vector,
        [truncate_dictionary(x_d,1e-3) for x_d in solar_system_synthetic_theory.x_dicts],
        [truncate_dictionary(y_d,1e-3) for y_d in solar_system_synthetic_theory.y_dicts]
    )
    return simpler_secular_theory