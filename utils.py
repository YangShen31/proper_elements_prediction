import rebound as rb
import numpy as np

EARTH_OBLIQUITY = 23.43929111 * np.pi / 180.0 # J2000.0 obliquity in radians
epoch = 2460200.5 # nesvorny epoch
SUN_GM = 0.295912208285591100e-3 # gravitation constant times the sun's mass
UNIT_CONVER = 365.2568983263 / (2 * np.pi) # days in year / (2 * pi)

sun_sim = rb.Simulation()
sun_sim.add("sun", plane="ecliptic", date="JD%f"%epoch)

FWD_ROTATION_MATRIX = np.array([
    [1.0, 0.0, 0.0],
    [0.0, np.cos(EARTH_OBLIQUITY), np.sin(EARTH_OBLIQUITY)],
    [0.0, -np.sin(EARTH_OBLIQUITY), np.cos(EARTH_OBLIQUITY)]
])

BWD_ROTATION_MATRIX = np.array([
    [1.0, 0.0, 0.0],
    [0.0, np.cos(EARTH_OBLIQUITY), -np.sin(EARTH_OBLIQUITY)],
    [0.0, np.sin(EARTH_OBLIQUITY), np.cos(EARTH_OBLIQUITY)]
])

def ecliptic_to_icrf(p: rb.Particle):
    p.xyz = np.array(p.xyz) @ FWD_ROTATION_MATRIX
    p.vxyz = np.array(p.vxyz) @ FWD_ROTATION_MATRIX
    return p

def icrf_to_ecliptic(p: rb.Particle):
    p.xyz = np.array(p.xyz) @ BWD_ROTATION_MATRIX
    p.vxyz = np.array(p.vxyz) @ BWD_ROTATION_MATRIX
    return p

def ecliptic_xyz_to_elements(p: rb.Particle):
    '''
    Particle mass should be in units where the GM for the sun is 1, AU, and yr
    '''
    # tmp = sun_sim.copy()
    # tmp.add(p.copy())
    # return tmp.particles[1].orbit(tmp.particles[0], G=1)
    return p.orbit(primary=sun_sim.particles[0], G=1)