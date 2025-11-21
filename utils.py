import rebound as rb
import numpy as np

# J2000.0 obliquity in radians
EARTH_OBLIQUITY = 23.43929111 * np.pi / 180.0

SUN_GM = 0.295912208285591100e-3

FWD_ROTATION_MATRIX = np.array([
    [1.0, 0.0, 0.0],
    [0.0, np.cos(EARTH_OBLIQUITY), -np.sin(EARTH_OBLIQUITY)],
    [0.0, np.sin(EARTH_OBLIQUITY), np.cos(EARTH_OBLIQUITY)]
])

BWD_ROTATION_MATRIX = np.array([
    [1.0, 0.0, 0.0],
    [0.0, np.cos(EARTH_OBLIQUITY), np.sin(EARTH_OBLIQUITY)],
    [0.0, -np.sin(EARTH_OBLIQUITY), np.cos(EARTH_OBLIQUITY)]
])

def ecliptic_to_icrf(p: rb.Particle):
    p.xyz = np.array(p.xyz) @ FWD_ROTATION_MATRIX
    p.vxyz = np.array(p.vxyz) @ FWD_ROTATION_MATRIX
    return p

def icrf_to_ecliptic(p: rb.Particle):
    p.xyz = np.array(p.xyz) @ BWD_ROTATION_MATRIX
    p.vxyz = np.array(p.vxyz) @ BWD_ROTATION_MATRIX
    return p

def ecliptic_xyz_to_elements(p):
    '''
    Particle mass should be in units where the GM for the sun is 1, AU, and yr
    '''
    sim = rb.Simulation()
    sim.add("sun", plane='frame')
    sim.add(p)
    ps = sim.particles
    return ps[1].orbit(primary=ps[0])