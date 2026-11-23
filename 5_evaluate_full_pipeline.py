# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplsbdb import SBDB
from pathlib import Path
import glob
import io
import warnings
import rebound as rb
import celmech as cm
import assist
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

from linear_theory import linear_theory_prediction, make_simpler_secular_theory
from utils import ecliptic_to_icrf, icrf_to_ecliptic, ecliptic_xyz_to_elements, UNIT_CONVER, epoch as nesvorny_epoch, sun_sim

from tqdm import tqdm

from celmech.nbody_simulation_utilities import get_simarchive_integration_results
import assist
# %%
# Load epehemeris files for ASSIST
ephem = assist.Ephem("data/assist/linux_m13000p17000.441", "data/assist/sb441-n16.bsp")
# %%
def read_family_file(file_path):
		return pd.read_csv(
			file_path,
			header=None,
			names=['propa', 'prope', 'propsini', 'g', 's', 'H', 'NumOpps', 'PackedName', 'UnpackedName']
		)

def query_asteroids_at_epoch(des, epoch):
	# Query horizons to get the position of each asteroid
	df1 = []
	for d in des:
		sim = rb.Simulation()
		sim.add(d, date = "JD%f"%epoch)
		p = sim.particles[-1]
		df1.append({
			"Des'n": d,
			"epoch": epoch,
			"x": p.x, 
			"y": p.y, 
			"z": p.z, 
			"vx": p.vx, 
			"vy": p.vy, 
			"vz": p.vz
		})

	# the list of asteroids we want to identify the families of (in J2000 ecliptic)
	df_e = pd.DataFrame(df1)
	return df_e

def integrate_asterods_to_nesvorny_epoch(df_e, epoch):
	# Use Assist to integrate asteroids back to JD2460200.5

	# Lists to store calculated osculating orbital elements
	a_list = []
	e_list = []
	inc_list = []
	Omega_list = []
	omega_list = []

	# Calculate osculating elements for all asteroids at the nesvorny epoch
	for i in range(len(df_e)):
		ast = df_e.iloc[i]

		### Convert orbital elements in xyz, vxyz ###
		sim = rb.Simulation()
		sim.add("Sun", date = "JD%f"%ast["epoch"])
		sim.add(x = ast["x"].item(),
				y = ast["y"].item(),
				z = ast["z"].item(),
				vx = ast["vx"].item()/UNIT_CONVER,
				vy = ast["vy"].item()/UNIT_CONVER,
				vz = ast["vz"].item()/UNIT_CONVER)
		p = sim.particles[1]
		# Rotate to icrf
		p = ecliptic_to_icrf(p)
		
		### Use ASSIST to integrate backwards to Nesvorny epoch ###
		# Setup sim
		sim_a = rb.Simulation()
		ex = assist.Extras(sim_a, ephem)
		ex.gr_eih_sources = 11
		sim_a.ri_ias15.adaptive_mode = 2
		sim_a.exit_max_distance = 50.0
		sim_a.t = epoch - ephem.jd_ref

		# Add particle
		sim_a.add(p, plane="frame")

		# Integrate
		ex.integrate_or_interpolate(nesvorny_epoch - ephem.jd_ref)
		
		# Retrieve the orbital elements
		p = sim_a.particles[-1]
		p = icrf_to_ecliptic(p)
		p.vxyz = np.array(p.vxyz) * UNIT_CONVER
		orbit = ecliptic_xyz_to_elements(p, sun_sim.particles[0])
		
		a_list.append(orbit.a)
		e_list.append(orbit.e)
		inc_list.append(orbit.inc)
		Omega_list.append(orbit.Omega)
		omega_list.append(orbit.omega)

	# Extract only the orbital elments we need into a separate dataframe that is used with the ML model
	merged_df = pd.DataFrame({"Des'n": df_e["Des'n"],
							'a': np.abs(a_list), 
							'e': e_list, 
							'Incl.': inc_list, 
							'Node': Omega_list, 
							'Peri.': omega_list})
	return merged_df

def calculate_proper_a(merged_df, conver = 58.132440867049, epoch = 2460200.5):
	# Estimate the proper semi-major axis for family classification of the selected asteroids

	sun = sun_sim.particles[0]
	def propa_calc(row):
		# Convert Nesvorny orbital elements into xyz, vxyz
		sim = rb.Simulation()
		ps = sim.particles
		sim.add(primary=sun, 
			a=row['a'], 
			e=row['e'], 
			inc=row['Incl.'], 
			Omega=row['Node'], 
			omega=row['Peri.'])
		p = sim.particles[-1]
		p = ecliptic_to_icrf(sim.particles[-1])
		p.vxyz = np.array(p.vxyz) / UNIT_CONVER

		# ASSIST integration
		Nout = int(1e3*np.pi)
		sim = rb.Simulation()
		ex = assist.Extras(sim, ephem)
		sim.t = epoch - ephem.jd_ref
		sim.ri_ias15.adaptive_mode = 2
		
		# We are doing the unit conversion ourself (using the conver variable)
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				message=r"Particle added is from a simulation that uses different units\.",
				category=RuntimeWarning,
			)
			sim.add(p)
		
		p = icrf_to_ecliptic(p)
		times = np.linspace(sim.t, 3e5 + sim.t, Nout)
		a = np.zeros(Nout)

		# Get the solar ephemeris for Sun's position at different times
		def sun_state_at_time(t):
			sun = ephem.get_particle("sun", t)
			sun = icrf_to_ecliptic(sun)
			return np.array([
				sun.x, sun.y, sun.z,
				sun.vx * UNIT_CONVER, sun.vy * UNIT_CONVER, sun.vz * UNIT_CONVER
			])

		sun_states = np.array([sun_state_at_time(t) for t in times])
		
		for i, time in enumerate(times):
			sim.integrate(time)
			p = sim.particles[-1]
			p = icrf_to_ecliptic(p)
			p.vxyz = np.array(p.vxyz) * conver
			sx, sy, sz, svx, svy, svz = sun_states[i]
			sun_now = rb.Particle(x=sx, y=sy, z=sz,
								vx=svx, vy=svy, vz=svz, m=1.0)
			orbit = ecliptic_xyz_to_elements(p, sun_now)
			p.vxyz = np.array(p.vxyz) / conver
			p = ecliptic_to_icrf(p)
			a[i] = orbit.a

		des = row["Des'n"]

		return np.mean(a)

	merged_df["propa"] = merged_df.apply(propa_calc, axis=1)
	return merged_df

def predict_linear_elements(merged_df):
	merged_df = merged_df.copy()
	# Linear proper element prediction
	# setup simulation that is used to rotate candidate asteroids into the invariable frame
	ref_sim = rb.Simulation()

	ref_sim.add("Sun", date="JD%f" % nesvorny_epoch)
	ref_sim.add("Jupiter", date="JD%f" % nesvorny_epoch)
	ref_sim.add("Saturn", date="JD%f" % nesvorny_epoch)
	ref_sim.add("Uranus", date="JD%f" % nesvorny_epoch)
	ref_sim.add("Neptune", date="JD%f" % nesvorny_epoch)
	ref_sim.move_to_com()

	L = ref_sim.angular_momentum()
	x_new = rb.Vec3d(L.y, -L.x, 0.0)

	# create a rotation we can apply later
	rot = rb.Rotation.to_new_axes(newz=L,newx=x_new)

	def rotate_to_invariable_frame(ast):
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
			a=ast["a"],
			e=ast["e"],
			inc=ast["Incl."],
			Omega=ast["Node"],
			omega=ast["Peri."],
			primary=sim.particles[0],
			date="JD%f" % nesvorny_epoch)
		
		sim.rotate(rot)

		p = sim.particles[-1]
		orb = p.orbit(primary=sim.particles[0])
		return orb

	# Load files for linear theory
	simpler_secular_theory = make_simpler_secular_theory()

	linear_invariable_elements = []

	# Calculate linear estimate for each candidate asteroid
	for i,ast in merged_df.iterrows():
		# get the asteroid's osculating elements in the invariable frame
		orb = rotate_to_invariable_frame(ast)
		# Calculate linear theory
		u0, v0, g0, s0 = linear_theory_prediction(orb.e, orb.inc, orb.pomega, orb.Omega, orb.a, simpler_secular_theory)

		linear_invariable_elements.append({
			'prope_linear':    abs(complex(u0)),
			'propsini_linear': np.sin(abs(complex(v0))),
			'g0':              g0,
			's0':              s0,
			'a':               orb.a,
			'e':               orb.e,
			'Incl.':           np.degrees(orb.inc),
			'Peri.':           np.degrees(orb.omega),
			'Node':            np.degrees(orb.Omega),
		})

	linear_invariable_df = pd.DataFrame(linear_invariable_elements)
	merged_df[linear_invariable_df.columns] = linear_invariable_df
	return merged_df

def predict_proper_elements(merged_df):
	# Predict the Proper Elements
	# Calculate feature vector for each asteroid
	node = np.deg2rad(merged_df["Node"])
	peri = np.deg2rad(merged_df["Peri."])
	inc = np.deg2rad(merged_df["Incl."])

	merged_df["ecospo"] = merged_df["e"] * np.cos(node + peri)
	merged_df["esinpo"] = merged_df["e"] * np.sin(node + peri)
	merged_df["sinicosO"] = np.sin(inc) * np.cos(node)
	merged_df["sinisinO"] = np.sin(inc) * np.sin(node)

	# Load trained models
	final_model_e = xgb.XGBRegressor()
	final_model_e.load_model("data/models/best_model_e_final.xgb")
	final_model_inc = xgb.XGBRegressor()
	final_model_inc.load_model("data/models/best_model_inc_final.xgb")

	# Use models to estimate proper elements, which will be stored in "pred_e" and "pred_sini"
	features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'prope_linear']
	features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'propsini_linear']

	dele = final_model_e.predict(merged_df[features_e])
	delsini = final_model_inc.predict(merged_df[features_inc])

	merged_df["pred_e"] = merged_df["e"] + dele
	merged_df["pred_sini"] = np.sin(np.deg2rad(merged_df["Incl."])) + delsini

	return merged_df

# Family Classification
def calculate_d(a_p, delta_a_p, delta_e_p, delta_sin_i_p):
	"""
	Arguments:
		a_p: average semi-major axis for the two asteroids
		delta_a_p: difference in semi-major axis between the two asteroids
		delta_e_p: difference in eccentricity axis between the two asteroids
		delta_sin_i_p: difference in sin(inclination) axis between the two asteroids
	"""
	numerator = 3e4  # 3 × 10^4 m/s
	denominator = np.sqrt(a_p)
	term1 = (delta_a_p / a_p) ** 2
	term2 = 2 * (delta_e_p ** 2)
	term3 = 2 * (delta_sin_i_p ** 2)
	inside_sqrt = (5 / 4) * term1 + term2 + term3
	d = (numerator / denominator) * np.sqrt(inside_sqrt)
	return d

def prep_family_data():
	df_d = pd.read_csv("data/d_cutoff_80_percent.csv")

	# "d Nesvorny" is the d threshold from the Nesvorny paper
	# "d(80% Precision)" is the d threshold that achieves 80% precision
	# "d(80% Percent)" is the d threshold that achieves 80% percent detected
	desired_d = "d(80% Percent)"
	# desired_d = "d(Last 80% Precision)"


	filenames = list(Path('data/family_tables').glob('*.csv'))

	family_file_map = {
		f.stem.split("_")[1]: f
		for f in filenames
	}
	# handle both families with the number 0
	del family_file_map['0']
	family_file_map['inner_0'] = Path('data/family_tables/inner_0_2012pm61_fam3.csv')
	family_file_map['middle_0'] = Path('data/family_tables/middle_0_2006qx49_fam3.csv')

	family_name_map = (
		df_d.assign(Number=df_d["Number"].astype(str))
			.set_index("Number")["Family Name"]
			.to_dict()
	)

	# turn df_d into a dictionary so we can more easily lookup thresholds
	d_cutoff_map = (
		df_d.assign(Number=df_d["Number"].astype(str))
			.set_index("Number")[desired_d] # use the desired d value selected in the cell above
			.to_dict()
	)

	df_nesvorny = pd.read_csv("data/nesvorny_catalog_dataset.csv", dtype={"Des'n": 'str'})
	df_lookup = df_nesvorny[["Des'n", "propa", "prope", "propsini"]].copy()
	# remove the asteroids we're evaluating from the comparison list
	df_lookup = df_lookup[~df_lookup["Des'n"].isin(des)]

	# gather all data for each family
	family_data = {}
	for family_number, file_path in family_file_map.items():
		if family_number not in d_cutoff_map:
			continue

		df_family = read_family_file(file_path)
		df_family = df_family[["PackedName"]]

		fam = df_family.merge(
			df_lookup,
			left_on="PackedName",
			right_on="Des'n",
			how="inner"
		)

		if fam.empty:
			continue

		family_data[family_number] = {
			"a": fam["propa"].to_numpy(float),
			"e": fam["prope"].to_numpy(float),
			"sini": fam["propsini"].to_numpy(float),
			"d_cutoff": d_cutoff_map[family_number],
		}
	
	# Find the distance between each candidate asteroid and every family
	fam_main_asteroids = []

	for filename in filenames:
		# Load the family
		df_family = pd.read_csv(filename, header=None, names=['propa', 'prope', 'propsini', 'g', 's', 'H',
				'NumOpps', 'PackedName', 'UnpackedName'])
		family_number = str(filename).split("/")[-1].split(".")[0].split("_")[1]

		# get the designator of the leading asteroid (which we assume is the first)
		desn = df_family.iloc[0, 7]

		# get the nbody proper elements of the leading asteroid
		ast_f = df_nesvorny[df_nesvorny["Des'n"] == str(desn)]
		fam_main_asteroids.append([family_number, ast_f['propa'].values[0], ast_f['prope'].values[0], ast_f['propsini'].values[0]])

	family_data_df = pd.DataFrame(fam_main_asteroids, columns=["family_number", "propa", "prope", "propsini"])
	family_data_knn=NearestNeighbors(n_neighbors=5,
						algorithm='auto',
						metric=lambda a,b: calculate_d((a[0]+b[0])/2, a[0]-b[0], a[1]-b[1], a[2]-b[2]) # 0: propa, 1: prope, 2: propsini
	)
	family_data_knn.fit(family_data_df[["propa", "prope", "propsini"]], family_data_df["family_number"])

	return family_name_map, family_data, family_data_knn, family_data_df
# %%
filenames = list(Path('data/family_tables').glob('*.csv'))

family_file_map = {
	f.stem.split("_")[1]: f
	for f in filenames
}
# handle both families with the number 0
del family_file_map['0']
family_file_map['inner_0'] = Path('data/family_tables/inner_0_2012pm61_fam3.csv')
family_file_map['middle_0'] = Path('data/family_tables/middle_0_2006qx49_fam3.csv')

df_merged_elements = pd.read_csv("data/merged_elements.csv", dtype={"Des'n": 'str'})
df_lookup = df_merged_elements[["Des'n", "test_set"]].copy()

# gather all data for each family
family_data = {}
for family_number, file_path in family_file_map.items():

	df_family = read_family_file(file_path)
	df_family = df_family[["PackedName"]]

	fam = df_family.merge(
		df_lookup,
		left_on="PackedName",
		right_on="Des'n",
		how="inner"
	)

	if fam.empty:
		print("skipped:", file_path)
		continue

	family_data[family_number] = fam["Des'n"][fam["test_set"] == 1]

true_families = []
des = []

# pick one random asteroid from the test set in each family
for no, desns in family_data.items():
	# make sure we pick one which starts with a "K" so it can be looked up in the horizons database
	desns = desns[desns.str.startswith('K')]
	if len(desns) == 0:
		print(f"skipped {no}. no members in the test set")
		continue
	true_families.append(no)
	des.append(desns.sample(1, random_state=1).to_list()[0])
# %%
epoch = 2460400.5
df_e = query_asteroids_at_epoch(des, epoch)

merged_df = integrate_asterods_to_nesvorny_epoch(df_e, epoch)

merged_df = calculate_proper_a(merged_df)

merged_df = predict_linear_elements(merged_df)

merged_df = predict_proper_elements(merged_df)

family_name_map, family_data, family_data_knn, family_data_df = prep_family_data()
# %%

# Now compare each candidate asteroid against all the family members in the top 5 families it is closest to
# if it is closer than the d threshold of any family, we assume it is in that family

top5_families_all = []
matched_families_all = []

for i,ast in merged_df.iterrows():
	# get the row in merged_df which corresponds to this candidate (assume there is only 1)
	ast_a = ast["propa"]
	ast_prope = ast["pred_e"]
	ast_propsini = ast["pred_sini"]

	# get the 5 closest families from the KNN
	top5_families_idxs = family_data_knn.kneighbors(
		[ast[["propa", "pred_e", "pred_sini"]].to_list()],
		5,
		return_distance=False
	)
	top5_families = family_data_df["family_number"][top5_families_idxs[0]]
	top5_families_all.append(top5_families.to_list())


	matched_families = []
	# print("\n", ast["Des'n"])
	for family_number in top5_families:

		if family_number not in family_data:
			continue

		fam = family_data[family_number]

		da = fam["a"] - ast_a
		de = fam["e"] - ast_prope
		dsini = fam["sini"] - ast_propsini

		d = calculate_d(ast_a, da, de, dsini)

		# print(family_number, np.min(d), fam["d_cutoff"])

		if np.min(d) <= fam["d_cutoff"]:
			matched_families.append(family_number)

	matched_families_all.append(matched_families)

results_df = pd.DataFrame({"Des'n": merged_df["Des'n"].tolist(), "matched_families":matched_families_all, "top5_families": top5_families_all})
results_df
# %%
results_df["true_family"] = true_families

n_correct = 0
n_top_5 = 0
for r, ast in results_df.iterrows():
	if ast['true_family'] in ast['top5_families']:
		n_top_5 += 1
	if len(ast['matched_families']) > 1:
		print(f"\nmultiple families matched to asteroids {ast["Des'n"]} ({ast['true_family']})")
		print([family_name_map[f] for f in ast['matched_families']], ast['matched_families'])
		if ast['true_family'] in ast['matched_families']:
			n_correct += 1
	if len(ast['matched_families']) == 1 and ast['matched_families'][0] == ast['true_family']:
		n_correct += 1

print()
print(f"Accuracy (top-5): {n_top_5/len(results_df):.3f}")
print(f"Accuracy (top-1): {n_correct/len(results_df):.3f}")
# %%
# check the asteroids that were assigned to multiple families
for i,f in enumerate(['63530', '37981']): # same family
# for i,f in enumerate(['29880', '5228']): # different families
# for i,f in enumerate(['4471', '392']): # different families
    df_family = read_family_file(family_file_map[f])
    fam_prope = df_family["prope"]
    fam_propsini = df_family["propsini"]
    plt.scatter(fam_prope, fam_propsini, s = 50-25*i, label = family_name_map[f])
plt.legend()
# %%
# visualize first 5 asteroids for debugging

# Plot the detected asteroids with their family in the proper elements phase space
n = 5
fig, axes = plt.subplots(n, 1, figsize=(10, int(n*5)))

for i in range(n):
	family_number = results_df.iloc[i]["matched_families"]
	### RIGHT PLOT
	# plot the family we identified the asteroid with
	axins = axes[i].inset_axes([1.2, 0, 0.7, 0.7])
	if len(family_number) != 0:
		df_family = read_family_file(family_file_map[family_number[0]])
		fam_prope = df_family["prope"]
		fam_propsini = df_family["propsini"]
		axins.scatter(fam_prope, fam_propsini, c = "grey", s = 10, alpha = 0.5, label = "Family members")
	# plot the candidate asteroid
	axins.scatter(merged_df.iloc[i]["pred_e"], merged_df.iloc[i]["pred_sini"], c = "black", marker='*', s = 100)
	axins.scatter(merged_df.iloc[i]["pred_e"], merged_df.iloc[i]["pred_sini"], c = "red", marker='*', s = 35, label = merged_df.iloc[i]["Des'n"])
	axins.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	axins.set_aspect(1)

	# connect the inset and large plots
	axes[i].indicate_inset_zoom(axins, edgecolor="black")

	### LEFT PLOT
	# plot all potential families
	top5_families = results_df.iloc[i]["top5_families"]
	for f in top5_families:
		df_family = read_family_file(family_file_map[f])
		fam_prope = df_family["prope"]
		fam_propsini = df_family["propsini"]
		axes[i].scatter(fam_prope, fam_propsini, s = 10, alpha = 0.3, label = family_name_map[f])

	# plot the asteroid
	axes[i].scatter(merged_df.iloc[i]["pred_e"], merged_df.iloc[i]["pred_sini"], c = "black", marker='*', s = 200)
	axes[i].scatter(merged_df.iloc[i]["pred_e"], merged_df.iloc[i]["pred_sini"], c = "red", marker='*', s = 100, label = merged_df.iloc[i]["Des'n"])

	leg = axes[i].legend(loc="best")
	for lh in leg.legend_handles: 
		lh.set_alpha(1)

	axes[i].set_aspect(1)

plt.tight_layout(pad=2)
# %%
