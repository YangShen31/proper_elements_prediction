import sys
sys.path.insert(0, 'SBDynT/src')
import sbdynt as sbd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplsbdb import SBDB
from tqdm import tqdm
from time import sleep
# %%
merged_df = pd.read_csv("../data/model_results.csv")

np.random.seed(42)

des = np.random.choice(merged_df["Des'n"].to_list(), size=200, replace=False)

rand_des_list = []
rand_e_sig_list = []
rand_i_sig_list = []

for name in tqdm(des, desc="Querying SBDB"):
    try:
        sbdb = SBDB.query(name, solution_epoch='2461000.5')
        orbit_data = sbdb.get('orbit', {}).get('elements', {})
        if 'e_sig' in orbit_data and 'i_sig' in orbit_data:
            rand_des_list.append(name)
            rand_e_sig_list.append(orbit_data['e_sig'])
            rand_i_sig_list.append(orbit_data['i_sig'])
    except Exception:
        continue
print(f"Successfully retrieved {len(rand_e_sig_list)} random asteroid orbits.")


# %%
query_rand_df = pd.DataFrame(list(zip(rand_des_list, rand_e_sig_list)), columns = ["Des'n", "e_sig"])

# %%
high_des_list = []
high_e_sig_list = []
high_i_sig_list = []

for i in range(0, 30000, 10000):
	des_high = merged_df["Des'n"].to_list()[i:i+10000]

	for name in tqdm(des_high, desc="Querying SBDB"):
		try:
			sbdb = SBDB.query(name, solution_epoch='2460200.5')
			orbit_data = sbdb.get('orbit', {}).get('elements', {})
			if 'e_sig' in orbit_data and 'i_sig' in orbit_data:
				high_des_list.append(name)
				high_e_sig_list.append(orbit_data['e_sig'])
				high_i_sig_list.append(orbit_data['i_sig'])
		except Exception:
			continue
	print(f"Successfully retrieved {len(high_e_sig_list)} asteroid orbits.")

# %%
query_high_df = pd.DataFrame(list(zip(high_des_list, high_e_sig_list)), columns = ["Des'n", "e_sig"])
query_high_df = query_high_df[query_high_df["e_sig"] > 0.002]

# %%
random_list = query_rand_df["Des'n"].to_list()
high_error_list = query_high_df["Des'n"].to_list()

# %%
clones = 30
columns = ["Des'n", "epoch", "x", "y", "z", "vx", "vy", "vz"]
df_random = pd.DataFrame(columns=columns)
for name in random_list[:100]:
	flag, epoch, x,y,z,vx,vy,vz, weights  = sbd.query_sb_from_jpl(des=name,clones=clones)
	df1 = []
	if(flag):
		print("queried %s and returned at epoch %f" % (name,epoch))
		print("cartesian heliocentric position (au), velocity (au/year)")
		print("best-fit orbit:")
		i=0
		print(6*"%15.8e " % (x[i],y[i],z[i],vx[i],vy[i],vz[i]))
		print("cloned orbits:")
		for j in range (1,clones):
			df1.append({
				"Des'n": name,
				"epoch": epoch,
				"no"
				"x": x[j], 
				"y": y[j], 
				"z": z[j], 
				"vx": vx[j], 
				"vy": vy[j], 
				"vz": vz[j]
			})
	df1 = pd.DataFrame(df1)
	df_random = pd.concat([df_random, df1])
	sleep(0.1)

# %%
df_random = df_random[df_random["x"]!= 0].iloc[:1500]
df_random["type"] = "random"
# %%
clones = 30
columns = ["Des'n", "epoch", "x", "y", "z", "vx", "vy", "vz"]
df_high_error = pd.DataFrame(columns=columns)
for name in high_error_list:
	flag, epoch, x,y,z,vx,vy,vz, weights  = sbd.query_sb_from_jpl(des=name,clones=clones)
	df1 = []
	if(flag):
		print("queried %s and returned at epoch %f" % (name,epoch))
		print("cartesian heliocentric position (au), velocity (au/year)")
		print("best-fit orbit:")
		i=0
		print(6*"%15.8e " % (x[i],y[i],z[i],vx[i],vy[i],vz[i]))
		print("cloned orbits:")
		for j in range (1,clones):
			df1.append({
				"Des'n": name,
				"epoch": epoch,
				"x": x[j], 
				"y": y[j], 
				"z": z[j], 
				"vx": vx[j], 
				"vy": vy[j], 
				"vz": vz[j]
			})
	df1 = pd.DataFrame(df1)
	df_high_error = pd.concat([df_high_error, df1])
	sleep(0.1)

# %%
df_high_error = df_high_error[df_high_error["x"]!= 0].iloc[:1500]
df_high_error["type"] = "high_error"
# %%
df_combined = pd.concat([df_random, df_high_error], ignore_index=True, sort=False)
df_combined.to_csv("../data/uncertainty_asteroids_sampled.csv")