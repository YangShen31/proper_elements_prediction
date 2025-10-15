# %%
import sys
#change the next line to reflect where you have downloaded the source code
sys.path.insert(0, './SBDynT/src')
import sbdynt as sbd

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
merged_df = pd.read_csv("../data/nesvorny_catalog_dataset.csv")
des = merged_df["Des'n"].to_list()[-10000:]

# %%
from astroquery.jplsbdb import SBDB

e_sig_list = []
i_sig_list = []

for i in range(10000):
	sbdb = SBDB.query(des[i])
	e_sig_list.append(sbdb['orbit']['elements']['e_sig'])
	i_sig_list.append(sbdb['orbit']['elements']['i_sig'])

# %%
query_df = pd.DataFrame(list(zip(des, e_sig_list)), columns = ["Des'n", "e_sig"])
query_df = query_df[query_df["e_sig"] > 0.002]

# %%
high_error_list = query_df["Des'n"].to_list()

# %%
# example with 5 clones, 
# the first index on the returned variables is best fit, 
# followed by 5 clones sampled from the covariance matrix
clones = 100
columns = ["Des'n", "epoch", "x", "y", "z", "vx", "vy", "vz"]
df = pd.DataFrame(columns=columns)
for name in high_error_list[:100]:
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
	df = pd.concat([df, df1])

# %%
df = df[df["x"]!= 0]
df.to_csv("../data/uncertainty_asteroids_sampled.csv")
len(df)
# %%
