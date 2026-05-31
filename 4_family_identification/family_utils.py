import pandas as pd
import numpy as np

d_values = np.linspace(0, 2000, 2000)

def calculate_d(a_p, delta_a_p, delta_e_p, delta_sin_i_p):
	numerator = 3e4  # 3 × 10^4 m/s
	denominator = np.sqrt(a_p)
	term1 = (delta_a_p / a_p) ** 2
	term2 = 2 * (delta_e_p ** 2)
	term3 = 2 * (delta_sin_i_p ** 2)
	inside_sqrt = (5 / 4) * term1 + term2 + term3
	d = (numerator / denominator) * np.sqrt(inside_sqrt)
	return d

# create a slab enclosing all family members
def slab_d_calc(family_df_copy, family_pred_df, merged_df, command):
	if command == "osculating":
		columns_bound = {"a": "a", "e": "e", "sini": "sini"}
		columns = {"a": "a", "e": "e", "sini": "sini"}
	elif command == "proper":
		columns_bound = {"a": "propa", "e": "prope", "sini": "propsini"}
		columns = {"a": "propa", "e": "prope", "sini": "propsini"}
	elif command == "pred":
		columns_bound = {"a": "propa", "e": "prope", "sini": "propsini"}
		columns = {"a": "propa", "e": "pred_e", "sini": "pred_sini"}

	# Add paddding to box increasing it by a factor of the box size or a constant amount
	# a_adds = (family_df_copy[columns_bound["a"]].max() - family_df_copy[columns_bound["a"]].min())/2
	# e_adds = (family_df_copy[columns_bound["e"]].max() - family_df_copy[columns_bound["e"]].min())/2
	# sini_adds = (family_df_copy[columns_bound["sini"]].max() - family_df_copy[columns_bound["sini"]].min())/2

	a_adds = 0
	e_adds = 0.001
	sini_adds = 0.001

	# Calculate the bounds of the box and build a slab
	a_min, a_max = family_df_copy[columns_bound["a"]].min() - a_adds, family_df_copy[columns_bound["a"]].max() + a_adds
	e_min, e_max = min((float(family_df_copy[columns_bound["e"]].min()), float(family_pred_df["pred_e"].min()))) - e_adds, max((float(family_df_copy[columns_bound["e"]].max()), float(family_pred_df["pred_e"].max()))) + e_adds
	sini_min, sini_max = min(float(family_df_copy[columns_bound["sini"]].min()), float(family_pred_df["pred_sini"].min())) - sini_adds, max(float(family_df_copy[columns_bound["sini"]].max()), float(family_pred_df["pred_sini"].max())) + sini_adds

	slab_df = merged_df[
		(merged_df[columns["a"]] >= a_min) & (merged_df[columns["a"]] <= a_max) &
		(merged_df[columns["e"]] >= e_min) & (merged_df[columns["e"]] <= e_max) &
		(merged_df[columns["sini"]] >= sini_min) & (merged_df[columns["sini"]] <= sini_max)
	]

	a_family, e_family, sini_family, names = family_df_copy[columns_bound["a"]].values, family_df_copy[columns_bound["e"]].values, family_df_copy[columns_bound["sini"]].values, family_df_copy["Des'n"].values

	d_results = []
	for idx, row in slab_df.iterrows():
		a = row[columns["a"]]
		e = row[columns["e"]]
		sini = row[columns["sini"]]
		name = row["Des'n"]
		
		for a_f, e_f, sini_f, name_f in zip(a_family, e_family, sini_family, names):
			if name == name_f:
				continue
			else:
				da = a_f - a
				de = e_f - e
				dsini = sini_f - sini
				d = calculate_d(a, da, de, dsini)
				d_results.append({
					"name_asteroid": name,
					"name_family_asteroid": name_f,
					"d": d
				})
	d_df = pd.DataFrame(d_results)

	return d_df