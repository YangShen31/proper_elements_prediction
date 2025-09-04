from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
import time
from sklearn.tree import DecisionTreeRegressor

from hadden_theory.test_particle_secular_hamiltonian import SyntheticSecularTheory, TestParticleSecularHamiltonian, calc_g0_and_s0
from hadden_theory import test_particle_secular_hamiltonian
# hack to make pickle load work
import sys
sys.modules['test_particle_secular_hamiltonian'] = test_particle_secular_hamiltonian

try:
	plt.style.use('/Users/dtamayo/.matplotlib/paper.mplstyle')
except:
	pass

from pathlib import Path
Path("tables_for_analysis").mkdir(exist_ok=True)

Path("models").mkdir(exist_ok=True)

# Read Nesvorny catalog dataset
nesvorny_df = pd.read_csv("nesvorny_catalog_dataset.csv")

# Read linear prediction results
prediction_path = Path("linear_predictions")
file_names = list(prediction_path.glob("*.npz"))
rows = []

for i in range(len(file_names)):
	soln_h = np.load(file_names[i])
	prope_value = soln_h["u"]
	propsini_value = soln_h["v"]
	g0_value = soln_h["g"]
	s0_value = soln_h["s"]
	des_n = file_names[i].stem.replace("integration_results_", "")
	rows.append([des_n, prope_value, propsini_value, g0_value, s0_value])

df_h = pd.DataFrame(rows, columns=["Des'n", "prope_h", "propsini_h", "g0", "s0"])

# Get merged dataframe for later model training
merged_df = pd.merge(nesvorny_df, df_h, on="Des'n", how="inner")

features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 'g0', 'prope_h']
features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 's0', 'propsini_h']
data_e = merged_df[features_e]
data_inc = merged_df[features_inc]
dela = merged_df['propa']-merged_df['a']
dele = merged_df['prope']-merged_df['e']
delsini = merged_df['propsini']-np.sin(merged_df['Incl.']*np.pi/180)
delg = merged_df['g0'] - merged_df['g']
s = merged_df['s']

trainX_e, testX_e, trainX_inc, testX_inc, trainY_e, testY_e, trainY_inc, testY_inc = train_test_split(data_e, data_inc, dele, delsini, test_size=0.4, random_state=42)
valX_e, testX_e, valX_inc, testX_inc, valY_e, testY_e, valY_inc, testY_inc = train_test_split(testX_e, testX_inc, testY_e, testY_inc, test_size=0.5, random_state=42)
space = {
	'max_depth': hp.qloguniform('x_max_depth', np.log(5), np.log(40), 1),
	'minibatch_frac': hp.uniform('minibatch_frac', 0.1, 1.0),
	'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3))
}

# Set of tuned hyperparameters
best_ecc = {'minibatch_frac': 0.497938840723922, 'learning_rate': 0.0403467366109875, 'max_depth': 22.0}
best_inc = {'minibatch_frac': 0.4670652921193636, 'learning_rate': 0.04872472351951210, 'max_depth': 14.0}

# Train the model for the proper eccentricity
# def objective(params):
# 	clf = NGBRegressor(
# 		Dist=Normal,
# 		Score=LogScore,
# 		verbose=False,
# 		minibatch_frac=params['minibatch_frac'],
# 		n_estimators=200,
# 		learning_rate=params['learning_rate'],
# 		Base=DecisionTreeRegressor(max_depth=int(params['max_depth']))
# 	)

# 	clf.fit(trainX_e, trainY_e)    
# 	preds = clf.pred_dist(valX_e)
# 	mu = preds.loc
# 	rmse = np.sqrt(np.mean((valY_e-mu)**2))

# 	return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()
start = time.time()

end = time.time()
print("Best hyperparameters:", best_ecc)
print("Optimization Time: %.2f seconds" % (end - start))

final_model_e = NGBRegressor(
	Dist=Beta,
	Score=LogScore,
	n_estimators=500,
	natural_gradient=True,
	minibatch_frac= best_ecc['minibatch_frac'], # 1
	learning_rate=best_ecc['learning_rate'], # 0.1
	Base=DecisionTreeRegressor(max_depth = int(max_depth = best_e['max_depth'])) # 6
)

# Save model for eccentricity
ngb = Path("models/best_model_e_final.ngb")
with ngb.open("wb") as f:
    pickle.dump(final_model_e, f)


# Train the model for proper inclination
# def objective(params):
# 	clf = NGBRegressor(
# 		Dist=Normal,
# 		Score=LogScore,
# 		verbose=False,
# 		minibatch_frac=params['minibatch_frac'],
# 		n_estimators=200,
# 		learning_rate=params['learning_rate'],
# 		Base=DecisionTreeRegressor(max_depth=int(params['max_depth']))
# 	)

# 	clf.fit(trainX_inc, trainY_inc)    
# 	preds = clf.pred_dist(valX_inc)
# 	mu = preds.loc
# 	rmse = np.sqrt(np.mean((valY_inc-mu)**2))

# 	return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()
start = time.time()

# best_inc = fmin(fn=objective, space = space, algo = tpe.suggest, max_evals = 10, trials = trials, rstate=np.random.default_rng(seed=0))
end = time.time()
print("Best hyperparameters:", best_inc)
print("Optimization Time: %.2f seconds" % (end - start))

final_model_inc = NGBRegressor(
	# Dist=Normal,
	# Score=LogScore,
	# n_estimators=500,
	# natural_gradient=True,
	minibatch_frac= best_inc['minibatch_frac'], # 1
	learning_rate=best_inc['learning_rate'], # 0.1
	Base=DecisionTreeRegressor(max_depth = int(max_depth = best_inc['max_depth'])) # 6
)

# Save model for inclination
ngb = Path("models/best_model_inc_final.ngb")
with ngb.open("wb") as f:
    pickle.dump(final_model_inc, f)

# Save all predicted values into a table for analysis
pred_dist = final_model_e.pred_dist(testX_e)

pred_e = pred_dist.loc
std_e = pred_dist.scale

pred_dist = final_model_inc.pred_dist(testX_inc)

pred_inc = pred_dist.loc
std_inc = pred_dist.scale

test_indices = testX_e.index.tolist()

df_ngb = pd.DataFrame(list(zip(testY_e, pred_e, std_e, testY_inc, pred_inc, std_inc)), columns = ["actual_dele", "pred_e", "error_e", "actual_delsini", "pred_inc", "error_inc"])
df_ngb = df_ngb.reset_index(drop=True)
test_data = merged_df.loc[test_indices].reset_index(drop=True)

df_ngb["Des'n"] = test_data["Des'n"]
df_ngb["e"] = test_data["e"]
df_ngb["Incl."] = test_data["Incl."]
df_ngb["propa"] = test_data["propa"]
df_ngb["prope"] = test_data["prope"]
df_ngb["prope_h"] = test_data["prope_h"]
df_ngb["propsini"] = test_data["propsini"]
df_ngb["propsini_h"] = test_data["propsini_h"]
df_ngb["da"] = test_data["da"]
df_ngb["dsini"] = test_data["dsini"]
df_ngb["de"] = test_data["de"]
df_ngb.to_csv("tables_for_analysis/NGBooster_result.csv")