# %%
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import time
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
# %%
# Read merged dataframe for model training
merged_df = pd.read_csv("data/merged_elements.csv")
# %%
features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 'g0', 'prope_h']
features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 's0', 'propsini_h']
data_e = merged_df[features_e]
data_inc = merged_df[features_inc]
dele = merged_df['prope']-merged_df['e']
delsini = merged_df['propsini']-np.sin(merged_df['Incl.']*np.pi/180)
delg = merged_df['g0'] - merged_df['g']
s = merged_df['s']

trainX_e, testX_e, trainX_inc, testX_inc, trainY_e, testY_e, trainY_inc, testY_inc = train_test_split(data_e, data_inc, dele, delsini, test_size=0.4, random_state=42)

space = {
	'max_depth': hp.qloguniform('max_depth', np.log(10), np.log(40), 1),
    'min_child_weight': hp.loguniform('min_child_weight', 0, np.log(30)),
    'subsample': hp.uniform ('subsample', 0.8, 1)
}
# %%
dtrain_e = xgb.DMatrix(trainX_e, trainY_e)
dtest_e = xgb.DMatrix(testX_e, testY_e)
dtrain_inc = xgb.DMatrix(trainX_inc, trainY_inc)
dtest_inc = xgb.DMatrix(testX_inc, testY_inc)
# %%
# Train the model for the proper eccentricity
def objective(params): # pyright: ignore[reportRedeclaration]
    clf = XGBRegressor(n_estimators = 50,
                            max_depth = int(params['max_depth']), 
                            min_child_weight = params['min_child_weight'],
                            subsample = params['subsample'],
                            learning_rate = 0.15, seed = 0,)

    score = xgb.cv(clf.get_xgb_params(), dtrain_e, nfold = 5, metrics = "rmse", early_stopping_rounds=10)
    avg_score =  np.mean(score["test-rmse-mean"])
    error = np.mean(score["test-rmse-std"])
    
    print("SCORE:", avg_score, "+/-", error)
    return {'loss': 1-avg_score, 'status': STATUS_OK, "cv_score": avg_score , "cv_error": error}

trials = Trials()
start = time.time()

best_ecc = fmin(fn=objective, space = space, algo = tpe.suggest, max_evals = 100, trials = trials, rstate=np.random.default_rng(seed=0))

end = time.time()
print("Best hyperparameters:", best_ecc)
print("Optimization Time: %.2f seconds" % (end - start))
# %%
final_model_e = XGBRegressor(learning_rate = 0.05, 
                         max_depth = 22, #int(best_ecc['max_depth']), 
                         subsample = best_ecc['subsample'],
                         min_child_weight = best_ecc['min_child_weight'])

final_model_e = xgb.train(dtrain=dtrain_e, params=final_model_e.get_params(), num_boost_round=2000)

# Save model for eccentricity
pth_e = Path("data/models/best_model_e_final.xgb")
final_model_e.save_model(str(pth_e))
# %%
# Train the model for the proper inclination
def objective(params): # pyright: ignore[reportRedeclaration]
    clf = XGBRegressor(n_estimators = 50,
                            max_depth = int(params['max_depth']), 
                            min_child_weight = params['min_child_weight'],
                            subsample = params['subsample'],
                            learning_rate = 0.15, seed = 0,)
    
    score = xgb.cv(clf.get_xgb_params(), dtrain_inc, nfold = 5, metrics = "rmse", early_stopping_rounds=10)
    avg_score =  np.mean(score["test-rmse-mean"])
    error = np.mean(score["test-rmse-std"])
    
    print("SCORE:", avg_score, "+/-", error)
    return {'loss': 1-avg_score, 'status': STATUS_OK, "cv_score": avg_score , "cv_error": error}

trials = Trials()
start = time.time()

best_inc = fmin(fn=objective, space = space, algo = tpe.suggest, max_evals = 100, trials = trials, rstate=np.random.default_rng(seed=0))

end = time.time()
print("Best hyperparameters:", best_inc)
print("Optimization Time: %.2f seconds" % (end - start))
# %%
final_model_inc = XGBRegressor(learning_rate = 0.05, 
                         max_depth = 22, #int(best_inc['max_depth']), 
                         subsample = best_inc['subsample'],
                         min_child_weight = best_inc['min_child_weight'])

final_model_inc = xgb.train(dtrain=dtrain_inc, params=final_model_inc.get_params(), num_boost_round=2000)

# Save model for inclination
pth_inc = Path("data/models/best_model_inc_final.xgb")
final_model_inc.save_model(str(pth_inc))