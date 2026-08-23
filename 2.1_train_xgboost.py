# %%
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
import time
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
# %%
# Read Nesvorny catalog dataset
nesvorny_df = pd.read_csv("data/nesvorny_catalog_dataset.csv", index_col=0, dtype={"Des'n": str})

# Read linear prediction results
linear_theory_df = pd.read_csv("data/linear_theory.csv", index_col=0, dtype={"Des'n": str})
linear_theory_df['u0'] = linear_theory_df['u0'].apply(lambda x: complex(x))
linear_theory_df['v0'] = linear_theory_df['v0'].apply(lambda x: complex(x))
linear_theory_df['prope_h'] = np.abs(linear_theory_df['u0'])
linear_theory_df['propsini_h'] = np.sin(np.abs(linear_theory_df['v0']))
# %%
# Get merged dataframe for later model training
merged_df = pd.merge(nesvorny_df, linear_theory_df, on="Des'n", how="inner")

merged_df["prope_h"] = np.abs(merged_df["prope_h"])
merged_df["propsini_h"] = np.abs(merged_df["propsini_h"])
merged_df['prope_h'] = pd.to_numeric(merged_df['prope_h'], errors='coerce')
merged_df['propsini_h'] = pd.to_numeric(merged_df['propsini_h'], errors='coerce')
merged_df['Node'] = pd.to_numeric(merged_df['Node'], errors='coerce')
merged_df['Peri.'] = pd.to_numeric(merged_df['Peri.'], errors='coerce')

node = np.deg2rad(merged_df["Node"])
peri = np.deg2rad(merged_df["Peri."])
inc = np.deg2rad(merged_df["Incl."])

merged_df["ecospo"] = merged_df["e"] * np.cos(node + peri)
merged_df["esinpo"] = merged_df["e"] * np.sin(node + peri)
merged_df["sinicosO"] = np.sin(inc) * np.cos(node)
merged_df["sinisinO"] = np.sin(inc) * np.sin(node)

merged_df.to_csv("data/merged_elements.csv")
# %%
# Read merged dataframe for model training
merged_df = pd.read_csv("data/merged_elements.csv", index_col=0, dtype={"Des'n": str})
# %%
features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'prope_h']
features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'propsini_h']
data_e = merged_df[features_e]
data_inc = merged_df[features_inc]
dele = merged_df['prope']-merged_df['e']
delsini = merged_df['propsini']-np.sin(merged_df['Incl.']*np.pi/180)

trainX_e, testX_e, trainX_inc, testX_inc, trainY_e, testY_e, trainY_inc, testY_inc = train_test_split(data_e, data_inc, dele, delsini, train_size=0.8, random_state=42)

param1_grid = {
    'max_depth': np.arange(3, 21.01, 3, dtype=int),
    'min_child_weight': np.arange(1, 4.01, 1, dtype=int),
    'subsample': np.arange(0.8, 1.01, 0.1),
    'colsample_bytree': np.arange(0.8, 1.01, 0.1)
}
param2_grid = {
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [900, 1200, 1500, 2000, 2500]
}
# %%
start = time.time()
grid_search1_e = GridSearchCV(estimator=XGBRegressor(random_state=42, learning_rate=0.3, n_estimators=500, n_jobs=40),
                           param_grid=param1_grid, cv=5, scoring="neg_mean_squared_error", verbose=1)

grid_search1_e.fit(trainX_e, trainY_e)
print(grid_search1_e.best_params_)

grid_search2_e = GridSearchCV(estimator=XGBRegressor(random_state=42, **grid_search1_e.best_params_, n_jobs=40),
                           param_grid=param2_grid, cv=5, scoring="neg_mean_squared_error", verbose=1)

grid_search2_e.fit(trainX_e, trainY_e)

end = time.time()
print(f"Best score: {grid_search2_e.best_score_:.3}")
print(f"Best parameters: {grid_search1_e.best_params_ | grid_search2_e.best_params_}")
print("Optimization Time: %.2f seconds" % (end - start))
# %%
# final_model_e = grid_search2_e.best_estimator_

# e_best_params = {'colsample_bytree': np.float64(1.0), 'max_depth': np.int64(9), 'min_child_weight': np.int64(1), 'subsample': np.float64(1.0), 'learning_rate': 0.05, 'n_estimators': 2500}
final_model_e = XGBRegressor(**{**grid_search1_e.best_params_, **grid_search2_e.best_params_}, n_jobs=40)
# final_model_e = XGBRegressor(**e_best_params, n_jobs=40)

final_model_e.fit(trainX_e, trainY_e)
# print(f"{np.sqrt(np.mean((final_model_e.predict(testX_e) - testY_e)**2)):.3}") # print RMS (also calculated in 3.1_calc_propa.py)

# Save model for eccentricity
pth_e = Path("data/models/best_model_e_final.xgb")
final_model_e.save_model(str(pth_e))
# %%
start = time.time()
grid_search1_inc = GridSearchCV(estimator=XGBRegressor(random_state=42, learning_rate=0.3, n_estimators=500, n_jobs=40),
                           param_grid=param1_grid, cv=5, scoring="neg_mean_squared_error", verbose=1)

grid_search1_inc.fit(trainX_inc, trainY_inc)

grid_search2_inc = GridSearchCV(estimator=XGBRegressor(random_state=42, **grid_search1_inc.best_params_, n_jobs=40),
                           param_grid=param2_grid, cv=5, scoring="neg_mean_squared_error", verbose=1)

grid_search2_inc.fit(trainX_inc, trainY_inc)

end = time.time()
print(f"Best score: {grid_search2_inc.best_score_:.3}")
print(f"Best parameters: {grid_search1_inc.best_params_ | grid_search2_inc.best_params_}")
print("Optimization Time: %.2f seconds" % (end - start))
# %%
# final_model_inc = grid_search2_inc.best_estimator_

# inc_best_params = {'colsample_bytree': np.float64(0.9), 'max_depth': np.int64(9), 'min_child_weight': np.int64(1), 'subsample': np.float64(1.0), 'learning_rate': 0.05, 'n_estimators': 2000}
final_model_inc = XGBRegressor(**{**grid_search1_inc.best_params_, **grid_search2_inc.best_params_}, n_jobs=40)
# final_model_inc = XGBRegressor(**inc_best_params, n_jobs=40)

final_model_inc.fit(trainX_inc, trainY_inc)
# print(f"{np.sqrt(np.mean((final_model_inc.predict(testX_inc) - testY_inc)**2)):.3}") # print RMS (also calculated in 3.1_calc_propa.py)

# Save model for inclination
pth_inc = Path("data/models/best_model_inc_final.xgb")
final_model_inc.save_model(str(pth_inc))
# %%
e_best_idx = grid_search2_e.best_index_
e_score_mean = grid_search2_e.cv_results_["mean_test_score"][e_best_idx]
e_score_std = grid_search2_e.cv_results_["std_test_score"][e_best_idx]
print(f"Ecc score: {-e_score_mean:.5} ± {e_score_std:.5}")

inc_best_idx = grid_search2_inc.best_index_
inc_score_mean = grid_search2_inc.cv_results_["mean_test_score"][inc_best_idx]
inc_score_std = grid_search2_inc.cv_results_["std_test_score"][inc_best_idx]
print(f"Inc score: {-inc_score_mean:.5} ± {inc_score_std:.5}")
# %%