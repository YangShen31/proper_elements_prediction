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
# valX_e, testX_e, valX_inc, testX_inc, valY_e, testY_e, valY_inc, testY_inc = train_test_split(testX_e, testX_inc, testY_e, testY_inc, test_size=0.5, random_state=42)

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
grid_search1 = GridSearchCV(estimator=XGBRegressor(random_state=42, learning_rate=0.3, n_estimators=500, n_jobs=40),
                           param_grid=param1_grid, cv=3, scoring="neg_mean_squared_error", verbose=10)

grid_search1.fit(trainX_e, trainY_e)
print(grid_search1.best_params_)

grid_search2 = GridSearchCV(estimator=XGBRegressor(random_state=42, **grid_search1.best_params_, n_jobs=40),
                           param_grid=param2_grid, cv=3, scoring="neg_mean_squared_error", verbose=10)

grid_search2.fit(trainX_e, trainY_e)

end = time.time()
print(f"Best score: {grid_search2.best_score_:.3}")
print(f"Best parameters: {grid_search1.best_params_ | grid_search2.best_params_}")
print("Optimization Time: %.2f seconds" % (end - start))
# %%
# {'colsample_bytree': np.float64(1.0), 'max_depth': np.int64(9), 'min_child_weight': np.int64(1), 'subsample': np.float64(1.0), 'learning_rate': 0.05, 'n_estimators': 2500}
final_model_e = XGBRegressor(**{**grid_search1.best_params_, **grid_search2.best_params_}, n_jobs=40)

final_model_e.fit(trainX_e, trainY_e)

# Save model for eccentricity
pth_e = Path("data/models/best_model_e_final.xgb")
final_model_e.save_model(str(pth_e))
# %%
start = time.time()
grid_search1 = GridSearchCV(estimator=XGBRegressor(random_state=42, learning_rate=0.3, n_estimators=500, n_jobs=40),
                           param_grid=param1_grid, cv=3, scoring="neg_mean_squared_error", verbose=10)

grid_search1.fit(trainX_inc, trainY_inc)

grid_search2 = GridSearchCV(estimator=XGBRegressor(random_state=42, **grid_search1.best_params_, n_jobs=40),
                           param_grid=param2_grid, cv=3, scoring="neg_mean_squared_error", verbose=10)

grid_search2.fit(trainX_inc, trainY_inc)

end = time.time()
print(f"Best score: {grid_search2.best_score_:.3}")
print(f"Best parameters: {grid_search1.best_params_ | grid_search2.best_params_}")
print("Optimization Time: %.2f seconds" % (end - start))
# %%
# {'colsample_bytree': np.float64(0.9), 'max_depth': np.int64(9), 'min_child_weight': np.int64(1), 'subsample': np.float64(1.0), 'learning_rate': 0.05, 'n_estimators': 2000
final_model_inc = XGBRegressor(**{**grid_search1.best_params_, **grid_search2.best_params_}, n_jobs=40)

final_model_inc.fit(trainX_inc, trainY_inc)

# Save model for inclination
pth_inc = Path("data/models/best_model_inc_final.xgb")
final_model_inc.save_model(str(pth_inc))
