# %%
import time
import numpy as np
import pandas as pd

import xgboost as xgb
# %%
merged_df = pd.read_csv("data/merged_elements.csv", index_col=0, dtype={"Des'n": str})

features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'prope_linear']
features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'propsini_linear']

# merged_df_test = merged_df[merged_df["test_set"] == 1]
# X corresponds to inputs and Y to outputs
X_e = merged_df[features_e]
Y_e = merged_df['prope']-merged_df['e'] # dele
X_inc = merged_df[features_inc]
Y_inc = merged_df['propsini']-np.sin(np.deg2rad(merged_df["Incl."])) # delsini
# %%
final_model_e = xgb.XGBRegressor()
final_model_e.load_model("data/models/best_model_e_final.xgb")
final_model_inc = xgb.XGBRegressor()
final_model_inc.load_model("data/models/best_model_inc_final.xgb")
# %%
print("eccentricity feature information gain")
print(final_model_e.get_booster().get_score(importance_type='gain'))
print("inclination feature information gain")
print(final_model_inc.get_booster().get_score(importance_type='gain'))
# %%
# Save all predicted values into a table for analysis
start_t = time.process_time()
pred_e = final_model_e.predict(X_e)

pred_inc = final_model_inc.predict(X_inc)
eval_t = time.process_time() - start_t
print(f"Model Evaluation Time: {eval_t:.2f} sec for {len(X_e)} asteroids. {eval_t/len(X_e):.4} sec / asteroid")
# Model Evaluation Time: 43.54 sec for 249811 asteroids. 0.0001743 sec / asteroid

df_xgb = pd.DataFrame(list(zip(Y_e, pred_e, Y_inc, pred_inc)), columns = ["actual_dele", "pred_dele", "actual_delsini", "pred_delsini"])
df_xgb = df_xgb.reset_index(drop=True)

df_xgb["Des'n"] = merged_df["Des'n"]

# oscillating
df_xgb["e"] = merged_df["e"]
df_xgb["a"] = merged_df["a"]
df_xgb["Incl."] = merged_df["Incl."]
df_xgb["Node"] = merged_df["Node"]
df_xgb["Peri."] = merged_df["Peri."]

# linear
df_xgb["prope_linear"] = merged_df["prope_linear"]
df_xgb["propsini_linear"] = merged_df["propsini_linear"]

# predicted
pred_e = df_xgb["pred_dele"] + df_xgb["e"]
pred_e[pred_e < 0] = 0 # clamp values less than 0
df_xgb["pred_e"] = pred_e

pred_sini = df_xgb["pred_delsini"] + np.sin((np.deg2rad(df_xgb["Incl."])))
df_xgb["pred_sini"] = pred_sini

# acutal
df_xgb["propa"] = merged_df["propa"]
df_xgb["prope"] = merged_df["prope"]
df_xgb["propsini"] = merged_df["propsini"]

df_xgb[merged_df["test_set"] == 1].to_csv("data/model_results.csv")
df_xgb[merged_df["test_set"] == 0].to_csv("data/model_results_train.csv")
# %%
