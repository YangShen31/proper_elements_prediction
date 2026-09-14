# %%
import time
import numpy as np
import pandas as pd

import xgboost as xgb
# %%
merged_df = pd.read_csv("data/merged_elements.csv", index_col=0, dtype={"Des'n": str})

features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'prope_linear']
features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'a', 'propsini_linear']

merged_df_test = merged_df[merged_df["test_set"] == 1]
testX_e = merged_df_test[features_e]
testY_e = merged_df_test['prope']-merged_df_test['e'] # dele
testX_inc = merged_df_test[features_inc]
testY_inc = merged_df_test['propsini']-np.sin(np.deg2rad(merged_df_test["Incl."])) # delsini
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
pred_e = final_model_e.predict(testX_e)

pred_inc = final_model_inc.predict(testX_inc)
eval_t = time.process_time() - start_t
print(f"Model Evaluation Time: {eval_t:.2f} sec for {len(testX_e)} asteroids. {eval_t/len(testX_e):.4} sec / asteroid")
# Model Evaluation Time: 47.05 sec for 249811 asteroids. 0.0001883 sec / asteroid

test_indices = testX_e.index.tolist()

df_xgb = pd.DataFrame(list(zip(testY_e, pred_e, testY_inc, pred_inc)), columns = ["actual_dele", "pred_dele", "actual_delsini", "pred_delsini"])
df_xgb = df_xgb.reset_index(drop=True)
test_data = merged_df.loc[test_indices].reset_index(drop=True)

df_xgb["Des'n"] = test_data["Des'n"]

# oscillating
df_xgb["e"] = test_data["e"]
df_xgb["a"] = test_data["a"]
df_xgb["Incl."] = test_data["Incl."]
df_xgb["Node"] = test_data["Node"]
df_xgb["Peri."] = test_data["Peri."]

# linear
df_xgb["prope_linear"] = test_data["prope_linear"]
df_xgb["propsini_linear"] = test_data["propsini_linear"]

# predicted
pred_e = df_xgb["pred_dele"] + df_xgb["e"]
pred_e[pred_e < 0] = 0 # clamp values less than 0
df_xgb["pred_e"] = pred_e

pred_sini = df_xgb["pred_delsini"] + np.sin((np.deg2rad(df_xgb["Incl."])))
df_xgb["pred_sini"] = pred_sini

# acutal
df_xgb["propa"] = test_data["propa"]
df_xgb["prope"] = test_data["prope"]
df_xgb["propsini"] = test_data["propsini"]

df_xgb.to_csv("data/model_results.csv")
# %%
