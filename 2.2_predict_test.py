# %%
import pickle
import numpy as np
import pandas as pd

from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
# %%
merged_df = pd.read_csv("data/merged_elements.csv")

features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 'g0', 'prope_h']
features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 's0', 'propsini_h']
data_e = merged_df[features_e]
data_inc = merged_df[features_inc]
dele = merged_df['prope']-merged_df['e']
delsini = merged_df['propsini']-np.sin(merged_df['Incl.']*np.pi/180)
delg = merged_df['g0'] - merged_df['g']
s = merged_df['s']

trainX_e, testX_e, trainX_inc, testX_inc, trainY_e, testY_e, trainY_inc, testY_inc = train_test_split(data_e, data_inc, dele, delsini, test_size=0.4, random_state=42)
valX_e, testX_e, valX_inc, testX_inc, valY_e, testY_e, valY_inc, testY_inc = train_test_split(testX_e, testX_inc, testY_e, testY_inc, test_size=0.5, random_state=42)
# %%
with open("data/models/best_model_e_final.ngb", "rb") as f:
    final_model_e = pickle.load(f)
with open("data/models/best_model_inc_final.ngb", "rb") as f:
    final_model_inc = pickle.load(f)
# %%
# Save all predicted values into a table for analysis
pred_dist = final_model_e.pred_dist(testX_e)

pred_e = pred_dist.loc
std_e = pred_dist.scale

pred_dist = final_model_inc.pred_dist(testX_inc)

pred_inc = pred_dist.loc
std_inc = pred_dist.scale

test_indices = testX_e.index.tolist()

df_ngb = pd.DataFrame(list(zip(testY_e, pred_e, std_e, testY_inc, pred_inc, std_inc)), columns = ["actual_dele", "pred_dele", "error_e", "actual_delsini", "pred_delsini", "error_sini"])
df_ngb = df_ngb.reset_index(drop=True)
test_data = merged_df.loc[test_indices].reset_index(drop=True)

df_ngb["Des'n"] = test_data["Des'n"]

# oscillating
df_ngb["e"] = test_data["e"]
df_ngb["Incl."] = test_data["Incl."]

# linear
df_ngb["prope_h"] = test_data["prope_h"]
df_ngb["propsini_h"] = test_data["propsini_h"]

# predicted
pred_e = df_ngb["pred_dele"] + df_ngb["e"]
pred_e[pred_e < 0] = 0 # clamp values less than 0
df_ngb["pred_e"] = pred_e

pred_sini = df_ngb["pred_delsini"] + np.sin((df_ngb["Incl."] * np.pi/180))
df_ngb["pred_sini"] = pred_sini

# acutal
df_ngb["propa"] = test_data["propa"]
df_ngb["prope"] = test_data["prope"]
df_ngb["propsini"] = test_data["propsini"]

df_ngb.to_csv("data/model_results.csv")