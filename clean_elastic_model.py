import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


X1 = pd.read_csv("/users/ramkd9/Lipid_Predict/feature_blankreduiction.csv", index_col=0)
X3 = pd.read_csv("newrna_cell_clair_filtered_symbol.csv", index_col=0).T
Y1 = pd.read_csv("/users/ramkd9/Bulk_lipids_cleaned_normalized_median_527.csv", index_col=0)
Y3 = pd.read_csv("/users/ramkd9/cell_lipids_cleaned_norm_median_286.csv", index_col=0)

# Collapse duplicated gene symbols after transpose
X3 = X3.groupby(X3.columns, axis=1).mean()

for df in (X1, X3, Y1, Y3):
    df.index = df.index.astype(str).str.strip()


# align RNA and lipid matrices
X_df = pd.concat([X1, X3], axis=0, join="inner")
Y_df = pd.concat([Y1, Y3], axis=0, join="inner")

X_df = X_df[~X_df.index.duplicated(keep="first")]
Y_df = Y_df[~Y_df.index.duplicated(keep="first")]

common_idx = X_df.index.intersection(Y_df.index)
X_df = X_df.loc[common_idx].copy()
Y_df = Y_df.loc[common_idx].copy()

X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)
X_df = X_df.loc[:, (X_df != 0).any(axis=0)]

# Keep only samples with complete lipid measurements
Y_df_clean = Y_df.dropna().copy()
X_df_clean = X_df.loc[Y_df_clean.index].copy()


#model fitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df_clean)

model = MultiOutputRegressor(
    ElasticNet(alpha=0.005, l1_ratio=0.7, max_iter=10000)
)
model.fit(X_scaled, Y_df_clean)

Y_pred = model.predict(X_scaled)
pred_df = pd.DataFrame(Y_pred, index=Y_df_clean.index, columns=Y_df_clean.columns)

#evaluation
pred_long = (
    pred_df.reset_index(names="sample")
    .melt(id_vars="sample", var_name="Lipid", value_name="Predicted")
)

true_long = (
    Y_df_clean.reset_index(names="sample")
    .melt(id_vars="sample", var_name="Lipid", value_name="True")
)

eval_df = pred_long.merge(true_long, on=["sample", "Lipid"], how="inner")
eval_df["error"] = eval_df["True"] - eval_df["Predicted"]


#sample annotations
eval_df["Organ"] = np.where(eval_df["sample"].str.startswith("D"), "lung", "serum")

suffix_to_celltype = {
    "_EPI": "EPI",
    "_END": "END",
    "_MIC": "MIC",
    "_PMX": "PMX",
    "_MES": "MES",
}

eval_df["CellType"] = "non_cell_type"
for suffix, label in suffix_to_celltype.items():
    eval_df.loc[eval_df["sample"].str.endswith(suffix), "CellType"] = label


#per-lipid performance
r2_per_lipid = (
    eval_df.groupby("Lipid")
    .apply(lambda df: r2_score(df["True"], df["Predicted"]))
    .rename("R2")
)


#lipid group cell type presence 
lipids_by_group = {
    group: set(sub.loc[sub["True"] > 0, "Lipid"])
    for group, sub in eval_df.groupby("CellType")
}

lipids_lung = set(
    eval_df.loc[(eval_df["Organ"] == "lung") & (eval_df["True"] > 0), "Lipid"]
)

groups_for_intersection = ["EPI", "MES", "END", "MIC", "PMX"]
lipid_sets = [lipids_by_group[g] for g in groups_for_intersection if g in lipids_by_group]
lipid_sets.append(lipids_lung)

lipids_all_groups = set.intersection(*lipid_sets) if lipid_sets else set()

print(f"Number of shared lipids in lung: {len(lipids_lung)}")
print(f"Number of lipids present in all selected groups: {len(lipids_all_groups)}")

pred_df_both_organs = eval_df[eval_df["Lipid"].isin(lipids_lung)].copy()
pred_df_all_organs = eval_df[eval_df["Lipid"].isin(lipids_all_groups)].copy()

#metrics
true_vals = eval_df["True"].to_numpy()
pred_vals = eval_df["Predicted"].to_numpy()

safe_true = np.where(true_vals == 0, np.nan, true_vals)

rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
norm_rmse = rmse / np.mean(true_vals)
pearson_corr = np.corrcoef(true_vals, pred_vals)[0, 1]
mape = np.nanmean(np.abs((true_vals - pred_vals) / safe_true))
nse = 1 - (
    np.sum((true_vals - pred_vals) ** 2) /
    np.sum((true_vals - np.mean(true_vals)) ** 2)
)

r = pearson_corr
alpha = np.std(pred_vals) / np.std(true_vals)
beta = np.mean(pred_vals) / np.mean(true_vals)
kge = 1 - np.sqrt((1 - r) ** 2 + (1 - alpha) ** 2 + (1 - beta) ** 2)

metrics = {
    "RMSE": rmse,
    "Normalized_RMSE": norm_rmse,
    "Pearson_r": pearson_corr,
    "MAPE": mape,
    "NSE": nse,
    "KGE": kge,
}

print(metrics)


import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", ConvergenceWarning)
    model.fit(X_scaled, Y_df_clean)

n_conv = sum(issubclass(wi.category, ConvergenceWarning) for wi in w)
print("Convergence warnings:", n_conv)