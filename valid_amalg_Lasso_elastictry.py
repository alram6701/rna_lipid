
import pickle
import numpy as np
import pandas as pd

# Load old ElasticNet bundle
with open("ElasticNet_model.pkl", "rb") as f:
    bundle_enet = pickle.load(f)

enet_model = bundle_enet["enet_fs"]
enet_scaler_x = bundle_enet["scaler"]
enet_X_cols = bundle_enet["X_columns"]
enet_Y_cols = bundle_enet["Y_columns"]

# Load multitask ElasticNet bundle
with open("elastic_multitask_try.pkl", "rb") as f:
    bundle_mt = pickle.load(f)

mt_model = bundle_mt["model"]
mt_scaler_x = bundle_mt["scaler_x"]
mt_scaler_y = bundle_mt.get("scaler_y", None)
mt_X_cols = bundle_mt["X_columns"]
mt_Y_cols = bundle_mt["Y_columns"]

print("Loaded both models.")
print("ElasticNet outputs:", len(enet_Y_cols))
print("MultiTask outputs:", len(mt_Y_cols))

X1 = pd.read_csv("/users/ramkd9/Lipid_Predict/feature_blankreduiction.csv", index_col=0)

X3 = pd.read_csv("newrna_cell_clair_filtered_symbol.csv",index_col=0).T
X3 = X3.groupby(X3.columns, axis=1).mean()

Y1 = pd.read_csv("/users/ramkd9/Bulk_lipids_cleaned_normalized_median_527.csv", index_col=0)
Y3 = pd.read_csv("/users/ramkd9/cell_lipids_cleaned_norm_median_286.csv", index_col=0)


counts = pd.read_csv(
    "/users/ramkd9/Lipid_Predict/exp.GSE161382_counts_matrix_CPTT-sample-revised-gene-pediatric.txt",
    sep="\t"
)

groups = pd.read_csv(
    "/users/ramkd9/Lipid_Predict/groups.GSE161382_counts_matrix_CPTT-sample-revised-prediatric.txt",
    sep="\t"
)

# collapse duplicated genes in X3
X3 = X3.groupby(X3.columns, axis=1).mean()

for df in [X1, X3, Y1, Y3]:
    df.index = df.index.astype(str).str.strip()

X_df = pd.concat([X1, X3], axis=0, join="inner")
Y_df = pd.concat([Y1, Y3], axis=0, join="inner")

# keep matched samples only
common_idx = X_df.index.intersection(Y_df.index)
X_df = X_df.loc[common_idx].copy()
Y_df = Y_df.loc[common_idx].copy()

# numeric cleanup
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.loc[:, X_df.notna().sum() > 0].fillna(0)

# remove duplicate samples
X_df = X_df[~X_df.index.duplicated(keep="first")]
Y_df = Y_df[~Y_df.index.duplicated(keep="first")]

print("X_df shape:", X_df.shape)
print("Y_df shape:", Y_df.shape)

#test inside train
# test_samples = {
#     'D041_EPI', 'D022_MIC', 'D024_EPI', 'D043_MES', 'D024_END',
#     'D043_EPI', 'D071_EPI', 'D044_MIC', 'D044_MES', 'D019_MIC',
#     'D071_END', 'D038_MIC', 'D043_END', 'D071_PMX', 'D044_PMX',
#     'D022_MES', 'D019_END', 'D024_PMX', 'D038_PMX', 'D018_END',
#     'D022_PMX'
# }

# sample_series = Y_df.index
# train_mask = ~sample_series.isin(test_samples)
# test_mask = sample_series.isin(test_samples)

#Lets exclude 4 patient identifiers from all training sets if the index starts with  "D018","D022","D024","D036" for example "D022_END" and "D022_EPI", "D024_END","D036_MES" 


# X_df = X_df[~X_df.index.duplicated(keep="first")]  # remove duplicate samples

for df in [X1, X3, Y1, Y3]:
    df.index = df.index.astype(str).str.strip()

Y_df = pd.concat([Y1, Y3], axis=0, join="inner")
X_df = pd.concat([X1, X3], axis=0, join = "inner")
print(Y_df.shape)  
print(X_df.shape)

common_idx = X_df.index.intersection(Y_df.index)
print("Matched samples:", len(common_idx))

X_df = X_df.loc[common_idx].copy()
Y_df = Y_df.loc[common_idx].copy()

X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.loc[:, X_df.notna().sum() > 0]

X_df = X_df.fillna(0)

common_samples = Y_df.index.intersection(X_df.index)
X_df = X_df.loc[common_samples]
Y_df = Y_df.loc[common_samples]

X_df = X_df[~X_df.index.duplicated(keep="first")]
Y_df = Y_df[~Y_df.index.duplicated(keep="first")]

# # HELD-OUT INTERNAL VALIDATION
# test_samples = {
#     'D041_EPI', 'D022_MIC', 'D024_EPI', 'D043_MES', 'D024_END',
#     'D043_EPI', 'D071_EPI', 'D044_MIC', 'D044_MES', 'D019_MIC',
#     'D071_END', 'D038_MIC', 'D043_END', 'D071_PMX', 'D044_PMX',
#     'D022_MES', 'D019_END', 'D024_PMX', 'D038_PMX', 'D018_END',
#     'D022_PMX'
# }

# sample_series = Y_df.index
# train_mask = ~sample_series.isin(test_samples)
# test_mask  = sample_series.isin(test_samples)

# X_train = X_df.loc[train_mask].copy()
# Y_train = Y_df.loc[train_mask].copy()

# X_holdout = X_df.loc[test_mask].copy()
# Y_holdout = Y_df.loc[test_mask].copy()

X_train = X_df
Y_train = Y_df
X_holdout = X_df
Y_holdout = Y_df


common_samples = X_holdout.index.intersection(Y_holdout.index)
common_lipids = Y_holdout.columns.intersection(Y_df.columns)

X_holdout_aligned = X_holdout.loc[common_samples].copy()
Y_holdout_aligned = Y_holdout.loc[common_samples, common_lipids].copy()

nan_counts = Y_holdout_aligned.isna().sum()
Y_holdout_aligned = Y_holdout_aligned.loc[:, nan_counts <= 16].fillna(0)

# align genes to model input
X_holdout_aligned_enet = X_holdout.reindex(columns=enet_X_cols, fill_value=0.0)
X_holdout_scaled_enet = enet_scaler_x.transform(X_holdout_aligned_enet)
Y_holdout_pred_enet = enet_model.predict(X_holdout_scaled_enet)

pred_holdout_enet_df = pd.DataFrame(
    Y_holdout_pred_enet,
    index=X_holdout_aligned_enet.index,
    columns=enet_Y_cols
)

X_holdout_aligned_mt = X_holdout.reindex(columns=mt_X_cols, fill_value=0.0)
X_holdout_scaled_mt = mt_scaler_x.transform(X_holdout_aligned_mt)
Y_holdout_pred_mt_scaled = mt_model.predict(X_holdout_scaled_mt)
Y_holdout_pred_mt = mt_scaler_y.inverse_transform(Y_holdout_pred_mt_scaled)

pred_holdout_mt_df = pd.DataFrame(
    Y_holdout_pred_mt,
    index=X_holdout_aligned_mt.index,
    columns=mt_Y_cols
)
# # restrict predicted lipids to those in truth
# common_lipids = Y_holdout_aligned.columns.intersection(pred_holdout_df.columns)
# Y_holdout_aligned = Y_holdout_aligned.loc[:, common_lipids]
# pred_holdout_df = pred_holdout_df.loc[Y_holdout_aligned.index, common_lipids]

meta_holdout = pd.DataFrame(index=Y_holdout_aligned.index)
meta_holdout["Organ"] = meta_holdout.index.str.split("_").str[0]
meta_holdout["CellType"] = meta_holdout.index.str.split("_").str[-1]


counts = counts.set_index(counts.columns[0])
groups.columns = ["sample", "cell_type"]

groups = groups[groups["sample"].isin(counts.columns)].copy()
counts = counts[groups["sample"]]

groups["donor"] = groups["sample"].str.split("__").str[0]
groups["celltype"] = groups["cell_type"]

pseudobulk = []
for (donor, ct), rows in groups.groupby(["donor", "celltype"]):
    cols = rows["sample"].tolist()
    expr = counts[cols].mean(axis=1)
    name = f"{donor}_{ct.replace(' ', '_').replace('/', '_').replace('-', '_')}"
    pseudobulk.append(expr.rename(name))

pediatric_X_test = pd.DataFrame(pseudobulk)
pediatric_X_test = pediatric_X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Pediatric input aligned to each model
# ElasticNet model
pediatric_X_test_aligned_enet = pediatric_X_test.reindex(columns=enet_X_cols, fill_value=0.0)

print("ElasticNet pediatric X shape:", pediatric_X_test_aligned_enet.shape)
print("ElasticNet same feature count?", pediatric_X_test_aligned_enet.shape[1] == len(enet_X_cols))
print("ElasticNet same order?", (pediatric_X_test_aligned_enet.columns == pd.Index(enet_X_cols)).all())

pediatric_X_scaled_enet = enet_scaler_x.transform(pediatric_X_test_aligned_enet)
pediatric_X_scaled_enet = np.nan_to_num(
    pediatric_X_scaled_enet, nan=0.0, posinf=0.0, neginf=0.0
)

pediatric_Y_pred_enet = enet_model.predict(pediatric_X_scaled_enet)

pediatric_pred_enet_df = pd.DataFrame(
    pediatric_Y_pred_enet,
    index=pediatric_X_test_aligned_enet.index,
    columns=enet_Y_cols
)


# MultiTask model
pediatric_X_test_aligned_mt = pediatric_X_test.reindex(columns=mt_X_cols, fill_value=0.0)

print("MultiTask pediatric X shape:", pediatric_X_test_aligned_mt.shape)
print("MultiTask same feature count?", pediatric_X_test_aligned_mt.shape[1] == len(mt_X_cols))
print("MultiTask same order?", (pediatric_X_test_aligned_mt.columns == pd.Index(mt_X_cols)).all())

pediatric_X_scaled_mt = mt_scaler_x.transform(pediatric_X_test_aligned_mt)
pediatric_X_scaled_mt = np.nan_to_num(
    pediatric_X_scaled_mt, nan=0.0, posinf=0.0, neginf=0.0
)

pediatric_Y_pred_mt_scaled = mt_model.predict(pediatric_X_scaled_mt)

if mt_scaler_y is not None:
    pediatric_Y_pred_mt = mt_scaler_y.inverse_transform(pediatric_Y_pred_mt_scaled)
else:
    pediatric_Y_pred_mt = pediatric_Y_pred_mt_scaled

pediatric_pred_mt_df = pd.DataFrame(
    pediatric_Y_pred_mt,
    index=pediatric_X_test_aligned_mt.index,
    columns=mt_Y_cols
)
meta_pediatric = pd.DataFrame(index=pediatric_pred_enet_df.index)
meta_pediatric["Organ"] = meta_pediatric.index.str.split("_").str[0]
meta_pediatric["CellType_raw"] = meta_pediatric.index.str.split("_", n=1).str[1]

def map_to_coarse(ct):
    ct = str(ct).lower()

    if ("pmn" in ct) or ("polymorph" in ct):
        return "PMX"
    if ("macrophage" in ct) or ("mono" in ct) or ("dc" in ct) or ("dendritic" in ct):
        return "MIC"
    if ("neutro" in ct):
        return "PMX"
    if ("endothelial" in ct) or ("cap" in ct) or ("vein" in ct) or ("arter" in ct):
        return "END"
    if ("fibro" in ct) or ("myofibro" in ct) or ("pericyte" in ct) or ("smooth_muscle" in ct) or ("stromal" in ct):
        return "MES"
    if ("at1" in ct) or ("at2" in ct) or ("club" in ct) or ("ciliated" in ct) or ("epithelial" in ct):
        return "EPI"
    return "OTHER"

meta_pediatric["CellType"] = meta_pediatric["CellType_raw"].map(map_to_coarse)

print(meta_pediatric["CellType"].value_counts(dropna=False))
meta_pediatric5 = meta_pediatric[
    meta_pediatric["CellType"].isin(["END", "EPI", "MES", "MIC", "PMX"])
].copy()
def build_valid_df(df, metadata, group_col="CellType", top_n=202, pairs=None):
    if pairs is None:
        celltypes_present = sorted(metadata[group_col].dropna().unique())
        pairs = list(combinations(celltypes_present, 2))

    valid_list = []

    for g1, g2 in pairs:
        stats = compute_pairwise_lipids(df, metadata, group_col, g1, g2)

        if stats is None or stats.empty:
            print(f"Skipping {g1}_vs_{g2}: empty")
            continue

        top = stats.head(top_n).copy()
        top["Comparison"] = f"{g1}_vs_{g2}"
        valid_list.append(top)

    if len(valid_list) == 0:
        print("No valid pairwise comparisons found.")
        return pd.DataFrame(columns=[
            "Lipid","Group1","Group2","log2FC","t_stat","pval","FDR",
            "EffectSize","Direction","Comparison"
        ])

    return pd.concat(valid_list, ignore_index=True)
# Combine all comparison
celltype_pairs = [
    ('END', 'MES'),
    ('END', 'MIC'),
    ('END', 'PMX'),
    ('EPI', 'MES'),
    ('EPI', 'MIC'),
    ('EPI', 'PMX'),
    ('MES', 'MIC'),
    ('MES', 'PMX'),
    ('MIC', 'PMX')
]
pediatric_pred_enet5 = pediatric_pred_enet_df.loc[meta_pediatric5.index].copy()
pediatric_pred_mt5 = pediatric_pred_mt_df.loc[meta_pediatric5.index].copy()

print("pediatric_pred_enet5:", pediatric_pred_enet5.shape)
print("pediatric_pred_mt5:", pediatric_pred_mt5.shape)
print(meta_pediatric5["CellType"].value_counts())
pediatric_enet_valid_df = build_valid_df(
    df=pediatric_pred_enet5,
    metadata=meta_pediatric5,
    group_col="CellType",
    top_n=202,
    pairs=celltype_pairs
)

pediatric_mt_valid_df = build_valid_df(
    df=pediatric_pred_mt5,
    metadata=meta_pediatric5,
    group_col="CellType",
    top_n=202,
    pairs=celltype_pairs
)

print("pediatric_enet_valid_df:", pediatric_enet_valid_df.shape)
print("pediatric_mt_valid_df:", pediatric_mt_valid_df.shape)


def predict_enet(X_input):
    """
    Predict using the old MultiOutput ElasticNet bundle.
    Assumes Y was NOT scaled during training.
    """
    X_aligned = X_input.reindex(columns=enet_X_cols, fill_value=0.0)
    X_scaled = enet_scaler_x.transform(X_aligned)
    Y_pred = enet_model.predict(X_scaled)
    return pd.DataFrame(Y_pred, index=X_aligned.index, columns=enet_Y_cols)


def predict_multitask(X_input):
    """
    Predict using the multitask ElasticNet bundle.
    Assumes Y WAS scaled during training.
    """
    X_aligned = X_input.reindex(columns=mt_X_cols, fill_value=0.0)
    X_scaled = mt_scaler_x.transform(X_aligned)
    Y_pred_scaled = mt_model.predict(X_scaled)

    if mt_scaler_y is not None:
        Y_pred = mt_scaler_y.inverse_transform(Y_pred_scaled)
    else:
        Y_pred = Y_pred_scaled

    return pd.DataFrame(Y_pred, index=X_aligned.index, columns=mt_Y_cols)


pred_holdout_enet = predict_enet(X_holdout_aligned)
pred_holdout_mt = predict_multitask(X_holdout_aligned)
pediatric_pred_enet = predict_enet(pediatric_X_test)
pediatric_pred_mt = predict_multitask(pediatric_X_test)

common_lipids_holdout = (
    Y_holdout_aligned.columns
    .intersection(pred_holdout_enet.columns)
    .intersection(pred_holdout_mt.columns)
)

Y_true_cmp = Y_holdout_aligned.loc[:, common_lipids_holdout].copy()
pred_enet_cmp = pred_holdout_enet.loc[Y_true_cmp.index, common_lipids_holdout].copy()
pred_mt_cmp = pred_holdout_mt.loc[Y_true_cmp.index, common_lipids_holdout].copy()

print("Comparison shapes:")
print("Truth:", Y_true_cmp.shape)
print("ElasticNet:", pred_enet_cmp.shape)
print("MultiTask:", pred_mt_cmp.shape)

from sklearn.metrics import r2_score

def compute_global_metrics(y_true_df, y_pred_df, model_name="model"):
    true_vals = y_true_df.to_numpy().ravel()
    pred_vals = y_pred_df.to_numpy().ravel()

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

    return pd.Series({
        "Model": model_name,
        "RMSE": rmse,
        "Normalized_RMSE": norm_rmse,
        "Pearson_r": pearson_corr,
        "MAPE": mape,
        "NSE": nse,
        "KGE": kge,
        "Global_R2": r2_score(true_vals, pred_vals)
    })
metrics_enet = compute_global_metrics(Y_true_cmp, pred_enet_cmp, model_name="ElasticNet_MultiOutput")
metrics_mt = compute_global_metrics(Y_true_cmp, pred_mt_cmp, model_name="MultiTaskElasticNetCV")

metrics_compare = pd.DataFrame([metrics_enet, metrics_mt])
print(metrics_compare)

def compute_r2_per_lipid(y_true_df, y_pred_df, model_name):
    rows = []
    common_lipids = y_true_df.columns.intersection(y_pred_df.columns)

    for lipid in common_lipids:
        yt = y_true_df[lipid].values
        yp = y_pred_df[lipid].values
        rows.append({
            "Lipid": lipid,
            "Model": model_name,
            "R2": r2_score(yt, yp)
        })

    return pd.DataFrame(rows)

r2_enet = compute_r2_per_lipid(Y_true_cmp, pred_enet_cmp, "ElasticNet_MultiOutput")
r2_mt = compute_r2_per_lipid(Y_true_cmp, pred_mt_cmp, "MultiTaskElasticNetCV")

r2_compare = r2_enet.merge(
    r2_mt,
    on="Lipid",
    suffixes=("_enet", "_mt")
)

r2_compare["R2_diff_mt_minus_enet"] = r2_compare["R2_mt"] - r2_compare["R2_enet"]
print(r2_compare.sort_values("R2_diff_mt_minus_enet", ascending=False).head(20))


pediatric_pred_enet5 = pediatric_pred_enet.loc[meta_pediatric5.index].copy()
pediatric_pred_mt5 = pediatric_pred_mt.loc[meta_pediatric5.index].copy()

pediatric_enet_valid_df = build_valid_df(
    df=pediatric_pred_enet5,
    metadata=meta_pediatric5,
    group_col="CellType",
    top_n=202,
    pairs=celltype_pairs
)

pediatric_mt_valid_df = build_valid_df(
    df=pediatric_pred_mt5,
    metadata=meta_pediatric5,
    group_col="CellType",
    top_n=202,
    pairs=celltype_pairs
)

print("pediatric_enet_valid_df:", pediatric_enet_valid_df.shape)
print("pediatric_mt_valid_df:", pediatric_mt_valid_df.shape)

# Predict both models on same holdout input
pred_holdout_enet = predict_enet(X_holdout_aligned)
pred_holdout_mt = predict_multitask(X_holdout_aligned)

common_lipids_holdout = (
    Y_holdout_aligned.columns
    .intersection(pred_holdout_enet.columns)
    .intersection(pred_holdout_mt.columns)
)

Y_true_cmp = Y_holdout_aligned.loc[:, common_lipids_holdout]
pred_enet_cmp = pred_holdout_enet.loc[Y_true_cmp.index, common_lipids_holdout]
pred_mt_cmp = pred_holdout_mt.loc[Y_true_cmp.index, common_lipids_holdout]

metrics_compare = pd.DataFrame([
    compute_global_metrics(Y_true_cmp, pred_enet_cmp, "ElasticNet_MultiOutput"),
    compute_global_metrics(Y_true_cmp, pred_mt_cmp, "MultiTaskElasticNetCV")
])

print(metrics_compare)


from upsetplot import UpSet, from_contents
import matplotlib.pyplot as plt

alpha = 0.05

true_sig = enet_valid_df[enet_valid_df["pval"] < alpha].copy()
pred_sig = pediatric_enet_valid_df[pediatric_enet_valid_df["pval"] < alpha].copy()

true_sig["Lipid"] = true_sig["Lipid"].astype(str).str.strip()
pred_sig["Lipid"] = pred_sig["Lipid"].astype(str).str.strip()

comparisons = sorted(set(true_sig["Comparison"]) | set(pred_sig["Comparison"]))

print("Comparisons found:", comparisons)

for comp in comparisons:
    true_set = set(true_sig.loc[true_sig["Comparison"] == comp, "Lipid"])
    pred_set = set(pred_sig.loc[pred_sig["Comparison"] == comp, "Lipid"])

    print(f"\n{comp}")
    print("Holdout ENet:", len(true_set))
    print("Pediatric ENet:", len(pred_set))
    print("Overlap:", len(true_set & pred_set))

    if len(true_set) == 0 and len(pred_set) == 0:
        print("Skipping empty comparison")
        continue

    sets = {
        "Holdout_ENet": true_set,
        "Pediatric_ENet": pred_set
    }

    upset_data = from_contents(sets)

    plt.figure(figsize=(6, 4))
    UpSet(
        upset_data,
        subset_size="count",
        show_counts=True
    ).plot()

    plt.suptitle(f"{comp} differential lipids (p < {alpha})")
    plt.tight_layout()
    plt.savefig(f"enet_{comp}_diff_lipids.png", dpi=300, bbox_inches="tight")
    plt.show()




alpha = 0.05

true_sig = mt_valid_df[mt_valid_df["pval"] < alpha].copy()
pred_sig = pediatric_mt_valid_df[pediatric_mt_valid_df["pval"] < alpha].copy()

true_sig["Lipid"] = true_sig["Lipid"].astype(str).str.strip()
pred_sig["Lipid"] = pred_sig["Lipid"].astype(str).str.strip()

comparisons = sorted(set(true_sig["Comparison"]) | set(pred_sig["Comparison"]))

print("Comparisons found:", comparisons)

for comp in comparisons:
    true_set = set(true_sig.loc[true_sig["Comparison"] == comp, "Lipid"])
    pred_set = set(pred_sig.loc[pred_sig["Comparison"] == comp, "Lipid"])

    print(f"\n{comp}")
    print("Holdout ENet:", len(true_set))
    print("Pediatric ENet:", len(pred_set))
    print("Overlap:", len(true_set & pred_set))

    if len(true_set) == 0 and len(pred_set) == 0:
        print("Skipping empty comparison")
        continue

    sets = {
        "Holdout_ENet": true_set,
        "Pediatric_ENet": pred_set
    }

    upset_data = from_contents(sets)

    plt.figure(figsize=(6, 4))
    UpSet(
        upset_data,
        subset_size="count",
        show_counts=True
    ).plot()

    plt.suptitle(f"{comp} differential lipids (p < {alpha})")
    plt.tight_layout()
    plt.savefig(f"mt_{comp}_diff_lipids.png", dpi=300, bbox_inches="tight")
    plt.show()
