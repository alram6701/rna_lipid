import os
import math
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from upsetplot import UpSet, from_contents


MODEL_PATH = "elastic_multitask_try.pkl"
#MODEL_PATH = "ElasticNet_model.pkl"

OUTDIR = "results_lipid_validation"

INTERNAL_RNA_PATH = ("data/feature_blankreduiction.csv")
CELL_RNA_PATH = ("data/newrna_cell_clair_filtered_symbol.csv")
INTERNAL_LIPID_PATH = ("data/Bulk_lipids_cleaned_normalized_median_527.csv")
CELL_LIPID_PATH = ("data/cell_lipids_cleaned_norm_median_286.csv")


PED_COUNTS_PATH =("data/exp.GSE161382_counts_matrix_CPTT-sample-revised-gene-pediatric.txt")
PED_GROUPS_PATH = ("data/groups.GSE161382_counts_matrix_CPTT-sample-revised-prediatric.txt")


# Holdout sample IDs for internal truth set
# HOLDOUT_SAMPLES = {
#     "D041_EPI", "D022_MIC", "D024_EPI", "D043_MES", "D024_END",
#     "D043_EPI", "D071_EPI", "D044_MIC", "D044_MES", "D019_MIC",
#     "D071_END", "D038_MIC", "D043_END", "D071_PMX", "D044_PMX",
#     "D022_MES", "D019_END", "D024_PMX", "D038_PMX", "D018_END",
#     "D022_PMX"
# }

CELLTYPE_PAIRS = [
    ("END", "EPI"),
    ("END", "MES"),
    ("END", "MIC"),
    ("END", "PMX"),
    ("EPI", "MES"),
    ("EPI", "MIC"),
    ("EPI", "PMX"),
    ("MES", "MIC"),
    ("MES", "PMX"),
    ("MIC", "PMX"),
]

ALPHA = 0.05
TOP_N = 202
PSEUDOCOUNT = 1e-6


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def load_model_bundle(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle.get("model", bundle.get("enet_fs"))
    scaler_x = bundle.get("scaler_x", bundle.get("scaler"))
    scaler_y = bundle.get("scaler_y", None)
    x_cols = bundle["X_columns"]
    y_cols = bundle["Y_columns"]

    if model is None or scaler_x is None:
        raise ValueError(
            "Could not find model/scaler in bundle. "
            "Expected keys like model/scaler_x or enet_fs/scaler."
        )

    return model, scaler_x, scaler_y, list(x_cols), list(y_cols)


def clean_index(df):
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    return df


def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_internal_data():
    x1 = pd.read_csv(INTERNAL_RNA_PATH, index_col=0)
    x3 = pd.read_csv(CELL_RNA_PATH, index_col=0).T
    y1 = pd.read_csv(INTERNAL_LIPID_PATH, index_col=0)
    y3 = pd.read_csv(CELL_LIPID_PATH, index_col=0)

    x1 = clean_columns(clean_index(x1))
    x3 = clean_columns(clean_index(x3))
    y1 = clean_columns(clean_index(y1))
    y3 = clean_columns(clean_index(y3))

    # collapse duplicated genes in X3
    x3 = x3.T.groupby(level=0).mean().T

    x_df = pd.concat([x1, x3], axis=0, join="inner")
    y_df = pd.concat([y1, y3], axis=0, join="inner")

    x_df = clean_index(x_df)
    y_df = clean_index(y_df)

    common_idx = x_df.index.intersection(y_df.index)
    x_df = x_df.loc[common_idx].copy()
    y_df = y_df.loc[common_idx].copy()

    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.loc[:, x_df.notna().sum() > 0].fillna(0)

    y_df = y_df.apply(pd.to_numeric, errors="coerce")

    x_df = x_df[~x_df.index.duplicated(keep="first")]
    y_df = y_df[~y_df.index.duplicated(keep="first")]

    common_idx = x_df.index.intersection(y_df.index)
    x_df = x_df.loc[common_idx].copy()
    y_df = y_df.loc[common_idx].copy()

    return x_df, y_df


def split_internal_holdout(x_df, y_df, holdout_samples):
    holdout_mask = y_df.index.isin(holdout_samples)

    x_holdout = x_df.loc[holdout_mask].copy()
    y_holdout = y_df.loc[holdout_mask].copy()

    if x_holdout.empty or y_holdout.empty:
        print("Warning: no holdout samples found; using all internal samples as holdout.")
        x_holdout = x_df.copy()
        y_holdout = y_df.copy()

    common_idx = x_holdout.index.intersection(y_holdout.index)
    x_holdout = x_holdout.loc[common_idx].copy()
    y_holdout = y_holdout.loc[common_idx].copy()

    return x_holdout, y_holdout


def predict_lipids(model, scaler_x, x_df, x_cols, y_cols):
    x_aligned = x_df.reindex(columns=x_cols, fill_value=0.0)
    x_scaled = scaler_x.transform(x_aligned)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    y_pred = model.predict(x_scaled)
    pred_df = pd.DataFrame(y_pred, index=x_aligned.index, columns=y_cols)
    return pred_df


def build_internal_metadata(index):
    meta = pd.DataFrame(index=index.copy())
    meta["Organ"] = meta.index.str.split("_").str[0]
    meta["CellType"] = meta.index.str.split("_").str[-1]
    return meta


def map_to_coarse(ct):
    ct = str(ct).lower()

    if ("pmn" in ct) or ("polymorph" in ct) or ("neutro" in ct):
        return "PMX"
    if ("macrophage" in ct) or ("mono" in ct) or ("dc" in ct) or ("dendritic" in ct):
        return "MIC"
    if ("endothelial" in ct) or ("cap" in ct) or ("vein" in ct) or ("arter" in ct):
        return "END"
    if (
        ("fibro" in ct)
        or ("myofibro" in ct)
        or ("pericyte" in ct)
        or ("smooth_muscle" in ct)
        or ("stromal" in ct)
    ):
        return "MES"
    if (
        ("at1" in ct)
        or ("at2" in ct)
        or ("club" in ct)
        or ("ciliated" in ct)
        or ("epithelial" in ct)
    ):
        return "EPI"
    return "OTHER"


def load_pediatric_pseudobulk():
    counts = pd.read_csv(PED_COUNTS_PATH, sep="\t")
    groups = pd.read_csv(PED_GROUPS_PATH, sep="\t")

    counts = counts.set_index(counts.columns[0])
    counts.index = counts.index.astype(str).str.strip()
    counts.columns = counts.columns.astype(str).str.strip()

    groups = groups.copy()
    groups.columns = ["sample", "cell_type"]
    groups["sample"] = groups["sample"].astype(str).str.strip()
    groups["cell_type"] = groups["cell_type"].astype(str).str.strip()

    groups = groups[groups["sample"].isin(counts.columns)].copy()
    counts = counts[groups["sample"]]

    groups["donor"] = groups["sample"].str.split("__").str[0]
    groups["celltype"] = groups["cell_type"]

    pseudobulk = []
    for (donor, ct), rows in groups.groupby(["donor", "celltype"]):
        cols = rows["sample"].tolist()
        expr = counts[cols].mean(axis=1)

        clean_ct = ct.replace(" ", "_").replace("/", "_").replace("-", "_")
        name = f"{donor}_{clean_ct}"
        pseudobulk.append(expr.rename(name))

    pediatric_x = pd.DataFrame(pseudobulk)
    pediatric_x = pediatric_x.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return pediatric_x


def build_pediatric_metadata(index):
    meta = pd.DataFrame(index=index.copy())
    meta["Organ"] = meta.index.str.split("_").str[0]
    meta["CellType_raw"] = meta.index.str.split("_", n=1).str[1]
    meta["CellType"] = meta["CellType_raw"].map(map_to_coarse)
    return meta


def compute_pairwise_lipids(df, metadata, group_col, group1, group2, pseudocount=1e-6):
    common_idx = df.index.intersection(metadata.index)
    df = df.loc[common_idx].copy()
    meta = metadata.loc[common_idx].copy()

    mask = meta[group_col].isin([group1, group2])
    df_sub = df.loc[mask].apply(pd.to_numeric, errors="coerce")
    meta_sub = meta.loc[mask]

    g1_idx = meta_sub[group_col] == group1
    g2_idx = meta_sub[group_col] == group2

    n1, n2 = int(g1_idx.sum()), int(g2_idx.sum())
    print(f"{group1} vs {group2} | n={n1} vs n={n2}")

    required_cols = [
        "Lipid", "Group1", "Group2", "log2FC", "t_stat",
        "pval", "FDR", "EffectSize", "Direction"
    ]

    if n1 < 2 or n2 < 2:
        return pd.DataFrame(columns=required_cols)

    results = []
    for lipid in df_sub.columns:
        vals1 = df_sub.loc[g1_idx, lipid].astype(float).dropna().values
        vals2 = df_sub.loc[g2_idx, lipid].astype(float).dropna().values

        if len(vals1) < 2 or len(vals2) < 2:
            continue

        mean1, mean2 = np.mean(vals1), np.mean(vals2)
        logfc = np.log2((mean1 + pseudocount) / (mean2 + pseudocount))
        stat, pval = ttest_ind(vals1, vals2, equal_var=False)

        results.append({
            "Lipid": lipid,
            "Group1": group1,
            "Group2": group2,
            "log2FC": logfc,
            "t_stat": stat,
            "pval": pval,
            "EffectSize": abs(mean1 - mean2),
            "Direction": f"{group1}_up" if logfc > 0 else f"{group2}_up"
        })

    out = pd.DataFrame(results)
    if out.empty:
        return pd.DataFrame(columns=required_cols)

    out["FDR"] = multipletests(out["pval"].values, method="fdr_bh")[1]
    out = out.sort_values(["FDR", "EffectSize"], ascending=[True, False]).reset_index(drop=True)

    return out[required_cols]


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
            "Lipid", "Group1", "Group2", "log2FC", "t_stat", "pval", "FDR",
            "EffectSize", "Direction", "Comparison"
        ])

    return pd.concat(valid_list, ignore_index=True)


def make_logfc_heatmap(valid_df, title, outname):
    if valid_df.empty:
        print(f"No data for heatmap: {title}")
        return

    heat_df = valid_df.pivot_table(
        index="Lipid",
        columns="Comparison",
        values="log2FC",
        aggfunc="mean"
    )

    if heat_df.empty:
        print(f"Heatmap pivot empty: {title}")
        return

    fig_h = max(8, min(24, 0.18 * heat_df.shape[0]))
    fig_w = max(10, min(18, 1.2 * heat_df.shape[1]))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(heat_df.values, aspect="auto")
    plt.colorbar(label="log2FC")
    plt.xticks(range(len(heat_df.columns)), heat_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(heat_df.index)), heat_df.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.close()


def build_combined_source_df(true_valid_df, pediatric_pred_valid_df):
    true_df = true_valid_df.copy()
    ped_df = pediatric_pred_valid_df.copy()

    true_df["Lipid"] = true_df["Lipid"].astype(str).str.strip()
    ped_df["Lipid"] = ped_df["Lipid"].astype(str).str.strip()

    true_df["Source"] = "True"
    ped_df["Source"] = "Pediatric_Pred"

    combined_df = pd.concat([true_df, ped_df], ignore_index=True)
    return combined_df


def save_per_comparison_upsets(combined_df, comparisons, alpha, outdir):
    sig_df = combined_df.loc[
        (combined_df["pval"] < alpha) &
        (combined_df["Source"].isin(["True", "Pediatric_Pred"]))
    ].copy()

    for comp in comparisons:
        sub = sig_df[sig_df["Comparison"] == comp].copy()

        true_set = set(sub.loc[sub["Source"] == "True", "Lipid"])
        ped_set = set(sub.loc[sub["Source"] == "Pediatric_Pred", "Lipid"])

        print(f"{comp}: True={len(true_set)} Pediatric_Pred={len(ped_set)} Overlap={len(true_set & ped_set)}")

        if len(true_set) == 0 and len(ped_set) == 0:
            continue

        upset_data = from_contents({
            "True_DE": true_set,
            "Pediatric_Pred_DE": ped_set
        })

        UpSet(
            upset_data,
            subset_size="count",
            show_counts=True
        ).plot()

        plt.suptitle(f"{comp} significant differential lipids (p < {alpha})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"upset_{comp}.png"), dpi=300, bbox_inches="tight")
        plt.close()


def summarize_overlap(combined_df, comparisons, alpha):
    sig_df = combined_df.loc[
        (combined_df["pval"] < alpha) &
        (combined_df["Source"].isin(["True", "Pediatric_Pred"]))
    ].copy()

    summary = []
    for comp in comparisons:
        sub = sig_df[sig_df["Comparison"] == comp].copy()

        true_set = set(sub.loc[sub["Source"] == "True", "Lipid"])
        ped_set = set(sub.loc[sub["Source"] == "Pediatric_Pred", "Lipid"])

        both = true_set & ped_set
        true_only = true_set - ped_set
        ped_only = ped_set - true_set

        union_n = len(true_set | ped_set)

        summary.append({
            "Comparison": comp,
            "True_only": len(true_only),
            "Pediatric_Pred_only": len(ped_only),
            "Both": len(both),
            "True_total": len(true_set),
            "Pediatric_Pred_total": len(ped_set),
            "Jaccard": (len(both) / union_n) if union_n > 0 else 0.0
        })

    return pd.DataFrame(summary).sort_values("Jaccard", ascending=False).reset_index(drop=True)


def save_overlap_barplots(summary_df, outpath):
    if summary_df.empty:
        print("No overlap summary to plot.")
        return

    n = len(summary_df)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, summary_df.iterrows()):
        vals = [row["True_only"], row["Both"], row["Pediatric_Pred_only"]]
        labels = ["True only", "Both", "Pediatric pred only"]

        ax.bar(labels, vals)
        ax.set_title(f'{row["Comparison"]}\nJaccard={row["Jaccard"]:.2f}')
        ax.set_ylabel("Number of significant lipids")
        ax.tick_params(axis="x", rotation=20)

    for ax in axes[len(summary_df):]:
        ax.axis("off")

    fig.suptitle("Significant Lipid Overlap by Cell-Type Pair", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ensure_outdir(OUTDIR)

    print("Loading model bundle...")
    model, scaler_x, scaler_y, x_cols, y_cols = load_model_bundle(MODEL_PATH)

    print("Loading internal matched RNA/lipid data...")
    x_df, y_df = load_internal_data()
    print("Internal X shape:", x_df.shape)
    print("Internal Y shape:", y_df.shape)

    print("Preparing internal holdout...")
    x_holdout, y_holdout = split_internal_holdout(x_df, y_df, HOLDOUT_SAMPLES)
    print("Internal holdout X shape:", x_holdout.shape)
    print("Internal holdout Y shape:", y_holdout.shape)

    print("Predicting internal holdout lipids...")
    pred_holdout_df = predict_lipids(model, scaler_x, x_holdout, x_cols, y_cols)

    common_lipids = y_holdout.columns.intersection(pred_holdout_df.columns)
    y_holdout_aligned = y_holdout.loc[pred_holdout_df.index, common_lipids].copy()

    nan_counts = y_holdout_aligned.isna().sum()
    keep_cols = nan_counts[nan_counts <= 16].index
    y_holdout_aligned = y_holdout_aligned[keep_cols].fillna(0)

    meta_holdout = build_internal_metadata(y_holdout_aligned.index)

    print("Loading pediatric pseudobulk data...")
    pediatric_x = load_pediatric_pseudobulk()
    pediatric_x.index = pediatric_x.index.astype(str).str.strip()
    pediatric_x.columns = pediatric_x.columns.astype(str).str.strip()

    print("Predicting pediatric lipids...")
    pediatric_pred_df = predict_lipids(model, scaler_x, pediatric_x, x_cols, y_cols)

    meta_pediatric = build_pediatric_metadata(pediatric_pred_df.index)
    meta_pediatric5 = meta_pediatric[
        meta_pediatric["CellType"].isin(["END", "EPI", "MES", "MIC", "PMX"])
    ].copy()
    pediatric_pred5 = pediatric_pred_df.loc[meta_pediatric5.index].copy()

    print("Pediatric coarse cell type counts:")
    print(meta_pediatric5["CellType"].value_counts(dropna=False))

    print("Computing pairwise stats: internal truth...")
    true_valid_df = build_valid_df(
        df=y_holdout_aligned,
        metadata=meta_holdout,
        group_col="CellType",
        top_n=TOP_N,
        pairs=CELLTYPE_PAIRS
    )

    print("Computing pairwise stats: pediatric external prediction...")
    pediatric_pred_valid_df = build_valid_df(
        df=pediatric_pred5,
        metadata=meta_pediatric5,
        group_col="CellType",
        top_n=TOP_N,
        pairs=CELLTYPE_PAIRS
    )

    print("true_valid_df:", true_valid_df.shape)
    print("pediatric_pred_valid_df:", pediatric_pred_valid_df.shape)

    true_valid_df.to_csv(os.path.join(OUTDIR, "true_valid_df.csv"), index=False)
    pediatric_pred_valid_df.to_csv(os.path.join(OUTDIR, "pediatric_pred_valid_df.csv"), index=False)

    print("Saving heatmaps...")
    make_logfc_heatmap(
        true_valid_df,
        title="True lipid log2FC across cell-type comparisons",
        outname=os.path.join(OUTDIR, "true_lipid_heatmap.png")
    )

    make_logfc_heatmap(
        pediatric_pred_valid_df,
        title="Pediatric predicted lipid log2FC across cell-type comparisons",
        outname=os.path.join(OUTDIR, "pediatric_pred_lipid_heatmap.png")
    )

    combined_df = build_combined_source_df(true_valid_df, pediatric_pred_valid_df)
    combined_df.to_csv(os.path.join(OUTDIR, "combined_true_pediatric_pred_valid_df.csv"), index=False)

    comparisons = sorted(combined_df["Comparison"].dropna().unique())

    print("Saving per-comparison UpSet plots...")
    save_per_comparison_upsets(
        combined_df=combined_df,
        comparisons=comparisons,
        alpha=ALPHA,
        outdir=OUTDIR
    )

    print("Saving overlap summary...")
    summary_df = summarize_overlap(
        combined_df=combined_df,
        comparisons=comparisons,
        alpha=ALPHA
    )
    summary_df.to_csv(os.path.join(OUTDIR, "overlap_summary.csv"), index=False)
    print(summary_df)

    save_overlap_barplots(
        summary_df,
        os.path.join(OUTDIR, "significant_lipid_overlap_by_pair.png")
    )

    print(f"\nDone. Outputs saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
