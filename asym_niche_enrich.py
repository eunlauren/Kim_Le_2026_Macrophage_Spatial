# ASYMMETRIC NICHE ENRICHMENT
# Author(s): Lauren Kim, Jennifer Le (UCSF)
# Asymmetric niche enrichment extending on CellCharter's native niche enrichment (Figure 5) for Kim, Le et al. 2026

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import cellcharter as cc
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

# SETTINGS
INPUT_H5AD = "FINAL_ALLRM_XENIUM_062026.h5ad"
OUTDIR = Path("ASYM_NICHE_ENRICH_OUTPUT")

CLUSTER_KEY_POOLED = "cellcharter_L2_K8_autok"
CLUSTER_KEY_PATIENT = "cellcharter"
CONNECT_KEY = "delaunay_uw"
LIBRARY_KEY = "patient"
N_JOBS = 1

# Pooled (all patients together)
N_PERMS_POOLED = 1000
ALPHA_POOLED = 0.05

# Per patient, then combined across patients
N_PERMS_PATIENT = 10000
ALPHA_PATIENT = 0.05
ALPHA_PAIR = 0.05
MIN_K_EFF = 3
MIN_SIGN_CONSISTENT = 5
MIN_SIG_CONSISTENT = 5

adata = sc.read_h5ad(INPUT_H5AD)

# POOLED
ne_asym_pvals = cc.gr.nhood_enrichment(
    adata,
    cluster_key=CLUSTER_KEY_POOLED,
    connectivity_key=CONNECT_KEY,
    only_inter=True,
    symmetric=False,
    pvalues=True,
    n_perms=N_PERMS_POOLED,
    n_jobs=N_JOBS,
    copy=True,
)

pvals_obj = ne_asym_pvals["pvalue"]
pvals_arr = pvals_obj.to_numpy(dtype=float)
cluster_labels = list(pvals_obj.index)

print(f"pvals_arr shape={pvals_arr.shape}, dtype={pvals_arr.dtype}.")

K = pvals_arr.shape[0]
off_diag = ~np.eye(K, dtype=bool)
pflat = pvals_arr[off_diag]
pflat_clipped = np.clip(pflat, 1.0 / (N_PERMS_POOLED + 1), 1.0)

reject, qvals_flat, _, _ = multipletests(pflat_clipped, alpha=ALPHA_POOLED, method="fdr_bh")

qvals = np.full_like(pvals_arr, np.nan)
qvals[off_diag] = qvals_flat

sig_q05 = np.zeros_like(pvals_arr, dtype=bool)
sig_q05[off_diag] = reject

enrich_arr = ne_asym_pvals["enrichment"].to_numpy()

records = []
for i in range(K):
    for j in range(K):
        if i == j:
            continue
        records.append({
            "from_cluster": cluster_labels[i],
            "to_cluster": cluster_labels[j],
            "NE_asym": enrich_arr[i, j],
            "pvalue": pvals_arr[i, j],
            "qvalue_BH": qvals[i, j],
            "sig_q05": sig_q05[i, j],
        })

results_df = (pd.DataFrame(records).sort_values("qvalue_BH").reset_index(drop=True))

def to_labeled_df(mat_obj, labels, present_labels=None):
    df = mat_obj.copy()
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    df = df.reindex(index=labels, columns=labels).astype(float)

    if present_labels is not None:
        present_labels = set(map(str, present_labels))
        absent = [c for c in labels if c not in present_labels]
        if absent:
            df.loc[absent, :] = np.nan
            df.loc[:, absent] = np.nan

    return df

# PER PATIENT
all_clusters = [str(c) for c in adata.obs[CLUSTER_KEY_PATIENT].cat.categories]
K = len(all_clusters)

patients = (
    adata.obs[LIBRARY_KEY]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
patients = sorted(patients)

off_diag = ~np.eye(K, dtype=bool)
p_floor = 1.0 / (N_PERMS_PATIENT + 1)
p_ceil = 1.0 - p_floor

per_patient = {}
for pid in patients:
    mask = adata.obs[LIBRARY_KEY].astype(str).values == pid
    sub = adata[mask].copy()

    sub.obs[CLUSTER_KEY_PATIENT] = (
        sub.obs[CLUSTER_KEY_PATIENT]
        .astype("category")
        .cat.set_categories(adata.obs[CLUSTER_KEY_PATIENT].cat.categories)
    )

    present = sub.obs[CLUSTER_KEY_PATIENT].dropna().astype(str).unique().tolist()
    if len(present) < 2:
        continue
    # CC nhood_enrichment
    res = cc.gr.nhood_enrichment(
        sub,
        cluster_key=CLUSTER_KEY_PATIENT,
        connectivity_key=CONNECT_KEY,
        only_inter=True,
        symmetric=False,
        pvalues=True,
        n_perms=N_PERMS_PATIENT,
        n_jobs=N_JOBS,
        copy=True,
    )

    enr_arr = to_labeled_df(
        res["enrichment"],
        all_clusters,
        present_labels=present,
    ).to_numpy(dtype=float)

    p_right_arr = to_labeled_df(
        res["pvalue"],
        all_clusters,
        present_labels=present,
    ).to_numpy(dtype=float, copy=True)

    valid_p = off_diag & np.isfinite(enr_arr) & np.isfinite(p_right_arr)

    p_right_arr[valid_p] = np.clip(p_right_arr[valid_p], p_floor, p_ceil)

    p_two_arr = np.full((K, K), np.nan, dtype=float)
    p_two_arr[valid_p] = np.minimum(
        1.0,
        2.0 * np.minimum(p_right_arr[valid_p], 1.0 - p_right_arr[valid_p]),
    )

    q_two_arr = np.full((K, K), np.nan, dtype=float)
    if valid_p.sum() > 0:
        q_two_arr[valid_p] = multipletests(
            p_two_arr[valid_p],
            alpha=ALPHA_PATIENT,
            method="fdr_bh",
        )[1]

    per_patient[pid] = {
        "enrichment": enr_arr,
        "p_right": p_right_arr,
        "p_two": p_two_arr,
        "q_two": q_two_arr,
    }

# COMBINE ACROSS PATIENTS
patient_ids = list(per_patient.keys())

rows = []
for i in range(K):
    for j in range(K):
        if i == j:
            continue

        ne_vec = np.array([per_patient[p]["enrichment"][i, j] for p in patient_ids])
        p_right_vec = np.array([per_patient[p]["p_right"][i, j] for p in patient_ids])
        q_two_vec = np.array([per_patient[p]["q_two"][i, j] for p in patient_ids])

        valid = np.isfinite(ne_vec) & np.isfinite(p_right_vec)
        ne_v = ne_vec[valid]
        p_right_v = p_right_vec[valid]
        q_two_v = q_two_vec[valid]

        k_eff = int(valid.sum())
        if k_eff < MIN_K_EFF:
            continue

        signs = np.sign(ne_v)
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())

        if n_pos > n_neg:
            majority = 1
        elif n_neg > n_pos:
            majority = -1
        else:
            majority = 0

        if majority == 0:
            n_consistent_sign = 0
            n_consistent_sig_q05 = 0
        else:
            same_sign = signs == majority
            n_consistent_sign = int(same_sign.sum())
            n_consistent_sig_q05 = int(
                (same_sign & np.isfinite(q_two_v) & (q_two_v < ALPHA_PATIENT)).sum()
            )

        z_vec = norm.isf(np.clip(p_right_v, p_floor, p_ceil))
        z_combined = float(z_vec.sum() / np.sqrt(k_eff))

        p_stouffer = float(2.0 * norm.sf(abs(z_combined)))
        z_signs = np.sign(z_vec)
        n_z_ne_mismatch = int(
            ((z_signs != 0) & (signs != 0) & (z_signs != signs)).sum()
        )
        rows.append({
            "from": all_clusters[i],
            "to": all_clusters[j],
            "k_eff": k_eff,
            "NE_mean": float(np.mean(ne_v)),
            "NE_median": float(np.median(ne_v)),
            "NE_min": float(np.min(ne_v)),
            "NE_max": float(np.max(ne_v)),
            "majority_sign": int(majority),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_consistent_sign": n_consistent_sign,
            "n_consistent_sig_q05": n_consistent_sig_q05,
            "n_z_ne_mismatch": n_z_ne_mismatch,
            "z_stouffer": z_combined,
            "p_stouffer": p_stouffer,
        })

summary = pd.DataFrame(rows)

m = summary["p_stouffer"].notna()
summary.loc[m, "q_stouffer"] = multipletests(
    summary.loc[m, "p_stouffer"].values,
    alpha=ALPHA_PAIR,
    method="fdr_bh",
)[1]

robust = (
    summary[
        (summary["majority_sign"] != 0)
        & (summary["n_consistent_sign"] >= MIN_SIGN_CONSISTENT)
        & (summary["n_consistent_sig_q05"] >= MIN_SIG_CONSISTENT)
        & (summary["q_stouffer"] < ALPHA_PAIR)
    ]
    .sort_values("z_stouffer", key=lambda s: np.abs(s), ascending=False)
    .reset_index(drop=True)
)

# SAVE
OUTDIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTDIR / "pooled_pair_enrichment.tsv", sep="\t", index=False)

ii, jj = np.where(off_diag)
per_patient_df = pd.concat(
    [
        pd.DataFrame({
            LIBRARY_KEY: pid,
            "from": [all_clusters[i] for i in ii],
            "to": [all_clusters[j] for j in jj],
            "NE": d["enrichment"][ii, jj],
            "p_right": d["p_right"][ii, jj],
            "p_two": d["p_two"][ii, jj],
            "q_two": d["q_two"][ii, jj],
        })
        for pid, d in per_patient.items()
    ],
    ignore_index=True,
)
per_patient_df.to_csv(OUTDIR / "per_patient_pair_enrichment.tsv", sep="\t", index=False)
summary.to_csv(OUTDIR / "cross_patient_pair_summary.tsv", sep="\t", index=False)
robust.to_csv(OUTDIR / "robust_pairs.tsv", sep="\t", index=False)

print(f"Patients analysed: {len(per_patient)}/{len(patients)}. Pairs summarised: {len(summary)}. Robust pairs: {len(robust)}.")
print(f"Outputs written to {OUTDIR.resolve()}")
