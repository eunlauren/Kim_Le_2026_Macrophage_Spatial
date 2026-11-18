# PAIR ENRICHMENT PERMUTATIONS
# Author(s): Lauren Kim, Jennifer Le (UCSF)
# Pair enrichment permutation (Figure 5) for Kim, Le et al. 2026

# IMPORTS
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import norm
from scipy.stats import chi2
from pathlib import Path
from typing import Optional, Dict, Sequence, Tuple, Union, Mapping

# SETTINGS
INPUT_H5AD = "FINAL_ALLRM_XENIUM_062026.h5ad"
OUTDIR = Path("PAIR_ENRICH_OUTPUT")

SAMPLE_KEY = "patient"
CELLTYPE_KEY = "celltype_1"
NICHE_KEY = "cellcharter_L2_K8_autok"
LABEL_KEY = "mixing_inference_labels"
GRAPH_KEY = "delaunay_uw_connectivities"

TRM_CELLTYPE = "Macrophage"
DNMAC_CELLTYPE = "Ductal niche macrophage"
DNMAC_LABEL = "DNMAC"
FOCUS_NICHE_TO_LABEL = {
    "C1": "C1_STRM",
    "C2": "C2_STRM",
    "C3": "C3_STRM",
    "C4": "C4_STRM",
    "C5": "C5_STRM",
    "C7": "C7_STRM",
}
FOCUS_LABELS = list(FOCUS_NICHE_TO_LABEL.values()) + [DNMAC_LABEL]
PARTNER_CELLTYPES = [
    "Adipocyte",
    "B cell",
    "Connective tissue",
    "Ductal niche macrophage",
    "Dendritic cell",
    "Endothelial",
    "Fibroblast",
    "LHS",
    "LASP",
    "Lymphatic endothelial",
    "Macrophage",
    "Monocyte",
    "Mural",
    "B-Myo",
    "Plasma cell",
    "T cell",
]

MIN_CELLS_PER_SAMPLE = 20
MIN_EXPECTED = 5.0
N_PERMS = 4999
BASE_SEED = 1000  # patient i uses seed BASE_SEED + i

WEIGHTS = "equal"
MIN_CONTRIB = 4
PSEUDOCOUNT = 0.5
ALPHA = 0.05
CONSISTENCY_THRESH = 0.75
N_TOP = 5

# LOAD IN DATA
adata = sc.read_h5ad(INPUT_H5AD)

def build_labels_for_mixing(
    obs: pd.DataFrame,
    *,
    celltype_key: str = "celltype_1",
    niche_key: str = "cellcharter_L2_K8_autok",
    trm_celltype: str = "Macrophage",
    dnmac_celltype: str = "Ductal niche macrophage",
    focus_niche_to_label: Optional[Mapping] = None,
    dnmac_label: str = "DNMAC",
    nonfocus_trm_policy: str = "keep_celltype",
    nonfocus_trm_label: str = "STRM_nonfocus",
    nan_niche_trm_policy: str = "nonfocus",
    partner_celltypes: Optional[Sequence[str]] = None,
    unknown_policy: str = "other",
    other_label: str = "Other",
    rare_min_cells_global: int = 0,
    protect_labels: Optional[Sequence[str]] = None,
    require_any_focus_trm: bool = True,
    return_qc: bool = True,
) -> Tuple[pd.Categorical, Optional[dict]]:

    if focus_niche_to_label is None:
        raise ValueError(
            "Warning: focus_niche_to_label is required."
        )

    if nonfocus_trm_policy not in {"keep_celltype", "relabel", "other"}:
        raise ValueError("Warning:: nonfocus_trm_policy must be in {'keep_celltype','relabel','other'}.")

    if nan_niche_trm_policy not in {"nonfocus", "other"}:
        raise ValueError("Warning: nan_niche_trm_policy must be in {'nonfocus','other'}.")

    if unknown_policy not in {"other", "error"}:
        raise ValueError("Warning: unknown_policy must be in {'other','error'}.")

    for k in (celltype_key, niche_key):
        if k not in obs.columns:
            raise KeyError(f"{k!r} not found.")

    n = len(obs)

    ct = obs[celltype_key].astype("string")
    nm = obs[niche_key].astype("string")
    ct_np = ct.to_numpy(dtype=object)

    focus_map = {str(k): str(v) for k, v in focus_niche_to_label.items()}
    niche_anchor = nm.map(focus_map)

    is_trm = ct.eq(trm_celltype).fillna(False).to_numpy(dtype=bool)
    is_dnmac = ct.eq(dnmac_celltype).fillna(False).to_numpy(dtype=bool)

    niche_present = nm.notna().to_numpy(dtype=bool)
    anchor_present = niche_anchor.notna().to_numpy(dtype=bool)
    anchor_np = niche_anchor.to_numpy(dtype=object)

    labels = np.full(n, None, dtype=object)


    labels[is_dnmac] = dnmac_label
    trm_focus = is_trm & anchor_present
    labels[trm_focus] = anchor_np[trm_focus]
    trm_nonfocus = is_trm & niche_present & (~anchor_present)

    if nonfocus_trm_policy == "keep_celltype":
        labels[trm_nonfocus] = trm_celltype
    elif nonfocus_trm_policy == "relabel":
        labels[trm_nonfocus] = nonfocus_trm_label
    else:
        labels[trm_nonfocus] = other_label

    trm_nan = is_trm & (~niche_present)

    if nan_niche_trm_policy == "nonfocus":
        if nonfocus_trm_policy == "keep_celltype":
            labels[trm_nan] = trm_celltype
        elif nonfocus_trm_policy == "relabel":
            labels[trm_nan] = nonfocus_trm_label
        else:
            labels[trm_nan] = other_label
    else:
        labels[trm_nan] = other_label

    partner_mask = ~(is_trm | is_dnmac)

    if partner_celltypes is not None:
        partner_celltypes = [str(x) for x in partner_celltypes]

        known = ct.isin(partner_celltypes).to_numpy(dtype=bool)
        unknown_partner = partner_mask & (~known)

        if unknown_policy == "error" and unknown_partner.any():
            raise ValueError(
                "Unmapped partner cell types:\n"
                f"{ct[unknown_partner].value_counts().head(20)}"
            )

        labels[partner_mask & known] = ct_np[partner_mask & known]
        labels[unknown_partner] = other_label

    else:
        ct_na = ct.isna().to_numpy(dtype=bool)
        labels[partner_mask & (~ct_na)] = ct_np[partner_mask & (~ct_na)]
        labels[partner_mask & ct_na] = other_label

    labels[pd.isna(labels)] = other_label
    labels = pd.Series(labels, index=obs.index, dtype="object").astype("string")

    anchor_labels = list(dict.fromkeys(list(focus_map.values()) + [dnmac_label]))

    observed_niche_values = set(nm.dropna().astype(str).unique().tolist())
    focus_map_keys = set(focus_map.keys())
    focus_map_keys_observed = sorted(focus_map_keys & observed_niche_values)
    focus_map_keys_missing_from_obs = sorted(focus_map_keys - observed_niche_values)

    if require_any_focus_trm and int(trm_focus.sum()) == 0:
        raise ValueError(
            "No focus TRM cells were assigned."
        )

    protect = set(anchor_labels) | set(protect_labels or [])
    collapsed_rare = []

    if rare_min_cells_global and rare_min_cells_global > 0:
        vc = labels.value_counts(dropna=False)
        collapsed_rare = [
            str(lab)
            for lab, c in vc.items()
            if int(c) < int(rare_min_cells_global)
            and str(lab) not in protect
            and str(lab) != other_label
        ]

        if collapsed_rare:
            labels = labels.where(~labels.isin(collapsed_rare), other_label).astype("string")

    present = set(labels.dropna().astype(str).unique().tolist())

    if partner_celltypes is not None:
        partner_present = [
            p for p in partner_celltypes
            if p in present and p not in set(anchor_labels) and p != other_label
        ]
        extras = sorted(
            present
            - set(anchor_labels)
            - set(partner_present)
            - {other_label}
        )
        partner_present = partner_present + extras
    else:
        partner_present = sorted(
            present
            - set(anchor_labels)
            - {other_label}
        )

    cat_order = anchor_labels + partner_present + [other_label]
    observed = [c for c in cat_order if c in present]

    labels_cat = pd.Categorical(labels, categories=observed, ordered=False)

    qc = None
    if return_qc:
        anchor_counts = labels.value_counts().reindex(anchor_labels, fill_value=0)

        qc = {
            "global_label_counts": labels.value_counts(),
            "anchor_counts": anchor_counts,
            "n_anchor_cells": int(labels.isin(anchor_labels).sum()),
            "n_trm_focus": int(trm_focus.sum()),
            "n_trm_nonfocus": int(trm_nonfocus.sum()),
            "n_trm_nan_niche": int(trm_nan.sum()),
            "n_dnmac": int(is_dnmac.sum()),
            "n_other": int((labels == other_label).sum()),
            "focus_map": focus_map,
            "focus_map_keys_observed": focus_map_keys_observed,
            "focus_map_keys_missing_from_obs": focus_map_keys_missing_from_obs,
            "observed_niche_values": sorted(list(observed_niche_values)),
            "collapsed_rare_labels": collapsed_rare,
            "category_order": observed,
        }

    return labels_cat, qc

mixing_labels, mixing_labels_qc = build_labels_for_mixing(
    adata.obs,
    celltype_key=CELLTYPE_KEY,
    niche_key=NICHE_KEY,
    trm_celltype=TRM_CELLTYPE,
    dnmac_celltype=DNMAC_CELLTYPE,
    focus_niche_to_label=FOCUS_NICHE_TO_LABEL,
    dnmac_label=DNMAC_LABEL,
    nonfocus_trm_policy="keep_celltype",
    nan_niche_trm_policy="nonfocus",
    partner_celltypes=PARTNER_CELLTYPES,
    unknown_policy="error",
    rare_min_cells_global=0,
    require_any_focus_trm=True,
    return_qc=True
)
adata.obs[LABEL_KEY] = mixing_labels

def generate_label_qc_tables(
    adata,
    sample_key: str = "patient",
    label_key: str = "mixing_inference_labels",
    *,
    categories: Optional[Sequence[str]] = None,
    dropna: bool = False,
) -> pd.DataFrame:
    for k in (sample_key, label_key):
        if k not in adata.obs.columns:
            raise KeyError(f"{k!r} not found in adata.obs.")

    s = adata.obs[sample_key]
    lab = adata.obs[label_key].astype("string")

    if lab.isna().any():
        na_count = int(lab.isna().sum())
        raise ValueError(
            f"label_key {label_key!r} contains {na_count} missing labels."
        )

    if categories is not None:
        final_categories = list(dict.fromkeys(map(str, categories)))

        present = set(lab.dropna().astype(str).unique())
        missing = present - set(final_categories)

        if missing:
            raise ValueError(
                f"label_key {label_key!r} has values absent from `categories`."
            )

        cat_dtype = pd.CategoricalDtype(categories=final_categories, ordered=True)
        lab = lab.astype(cat_dtype)

    else:
        lab = lab.astype("category")
        final_categories = list(lab.cat.categories)

    node_counts = pd.crosstab(s, lab, dropna=dropna)


    node_counts = node_counts.reindex(columns=final_categories, fill_value=0)
    expected_per_sample = s.value_counts(dropna=False).reindex(node_counts.index, fill_value=0)
    observed_per_sample = node_counts.sum(axis=1)

    bad = observed_per_sample.ne(expected_per_sample)
    if bad.any():
        raise ValueError(
            "Node count table row totals do not match adata.obs sample counts."
        )

    return node_counts

def build_testable_interaction_mask(
    node_counts_df: pd.DataFrame,
    *,
    min_cells_per_sample: int = 30,
    focus_labels: Optional[Sequence[str]] = None,
    include_diagonal: bool = True,
    return_pairs: bool = False,
) -> Dict[str, object]:

    if min_cells_per_sample < 0:
        raise ValueError("Warning: min_cells_per_sample must be non-negative.")

    if node_counts_df.shape[1] == 0:
        raise ValueError("Warning: node_counts_df has zero label columns.")

    labels = list(node_counts_df.columns)
    labels_str = [str(x) for x in labels]
    K = len(labels)

    focus_set = None
    if focus_labels is not None:
        focus_set = set(map(str, focus_labels))
        col_set = set(labels_str)

        inter = focus_set & col_set
        if len(inter) == 0:
            raise ValueError(
                "None of focus_labels match node_counts columns."
            )

        missing = focus_set - col_set
        if missing:
            warnings.warn(
                "focus_labels absent from node_counts columns."
            )

        is_focus = np.array([lab in focus_set for lab in labels_str], dtype=bool)
    else:
        is_focus = None

    valid_labels = {}
    pair_mask = {}
    pairs = {} if return_pairs else None

    for sample in node_counts_df.index:
        counts = node_counts_df.loc[sample]

        counts_arr = counts.to_numpy()
        if not np.issubdtype(counts_arr.dtype, np.number):
            counts_arr = counts.astype(float).to_numpy()

        if np.any(~np.isfinite(counts_arr)) or np.any(counts_arr < 0):
            raise ValueError(f"Invalid node counts for sample {sample!r}.")

        v = counts_arr >= min_cells_per_sample
        M = np.outer(v, v)

        if not include_diagonal:
            np.fill_diagonal(M, False)

        if is_focus is not None:
            focus_M = (
                np.outer(is_focus, np.ones(K, dtype=bool))
                | np.outer(np.ones(K, dtype=bool), is_focus)
            )
            M = M & focus_M

        valid_labels[sample] = v
        pair_mask[sample] = M

        if return_pairs:
            ii, jj = np.where(M)
            pairs[sample] = [
                (labels[i], labels[j])
                for i, j in zip(ii.tolist(), jj.tolist())
            ]

    out = {
        "labels": labels,
        "valid_labels": valid_labels,
        "pair_mask": pair_mask,
    }

    if return_pairs:
        out["pairs"] = pairs

    return out

categories = list(adata.obs[LABEL_KEY].cat.categories)
node_counts = generate_label_qc_tables(
    adata,
    sample_key=SAMPLE_KEY,
    label_key=LABEL_KEY,
    categories=categories,
    dropna=False,
)

testability = build_testable_interaction_mask(
    node_counts_df=node_counts,
    min_cells_per_sample=MIN_CELLS_PER_SAMPLE,
    focus_labels=FOCUS_LABELS,
    include_diagonal=True,
    return_pairs=False,
)
labels = testability["labels"]
pair_mask_A = testability["pair_mask"]

# PERMUTATION
def upper_triangle_edges_from_sparse(A: sparse.spmatrix):
    A = A.tocoo()
    A.sum_duplicates()
    mask = (A.row < A.col) & (A.data != 0)
    return A.row[mask].astype(np.int64), A.col[mask].astype(np.int64)

def mixing_counts_from_endpoint_labels(li: np.ndarray, lj: np.ndarray, K: int):
    a = np.minimum(li, lj)
    b = np.maximum(li, lj)
    code = a * K + b
    X = np.bincount(code, minlength=K * K).reshape(K, K).astype(float)
    X = X + X.T - np.diag(np.diag(X))
    return X

def mu_analytic_random_labeling(n_counts: np.ndarray, N: int, m_edges: int):
    if N < 2 or m_edges <= 0:
        K = len(n_counts)
        return np.zeros((K, K), dtype=float)

    denom = N * (N - 1)
    K = len(n_counts)

    outer = np.outer(n_counts, n_counts).astype(float)
    mu = (m_edges * 2.0 / denom) * outer
    diag = (m_edges / denom) * (n_counts * (n_counts - 1))
    np.fill_diagonal(mu, diag.astype(float))
    return mu

def strategy_b_mask_from_mu(mu_analytic: np.ndarray, *, min_expected: float = 5.0, include_diagonal: bool = True):
    M = mu_analytic >= float(min_expected)
    if not include_diagonal:
        np.fill_diagonal(M, False)
    return M

def permute_within_groups(labels_int: np.ndarray, group_ids: np.ndarray, rng: np.random.Generator):
    labels_perm = labels_int.copy()
    group_ids = np.asarray(group_ids)

    order = np.argsort(group_ids, kind="mergesort")
    gids_sorted = group_ids[order]
    boundaries = np.flatnonzero(gids_sorted[1:] != gids_sorted[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(order)]

    for s, e in zip(starts, ends):
        idx = order[s:e]
        labels_perm[idx] = rng.permutation(labels_perm[idx])
    return labels_perm

def per_sample_ct_enrichment(
    *,
    rows_s: np.ndarray,
    cols_s: np.ndarray,
    node_idx_s: np.ndarray,
    labels_int_global: np.ndarray,
    K: int,
    pair_mask_A: np.ndarray,
    min_expected: float = 5.0,
    n_perms: int = 1000,
    seed: int = 0,
    groups_s: np.ndarray | None = None,
    compute_perm_moments: bool = False,
    expected_n_counts: np.ndarray | None = None,
):
    rows_s = np.asarray(rows_s, dtype=np.int64)
    cols_s = np.asarray(cols_s, dtype=np.int64)
    node_idx_s = np.asarray(node_idx_s, dtype=np.int64)
    labels_int_global = np.asarray(labels_int_global, dtype=np.int64)
    pair_mask_A = np.asarray(pair_mask_A, dtype=bool)

    if rows_s.shape != cols_s.shape:
        raise ValueError(
            f"rows_s and cols_s must have the same shape."
        )

    if K <= 0:
        raise ValueError("K must be positive.")

    if pair_mask_A.shape != (K, K):
        raise ValueError(
            f"pair_mask_A shape {pair_mask_A.shape} != ({K}, {K})."
        )

    if n_perms < 1:
        raise ValueError("n_perms must be >= 1 for permutation p-values.")

    if min_expected < 0:
        raise ValueError("min_expected must be non-negative.")

    if node_idx_s.size == 0:
        raise ValueError("node_idx_s is empty, cannot run per-sample enrichment.")

    if labels_int_global.ndim != 1:
        raise ValueError("labels_int_global must be a 1D array.")

    if node_idx_s.min(initial=0) < 0 or node_idx_s.max(initial=0) >= labels_int_global.shape[0]:
        raise ValueError("node_idx_s contains indices outside labels_int_global.")

    if rows_s.size > 0:
        if rows_s.min(initial=0) < 0 or cols_s.min(initial=0) < 0:
            raise ValueError("rows_s/cols_s contain negative indices.")
        if rows_s.max(initial=0) >= labels_int_global.shape[0] or cols_s.max(initial=0) >= labels_int_global.shape[0]:
            raise ValueError("rows_s/cols_s contain indices outside labels_int_global.")

    rng = np.random.default_rng(seed)

    N_total = labels_int_global.shape[0]
    inv = np.full(N_total, -1, dtype=np.int64)
    inv[node_idx_s] = np.arange(len(node_idx_s), dtype=np.int64)

    row_loc = inv[rows_s]
    col_loc = inv[cols_s]

    if (row_loc < 0).any() or (col_loc < 0).any():
        raise ValueError(
            "Edge list contains nodes outside node_idx_s."
        )

    labels_s = labels_int_global[node_idx_s].astype(np.int64)

    if (labels_s < 0).any():
        raise ValueError(
            "Found negative label codes in labels_s. "
        )
    if (labels_s >= K).any():
        bad_codes = np.unique(labels_s[labels_s >= K])
        raise ValueError(
            f"Found label codes >= K={K}: {bad_codes[:20]}: mismatch between labels_int_global and pair_mask_A. "
        )

    if groups_s is not None:
        groups_s = np.asarray(groups_s)
        if groups_s.shape[0] != labels_s.shape[0]:
            raise ValueError(
                "groups_s length must match sample node count."
            )

    li_obs = labels_s[row_loc]
    lj_obs = labels_s[col_loc]
    x_obs = mixing_counts_from_endpoint_labels(li_obs, lj_obs, K=K)

    N = labels_s.size
    m_edges = rows_s.size
    n_counts = np.bincount(labels_s, minlength=K)
    if n_counts.shape[0] != K:
        raise RuntimeError(
            f"Internal error: n_counts length {n_counts.shape[0]} != K {K}."
        )

    if expected_n_counts is not None:
        expected_n_counts = np.asarray(expected_n_counts).ravel()

        if expected_n_counts.shape[0] != K:
            raise ValueError(
                f"expected_n_counts length {expected_n_counts.shape[0]} != K {K}."
            )

        if not np.all(np.isfinite(expected_n_counts)):
            raise ValueError("expected_n_counts contains non-finite values.")

        expected_n_counts_int = expected_n_counts.astype(n_counts.dtype, copy=False)

        if not np.array_equal(n_counts, expected_n_counts_int):
            raise ValueError(
                "Per-sample label counts derived from labels_int_global do not match expected_n_counts from node_counts."
            )

    mu_analytic = mu_analytic_random_labeling(
        n_counts=n_counts,
        N=N,
        m_edges=m_edges,
    )

    if mu_analytic.shape != (K, K):
        raise RuntimeError(
            f"mu_analytic shape {mu_analytic.shape} != ({K}, {K})."
        )

    mask_B = strategy_b_mask_from_mu(
        mu_analytic,
        min_expected=min_expected,
        include_diagonal=True,
    )

    final_mask = pair_mask_A & mask_B

    flat_idx = np.flatnonzero(final_mask.ravel())
    x_obs_flat = x_obs.ravel()[flat_idx]

    if flat_idx.size == 0:
        p_enrich = np.full((K, K), np.nan, dtype=float)
        p_deplete = np.full((K, K), np.nan, dtype=float)
        Z = np.full((K, K), np.nan, dtype=float)

        testability = {
            "sample": None,
            "N_nodes": int(N),
            "m_edges": int(m_edges),
            "min_expected": float(min_expected),
            "n_diag_true_A": int(np.sum(np.diag(pair_mask_A))),
            "n_testable_pairs_final": 0,
            "n_testable_pairs_uppertri_final": 0,
            "min_mu_tested": float("nan"),
            "median_mu_tested": float("nan"),
        }

        return {
            "x_obs": x_obs,
            "mu_analytic": mu_analytic,
            "pair_mask_A": pair_mask_A,
            "mask_B": mask_B,
            "final_mask": final_mask,
            "p_enrich": p_enrich,
            "p_deplete": p_deplete,
            "Z_signed": Z,
            "MU_perm": None,
            "SD_perm": None,
            "testability": testability,
            "n_counts": n_counts,
        }

    ge = np.zeros_like(x_obs_flat, dtype=np.int64)
    le = np.zeros_like(x_obs_flat, dtype=np.int64)

    if compute_perm_moments:
        sum_x = np.zeros_like(x_obs_flat, dtype=float)
        sumsq_x = np.zeros_like(x_obs_flat, dtype=float)

    for _ in range(n_perms):
        if groups_s is None:
            perm_labels = rng.permutation(labels_s)
        else:
            perm_labels = permute_within_groups(labels_s, groups_s, rng)

        li = perm_labels[row_loc]
        lj = perm_labels[col_loc]

        Xb = mixing_counts_from_endpoint_labels(li, lj, K=K).ravel()[flat_idx]

        ge += (Xb >= x_obs_flat)
        le += (Xb <= x_obs_flat)

        if compute_perm_moments:
            sum_x += Xb
            sumsq_x += Xb * Xb

    p_enrich_flat = (ge + 1.0) / (n_perms + 1.0)
    p_deplete_flat = (le + 1.0) / (n_perms + 1.0)

    p_enrich = np.full((K, K), np.nan, dtype=float)
    p_deplete = np.full((K, K), np.nan, dtype=float)

    p_enrich.ravel()[flat_idx] = p_enrich_flat
    p_deplete.ravel()[flat_idx] = p_deplete_flat

    effect = x_obs - mu_analytic
    direction = np.sign(effect).astype(float)

    p_aligned = np.full((K, K), np.nan, dtype=float)

    enriched = effect > 0
    depleted = effect < 0

    p_aligned[final_mask & enriched] = p_enrich[final_mask & enriched]
    p_aligned[final_mask & depleted] = p_deplete[final_mask & depleted]

    p_two_aligned = np.minimum(1.0, 2.0 * p_aligned)
    min_p_two = min(1.0, 2.0 / (n_perms + 1.0))
    p_two_aligned = np.clip(p_two_aligned, min_p_two, 1.0)

    Z = np.full((K, K), np.nan, dtype=float)
    z_mag = norm.isf(p_two_aligned / 2.0)

    Z[final_mask] = direction[final_mask] * z_mag[final_mask]
    Z[final_mask & (direction == 0)] = 0.0

    # optional
    MU_perm = SD_perm = None

    if compute_perm_moments:
        MU = sum_x / n_perms
        VAR = (sumsq_x - n_perms * MU * MU) / max(1, (n_perms - 1))
        SD = np.sqrt(np.maximum(VAR, 0.0))

        MU_perm = np.full((K, K), np.nan, dtype=float)
        SD_perm = np.full((K, K), np.nan, dtype=float)

        MU_perm.ravel()[flat_idx] = MU
        SD_perm.ravel()[flat_idx] = SD

    if final_mask.any():
        tested_mu = mu_analytic[final_mask]
        min_mu = float(np.nanmin(tested_mu))
        med_mu = float(np.nanmedian(tested_mu))
    else:
        min_mu = med_mu = float("nan")

    testability = {
        "sample": None,
        "N_nodes": int(N),
        "m_edges": int(m_edges),
        "min_expected": float(min_expected),
        "n_diag_true_A": int(np.sum(np.diag(pair_mask_A))),
        "n_testable_pairs_final": int(final_mask.sum()),
        "n_testable_pairs_uppertri_final": int(np.triu(final_mask, k=0).sum()),
        "min_mu_tested": min_mu,
        "median_mu_tested": med_mu,
    }

    return {
        "x_obs": x_obs,
        "mu_analytic": mu_analytic,
        "pair_mask_A": pair_mask_A,
        "mask_B": mask_B,
        "final_mask": final_mask,
        "p_enrich": p_enrich,
        "p_deplete": p_deplete,
        "Z_signed": Z,
        "MU_perm": MU_perm,
        "SD_perm": SD_perm,
        "testability": testability,
        "n_counts": n_counts,
    }

K = len(labels)

sample_order=list(node_counts.index)

# GRAPH
A = adata.obsp[GRAPH_KEY]

if not sparse.issparse(A):
    A = sparse.csr_matrix(A)

rows, cols = upper_triangle_edges_from_sparse(A)

sample_values = adata.obs[SAMPLE_KEY].to_numpy()

# LABEL
lab = adata.obs[LABEL_KEY].astype("string")

cat_dtype = pd.CategoricalDtype(categories=list(map(str, labels)), ordered=False)
lab_cat = lab.astype(cat_dtype)

labels_int_global = lab_cat.cat.codes.to_numpy()

results_all = {}

for si, sample in enumerate(sample_order):
    print(f"[{si + 1}/{len(sample_order)}] | Running {sample}...")

    node_idx_s = np.flatnonzero(sample_values == sample)

    edge_mask_s = (sample_values[rows] == sample) & (sample_values[cols] == sample)
    rows_s = rows[edge_mask_s]
    cols_s = cols[edge_mask_s]

    res_s = per_sample_ct_enrichment(
        rows_s=rows_s,
        cols_s=cols_s,
        node_idx_s=node_idx_s,
        labels_int_global=labels_int_global,
        K=K,
        pair_mask_A=pair_mask_A[sample],
        min_expected=MIN_EXPECTED,
        n_perms=N_PERMS,
        seed=BASE_SEED + si,
        groups_s=None,
        compute_perm_moments=False,
        expected_n_counts=node_counts.loc[sample, labels].to_numpy(),
    )
    res_s["testability"]["sample"] = sample
    results_all[sample] = res_s

print("All per-sample permutations finished.")

# PREPARE FOR STATISTICAL TESTS
anchor_idx = [labels.index(a) for a in FOCUS_LABELS]
is_anchor = np.zeros(K, bool); is_anchor[anchor_idx] = True
z_ceiling = norm.isf(1.0/(N_PERMS+1.0))

contrib = np.zeros((K,K), int)
for s in sample_order: contrib += results_all[s]["final_mask"].astype(int)
iu, ju = np.triu_indices(K, k=0)
fam = is_anchor[iu] | is_anchor[ju]
cov = contrib[iu, ju][fam]; ever = cov[cov > 0]
print("n_contrib distribution (anchor upper-tri pairs ever tested):")
print(pd.Series(ever).value_counts().sort_index())
print(f"Tested in >={MIN_CONTRIB} samples: {(ever>=MIN_CONTRIB).sum()}/{ever.size}. Singletons: {(ever==1).sum()}.")

for s in sample_order:
    Z, fm = results_all[s]["Z_signed"], results_all[s]["final_mask"]
    print(s, "Fraction |Z| at ceiling:", round(float(np.mean(np.abs(Z[fm]) >= z_ceiling-1e-9)), 3))

diagc = np.diag(contrib)
print({labels[i]: int(diagc[i]) for i in anchor_idx})

for a in FOCUS_LABELS:
    ai = labels.index(a)
    cnt = sum(results_all[s]["final_mask"][ai, :].any() for s in sample_order)
    print(a, f"{cnt}/{len(sample_order)}", f"<- <{MIN_CONTRIB}: no shared claim" if cnt < MIN_CONTRIB else "")

for s in sample_order:
    Z, fm = results_all[s]["Z_signed"], results_all[s]["final_mask"]
    assert np.array_equal(np.nan_to_num(Z), np.nan_to_num(Z.T))
    assert np.isfinite(Z[fm]).all() and np.isnan(Z[~fm]).all()
print("Symmetry + potential NaN structure okay.")

# STATISTICS
def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)

    mask = np.isfinite(pvals)
    p = pvals[mask]
    m = p.size
    if m == 0:
        return q

    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype=float)

    q_sorted = p[order] * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    idx_mask = np.where(mask)[0]
    q[idx_mask[order]] = q_sorted
    return q

# ALT TEST (Benjamini-Yekutieli)
def fdr_by(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)

    mask = np.isfinite(pvals)
    p = pvals[mask]
    m = p.size
    if m == 0:
        return q

    c_m = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype=float)

    q_sorted = p[order] * m * c_m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    idx_mask = np.where(mask)[0]
    q[idx_mask[order]] = q_sorted
    return q

def stack_metric(
    results_all: Dict[str, dict],
    sample_order: Sequence[str],
    metric: str,
) -> np.ndarray:
    mats = []
    for s in sample_order:
        mats.append(np.asarray(results_all[s][metric]))
    return np.stack(mats, axis=0)

def combine_stouffer_signed(
    Z_stack: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    Z_stack = np.asarray(Z_stack, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if Z_stack.ndim != 3:
        raise ValueError("Z_stack must be (S, K, K).")
    if weights.ndim != 1 or weights.shape[0] != Z_stack.shape[0]:
        raise ValueError("Weights must be shape (S,) matching Z_stack.shape[0].")

    valid = np.isfinite(Z_stack)
    W = weights[:, None, None]

    Z_filled = np.where(valid, Z_stack, 0.0)
    num = np.sum(W * Z_filled, axis=0)
    denom = np.sqrt(np.sum((W ** 2) * valid, axis=0))

    Z_global = np.where(denom > 0, num / denom, np.nan)
    p_two = np.where(np.isfinite(Z_global), 2.0 * norm.sf(np.abs(Z_global)), np.nan)
    n_contrib = valid.sum(axis=0).astype(int)

    return Z_global, p_two, n_contrib

def combine_fisher_two_sided(
    p_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    p_stack = np.asarray(p_stack, dtype=float)
    valid = np.isfinite(p_stack)

    p_clip = np.clip(np.where(valid, p_stack, 1.0), 1e-300, 1.0)

    stat = -2.0 * np.sum(np.where(valid, np.log(p_clip), 0.0), axis=0)
    k = valid.sum(axis=0).astype(int)

    p_global = np.where(k > 0, chi2.sf(stat, df=2 * k), np.nan)
    return p_global, k

def build_cross_sample_ct_table(
    results_all: Dict[str, dict],
    labels: Sequence[str],
    *,
    sample_order: Optional[Sequence[str]] = None,
    weights: Union[str, Dict[str, float]] = "equal",
    include_diagonal: bool = True, # False for heterotypic
    min_contrib: int = 4,
    pseudocount: float = 0.5,
    alpha: float = 0.05,
    consistency_thresh: float = 0.75,
    compute_by_q: bool = True,
) -> pd.DataFrame:
    if sample_order is None:
        sample_order = list(results_all.keys())
    sample_order = [str(s) for s in sample_order]
    K = len(labels)

    Z_stack = stack_metric(results_all, sample_order, "Z_signed")

    if isinstance(weights, dict):
        w = np.array([float(weights[s]) for s in sample_order], dtype=float)
    elif weights == "equal":
        w = np.ones(len(sample_order), dtype=float)
    elif weights == "sqrt_edges":
        w = np.array([np.sqrt(results_all[s]["testability"]["m_edges"]) for s in sample_order], dtype=float)
    elif weights == "sqrt_nodes":
        w = np.array([np.sqrt(results_all[s]["testability"]["N_nodes"]) for s in sample_order], dtype=float)
    else:
        raise ValueError("Weights must be 'equal', 'sqrt_edges', 'sqrt_nodes', or dict.")

    Z_global, p_two_global, n_contrib = combine_stouffer_signed(Z_stack, w)

    p_two_list = []
    for s in sample_order:
        pE = np.asarray(results_all[s]["p_enrich"], dtype=float)
        pD = np.asarray(results_all[s]["p_deplete"], dtype=float)
        m = np.asarray(results_all[s]["final_mask"], dtype=bool)
        p2 = np.minimum(1.0, 2.0 * np.minimum(pE, pD))
        p_two_list.append(np.where(m, p2, np.nan))
    p_fisher, _ = combine_fisher_two_sided(np.stack(p_two_list, axis=0))

    eff_list = []
    for s in sample_order:
        x = np.asarray(results_all[s]["x_obs"], dtype=float)
        mu = np.asarray(results_all[s]["mu_analytic"], dtype=float)
        m = np.asarray(results_all[s]["final_mask"], dtype=bool)
        eff_list.append(np.where(m, np.log((x + pseudocount) / (mu + pseudocount)), np.nan))
    eff_stack = np.stack(eff_list, axis=0)

    # Per-pair median/mean log-enrichment across samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        eff_median = np.nanmedian(eff_stack, axis=0)
        eff_mean = np.nanmean(eff_stack, axis=0)

    valid = np.isfinite(Z_stack)
    n_pos = np.sum((Z_stack > 0) & valid, axis=0).astype(int)
    n_neg = np.sum((Z_stack < 0) & valid, axis=0).astype(int)

    k0 = 0 if include_diagonal else 1
    ii, jj = np.triu_indices(K, k=k0)

    df = pd.DataFrame({
        "i": ii, "j": jj,
        "label_i": [labels[i] for i in ii], "label_j": [labels[j] for j in jj],
        "n_contrib": n_contrib[ii, jj],
        "Z_global": Z_global[ii, jj],
        "p_global_two_sided": p_two_global[ii, jj],
        "p_fisher_two_sided": p_fisher[ii, jj],
        "log_enrich_median": eff_median[ii, jj],
        "log_enrich_mean": eff_mean[ii, jj],
        "n_enriched_samples": n_pos[ii, jj],
        "n_depleted_samples": n_neg[ii, jj],
    })

    low = df["n_contrib"] < min_contrib
    df.loc[low, ["Z_global", "p_global_two_sided", "p_fisher_two_sided"]] = np.nan

    df["q_stouffer"] = fdr_bh(df["p_global_two_sided"].to_numpy())
    df["q_fisher"] = fdr_bh(df["p_fisher_two_sided"].to_numpy())
    # ALT TEST
    if compute_by_q:
        df["q_by"] = fdr_by(df["p_global_two_sided"].to_numpy())

    nc = df["n_contrib"].to_numpy().astype(float)
    frac_pos = np.divide(df["n_enriched_samples"], nc, out=np.zeros(len(df)), where=nc > 0)
    frac_neg = np.divide(df["n_depleted_samples"], nc, out=np.zeros(len(df)), where=nc > 0)
    df["frac_pos"] = frac_pos
    df["frac_neg"] = frac_neg
    df["sign_consistency"] = np.maximum(frac_pos, frac_neg)

    qS = df["q_stouffer"].to_numpy(); qF = df["q_fisher"].to_numpy()
    Zg = df["Z_global"].to_numpy(); ml = df["log_enrich_median"].to_numpy()
    npos = df["n_enriched_samples"].to_numpy(); nneg = df["n_depleted_samples"].to_numpy()
    sign_cons = df["sign_consistency"].to_numpy()
    cond_not_testable = nc < min_contrib
    cond_shared_enr = (qS < alpha) & (Zg > 0) & (frac_pos >= consistency_thresh) & (ml > 0)
    cond_shared_dep = (qS < alpha) & (Zg < 0) & (frac_neg >= consistency_thresh) & (ml < 0)
    cond_hetero = (qF < alpha) & (sign_cons < consistency_thresh) & (npos >= 1) & (nneg >= 1)
    df["category"] = np.select(
        [cond_not_testable, cond_shared_enr, cond_shared_dep, cond_hetero],
        ["not_testable", "shared_enriched", "shared_depleted", "heterogeneous"],
        default="inconclusive",
    )
    return df

cross_sample_df = build_cross_sample_ct_table(
    results_all=results_all,
    labels=labels,
    sample_order=sample_order,
    weights=WEIGHTS,
    include_diagonal=True,
    min_contrib=MIN_CONTRIB,
    pseudocount=PSEUDOCOUNT,
    alpha=ALPHA,
    consistency_thresh=CONSISTENCY_THRESH,
    compute_by_q=True,
)

def summarize_top_shared_pairs(df, focus_labels, *, n_top=5,
                               effect_col="log_enrich_median", verbose=True):
    focus_set = set(map(str, focus_labels))
    li = df["label_i"].astype(str); lj = df["label_j"].astype(str)
    homo = (li == lj).to_numpy()

    a1 = df.assign(anchor=li.to_numpy(), partner=lj.to_numpy())
    a2 = df.assign(anchor=lj.to_numpy(), partner=li.to_numpy())
    view = pd.concat([a1, a2[~homo]], ignore_index=True)
    view = view[view["anchor"].isin(focus_set)].copy()

    keep = ["anchor", "partner", "category", effect_col, "n_contrib",
            "frac_pos", "frac_neg", "sign_consistency",
            "n_enriched_samples", "n_depleted_samples", "q_stouffer"]
    view = view[keep]

    out = {}
    for a in map(str, focus_labels):
        sub = view[view["anchor"] == a]
        enr = (sub[sub["category"] == "shared_enriched"]
               .sort_values(effect_col, ascending=False).head(n_top).reset_index(drop=True))
        dep = (sub[sub["category"] == "shared_depleted"]
               .sort_values(effect_col, ascending=True).head(n_top).reset_index(drop=True))
        out[a] = {"enriched": enr, "depleted": dep}
        if verbose:
            n_e = int((sub["category"] == "shared_enriched").sum())
            n_d = int((sub["category"] == "shared_depleted").sum())
            print(f"\n=== {a} ===")
            print(f"TOP SHARED ENRICHED ({n_e} total):")
            if len(enr):
                for _, e in enr.iterrows():
                    print(f"{e['partner']:<26} log(x/mu)={e[effect_col]:+.3f}  "
                          f"n_contrib={int(e['n_contrib'])}  pos_consist={e['frac_pos']:.2f}  q={e['q_stouffer']:.1e}")
            else:
                print("No significantly enriched pairs.")
            print(f"TOP SHARED DEPLETED ({n_d} total):")
            if len(dep):
                for _, e in dep.iterrows():
                    print(f"{e['partner']:<26} log(x/mu)={e[effect_col]:+.3f}  "
                          f"n_contrib={int(e['n_contrib'])}  neg_consist={e['frac_neg']:.2f}  q={e['q_stouffer']:.1e}")
            else:
                print("No significantly depleted pairs.")
    return out, view

top, anchor_view = summarize_top_shared_pairs(
    cross_sample_df, focus_labels=FOCUS_LABELS, n_top=N_TOP
)

# SAVE
OUTDIR.mkdir(parents=True, exist_ok=True)
node_counts.to_csv(OUTDIR / "label_counts_per_patient.tsv", sep="\t")
pd.DataFrame([results_all[s]["testability"] for s in sample_order]).assign(
    seed=[BASE_SEED + si for si in range(len(sample_order))]
).to_csv(OUTDIR / "per_patient_testability.tsv", sep="\t", index=False)
cross_sample_df.to_csv(OUTDIR / "cross_sample_pair_enrichment.tsv", sep="\t", index=False)
anchor_view.to_csv(OUTDIR / "anchor_pair_view.tsv", sep="\t", index=False)
print(f"Outputs written to {OUTDIR.resolve()}")
