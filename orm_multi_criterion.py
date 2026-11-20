# ORM MULTI-CRITERION
# Author(s): Lauren Kim, Jennifer Le (UCSF)
# Multi-criterion ORM-based filtering (Figure 3) for Kim, Le et al. 2026

import numpy as np
import pandas as pd
from pathlib import Path

# FUNCTIONS
def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "t": True,
            "1": True,
            "false": False,
            "f": False,
            "0": False,
            "nan": False,
            "na": False,
            "none": False,
            "": False,
        })
        .fillna(False)
        .astype(bool)
    )

def read_patient_orm_file(patient_dir: Path) -> pd.DataFrame:
    patient_id = patient_dir.name.replace("_ORM", "")

    matches = list(patient_dir.glob("*_overlap_metrics_all_celltypes.tsv.gz")) # all patients
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Error: expected exactly one all-celltypes ORM TSV.GZ in {patient_dir}."
        )

    df = pd.read_csv(matches[0], sep="\t")

    if PATIENT_COL not in df.columns:
        df[PATIENT_COL] = patient_id
    return df

def summarize_group(g: pd.DataFrame) -> pd.Series:
    evaluable = g[g["orm_evaluable"]].copy()
    finite = evaluable[np.isfinite(evaluable["ratio"])].copy()

    n_evaluable = evaluable[PATIENT_COL].nunique()
    n_pass = int(evaluable["orm_pass_clean"].sum())

    if n_evaluable > 0:
        pass_rate = n_pass / n_evaluable
    else:
        pass_rate = np.nan

    if len(finite) > 0:
        median_ratio = finite["ratio"].median()
        q75_ratio = finite["ratio"].quantile(0.75)
    else:
        median_ratio = np.nan
        q75_ratio = np.nan

    return pd.Series({
        "n_patients_evaluable": n_evaluable,
        "n_pass": n_pass,
        "pass_rate": pass_rate,
        "n_finite": int(np.isfinite(evaluable["ratio"]).sum()),
        "n_inf": int(np.isposinf(evaluable["ratio"]).sum()),
        "n_nan_or_na": int((~g["orm_evaluable"]).sum()),
        "median_ratio": median_ratio,
        "q75_ratio": q75_ratio,
    })

# CONFIGURATION
ORM_ROOT = Path("062026_SMIDE_ORM_BY_PATIENT_CELLTYPE_UPDATED")
OUTDIR = ORM_ROOT / "CELLTYPE_ORM_FILTERS"

# OBS
CELLTYPE_COL = "celltype_1"
GENE_COL = "target"
PATIENT_COL = "patient"

ORM_CUTOFF = 1                   
MIN_EVALUABLE_PATIENTS = 5
STRICT_MAJORITY_PASS_RATE = 0.625 # for 8 patients
MULTI_PASS_RATE = 0.5
MEDIAN_RATIO_MAX = 1.0 
TAU_PRIMARY = 1.5 # primary
TAU_SENSITIVITY = 2.0 # sensitivity

# COMBINE
patient_dirs = sorted([
    p for p in ORM_ROOT.iterdir()
    if p.is_dir() and p.name.endswith("_ORM")
])

if len(patient_dirs) == 0:
    raise FileNotFoundError(f"Error: No patient directories ending in '_ORM' found in {ORM_ROOT}.")

all_orm = pd.concat(
    [read_patient_orm_file(p) for p in patient_dirs],
    ignore_index=True
)

required_cols = {
    PATIENT_COL,
    CELLTYPE_COL,
    GENE_COL,
    "ratio",
    "avg_neighbor_othercluster",
    "avg_cluster",
}

missing = required_cols - set(all_orm.columns)
if missing:
    raise ValueError(f"Combined ORM table is missing required columns: {missing}.")

for col in ["ratio", "avg_neighbor_othercluster", "avg_cluster"]:
    all_orm[col] = pd.to_numeric(all_orm[col], errors="coerce")

all_orm["orm_evaluable"] = np.isfinite(all_orm["ratio"]) | np.isposinf(all_orm["ratio"])
all_orm["orm_pass_clean"] = np.isfinite(all_orm["ratio"]) & (all_orm["ratio"] <= ORM_CUTOFF)

if "orm_pass" in all_orm.columns:
    all_orm["orm_pass_from_R"] = parse_bool_series(all_orm["orm_pass"])
# QC
dup = (
    all_orm
    .groupby([PATIENT_COL, CELLTYPE_COL, GENE_COL], dropna=False)
    .size()
    .reset_index(name="n")
    .query("n > 1")
)

if len(dup) > 0:
    raise ValueError(
        "Duplicate rows detected."
    )

print(f"Loaded ORM rows: {all_orm.shape[0]:,}.")
print(f"List of patients: {all_orm[PATIENT_COL].nunique()}.")
print(f"Cell types: {all_orm[CELLTYPE_COL].nunique()}.")
print(f"Genes: {all_orm[GENE_COL].nunique()}.")

# SAVE
OUTDIR.mkdir(parents=True, exist_ok=True)
all_orm.to_csv(OUTDIR / "all_patients_all_celltypes_ORM_combined.tsv.gz",
               sep="\t", index=False, compression="gzip")

gene_summary = (
    all_orm
    .groupby([CELLTYPE_COL, GENE_COL], dropna=False)
    .apply(summarize_group)
    .reset_index()
)

gene_summary["enough_patients"] = (
    gene_summary["n_patients_evaluable"] >= MIN_EVALUABLE_PATIENTS
)

# CRITERIA 1: STRICT MAJORITY
gene_summary["retain_strict_majority"] = (
    gene_summary["enough_patients"]
    & (gene_summary["pass_rate"] >= STRICT_MAJORITY_PASS_RATE)
)

# CRITERIA 2 & 3: PRIMARY CRITERION
gene_summary["retain_multi_tau1p5"] = (
    gene_summary["enough_patients"]
    & (gene_summary["pass_rate"] >= MULTI_PASS_RATE)
    & (gene_summary["median_ratio"] <= MEDIAN_RATIO_MAX)
    & (gene_summary["q75_ratio"] <= TAU_PRIMARY)
)

# SENSITIVITY
gene_summary["retain_multi_tau2p0"] = (
    gene_summary["enough_patients"]
    & (gene_summary["pass_rate"] >= MULTI_PASS_RATE)
    & (gene_summary["median_ratio"] <= MEDIAN_RATIO_MAX)
    & (gene_summary["q75_ratio"] <= TAU_SENSITIVITY)
)

# HIGH CONFIDENCE
gene_summary["retain_high_confidence"] = (
    gene_summary["retain_strict_majority"]
    & gene_summary["retain_multi_tau1p5"]
)

# SAVE
gene_summary.to_csv(
    OUTDIR / "cohort_ORM_gene_summary_filters.tsv",
    sep="\t",
    index=False
)

retain_cols = ["retain_strict_majority", "retain_multi_tau1p5", "retain_multi_tau2p0", "retain_high_confidence"]
filter_counts = gene_summary.groupby(CELLTYPE_COL)[retain_cols].sum()
filter_counts.loc["ALL"] = filter_counts.sum()
filter_counts.to_csv(OUTDIR / "retained_gene_counts_by_celltype.tsv", sep="\t")
print(filter_counts.to_string())
print(f"All outputs written to {OUTDIR.resolve()}.")