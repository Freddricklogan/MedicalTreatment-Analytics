"""Loading and cleaning the two study files. The merge, the duplicate rule and the final-timepoint
selection are the decisions everything downstream depends on, so each is a small tested function."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

META_COLS = ("Mouse ID", "Drug Regimen", "Sex", "Age_months", "Weight (g)")
RESULT_COLS = ("Mouse ID", "Timepoint", "Tumor Volume (mm3)", "Metastatic Sites")
DATA_SOURCE = (
    "Pymaceuticals mouse study as distributed with the Matplotlib data-analysis exercise: 249 mice "
    "across ten regimens, tumour volume recorded every five days to day 45; vendored in data/"
)


@dataclass(frozen=True)
class Study:
    merged: pd.DataFrame  # one row per mouse per timepoint
    dropped_mice: list[str]  # mice removed for duplicated timepoints
    final: pd.DataFrame  # one row per mouse: its last recorded timepoint


def _check(df: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"{name} is missing columns: {', '.join(missing)}"
        raise ValueError(msg)


def load(meta_path: Path, results_path: Path) -> Study:
    meta = pd.read_csv(meta_path)
    results = pd.read_csv(results_path)
    _check(meta, META_COLS, "metadata")
    _check(results, RESULT_COLS, "results")
    if meta["Mouse ID"].duplicated().any():
        msg = "metadata has duplicate Mouse ID rows"
        raise ValueError(msg)
    merged = results.merge(meta, on="Mouse ID", how="left", validate="many_to_one")
    if merged["Drug Regimen"].isna().any():
        msg = "results reference mice absent from metadata"
        raise ValueError(msg)
    dup = merged[merged.duplicated(subset=["Mouse ID", "Timepoint"], keep=False)]
    dropped = sorted(dup["Mouse ID"].unique().tolist())
    clean = merged[~merged["Mouse ID"].isin(dropped)].reset_index(drop=True)
    last = clean.groupby("Mouse ID")["Timepoint"].max().reset_index()
    final = clean.merge(last, on=["Mouse ID", "Timepoint"], how="inner").reset_index(drop=True)
    return Study(merged=clean, dropped_mice=[str(m) for m in dropped], final=final)
