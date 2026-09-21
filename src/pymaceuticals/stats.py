"""Statistics over the cleaned study: per-regimen summary, IQR outliers, regression, one-way ANOVA
with Tukey HSD, and Kaplan-Meier survival with a log-rank test. Every number the report prints
comes from a function here, and each has a test against a hand-computed or library value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.stats.multicomp import pairwise_tukeyhsd

VOL = "Tumor Volume (mm3)"
FOUR = ("Capomulin", "Ramicane", "Infubinol", "Ceftamin")


def summary_by_regimen(merged: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, variance, standard deviation and SEM of tumour volume per regimen."""
    g = merged.groupby("Drug Regimen")[VOL]
    out = pd.DataFrame(
        {
            "mean": g.mean(),
            "median": g.median(),
            "var": g.var(),
            "std": g.std(),
            "sem": g.sem(),
            "n": g.size(),
        }
    )
    return out.sort_values("mean")


@dataclass(frozen=True)
class Quartiles:
    regimen: str
    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float
    outliers: list[float]
    n: int


def quartiles(final: pd.DataFrame, regimen: str) -> Quartiles:
    v = final.loc[final["Drug Regimen"] == regimen, VOL].to_numpy(dtype=float)
    if v.size == 0:
        msg = f"no mice for regimen {regimen!r}"
        raise ValueError(msg)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return Quartiles(
        regimen,
        float(q1),
        float(q3),
        float(iqr),
        float(lower),
        float(upper),
        sorted(float(x) for x in v if x < lower or x > upper),
        int(v.size),
    )


@dataclass(frozen=True)
class Regression:
    slope: float
    intercept: float
    r: float
    r_squared: float
    p_value: float
    n: int


def weight_volume_regression(merged: pd.DataFrame, regimen: str = "Capomulin") -> Regression:
    """Average tumour volume per mouse against its weight, within one regimen."""
    sub = merged[merged["Drug Regimen"] == regimen]
    per = sub.groupby("Mouse ID").agg(weight=("Weight (g)", "first"), volume=(VOL, "mean"))
    if len(per) < 3:
        msg = "regression needs at least 3 mice"
        raise ValueError(msg)
    res = sps.linregress(per["weight"], per["volume"])
    return Regression(
        float(res.slope),
        float(res.intercept),
        float(res.rvalue),
        float(res.rvalue**2),
        float(res.pvalue),
        len(per),
    )


@dataclass(frozen=True)
class Anova:
    f: float
    p_value: float
    groups: list[str]
    tukey: list[tuple[str, str, float, float, bool]]  # a, b, mean difference, adjusted p, reject


def anova_final_volume(final: pd.DataFrame, regimens: tuple[str, ...] = FOUR) -> Anova:
    """One-way ANOVA on final tumour volume across regimens, then Tukey HSD pairwise comparisons."""
    sub = final[final["Drug Regimen"].isin(regimens)]
    groups = [sub.loc[sub["Drug Regimen"] == r, VOL].to_numpy(dtype=float) for r in regimens]
    if any(g.size < 2 for g in groups):
        msg = "every regimen needs at least 2 mice"
        raise ValueError(msg)
    f, p = sps.f_oneway(*groups)
    tk = pairwise_tukeyhsd(
        sub[VOL].to_numpy(dtype=float), sub["Drug Regimen"].to_numpy(), alpha=0.05
    )
    rows = [
        (str(a), str(b), float(d), float(pv), bool(rej))
        for a, b, d, pv, rej in zip(
            tk.groupsunique[tk._multicomp.pairindices[0]],
            tk.groupsunique[tk._multicomp.pairindices[1]],
            tk.meandiffs,
            tk.pvalues,
            tk.reject,
            strict=True,
        )
    ]
    return Anova(float(f), float(p), list(regimens), rows)


@dataclass(frozen=True)
class Survival:
    regimen: str
    times: list[float]
    surv: list[float]  # Kaplan-Meier estimate at each time
    n: int
    events: int


def survival(final: pd.DataFrame, regimen: str, horizon: float = 45.0) -> Survival:
    """Kaplan-Meier over each mouse's last timepoint: an event is a record that ends before the
    study horizon (the mouse left the study); mice observed at the horizon are censored."""
    sub = final[final["Drug Regimen"] == regimen]
    t = sub["Timepoint"].to_numpy(dtype=float)
    event = (t < horizon).astype(int)
    if t.size == 0:
        msg = f"no mice for regimen {regimen!r}"
        raise ValueError(msg)
    sf = SurvfuncRight(t, event)
    return Survival(
        regimen,
        [float(x) for x in sf.surv_times],
        [float(x) for x in sf.surv_prob],
        int(t.size),
        int(event.sum()),
    )


def logrank(final: pd.DataFrame, a: str, b: str, horizon: float = 45.0) -> tuple[float, float]:
    """Log-rank test statistic and p-value between two regimens' time-in-study curves."""
    sub = final[final["Drug Regimen"].isin([a, b])]
    t = sub["Timepoint"].to_numpy(dtype=float)
    event = (t < horizon).astype(int)
    group = sub["Drug Regimen"].to_numpy()
    stat, p = survdiff(t, event, group)
    return float(stat), float(p)


def mice_per_regimen(merged: pd.DataFrame) -> pd.Series:
    return merged.groupby("Drug Regimen")["Mouse ID"].nunique().sort_values(ascending=False)


def sex_split(merged: pd.DataFrame) -> pd.Series:
    return merged.drop_duplicates("Mouse ID")["Sex"].value_counts()
