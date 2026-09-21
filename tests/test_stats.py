from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymaceuticals.data import load
from pymaceuticals.stats import (
    FOUR,
    anova_final_volume,
    logrank,
    mice_per_regimen,
    quartiles,
    sex_split,
    summary_by_regimen,
    survival,
    weight_volume_regression,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def study():  # type: ignore[no-untyped-def]
    return load(DATA / "Mouse_metadata.csv", DATA / "Study_results.csv")


def test_load_and_clean(study) -> None:  # type: ignore[no-untyped-def]
    assert study.dropped_mice == ["g989"]
    assert study.merged["Mouse ID"].nunique() == 249
    assert len(study.merged) == 1709
    assert not study.merged.duplicated(subset=["Mouse ID", "Timepoint"]).any()
    assert len(study.final) == 249
    assert (study.final.groupby("Mouse ID").size() == 1).all()


def test_load_rejections(tmp_path: Path) -> None:
    meta = tmp_path / "m.csv"
    res = tmp_path / "r.csv"
    meta.write_text("Mouse ID,Drug Regimen\na,X\n")
    res.write_text("Mouse ID,Timepoint,Tumor Volume (mm3),Metastatic Sites\na,0,45,0\n")
    with pytest.raises(ValueError, match="metadata is missing"):
        load(meta, res)
    meta.write_text("Mouse ID,Drug Regimen,Sex,Age_months,Weight (g)\na,X,F,3,20\na,X,F,3,20\n")
    with pytest.raises(ValueError, match="duplicate Mouse ID"):
        load(meta, res)
    meta.write_text("Mouse ID,Drug Regimen,Sex,Age_months,Weight (g)\nb,X,F,3,20\n")
    with pytest.raises(ValueError, match="absent from metadata"):
        load(meta, res)


def test_summary_and_counts(study) -> None:  # type: ignore[no-untyped-def]
    s = summary_by_regimen(study.merged)
    assert list(s.index[:2]) == ["Capomulin", "Ramicane"]
    cap = study.merged.loc[study.merged["Drug Regimen"] == "Capomulin", "Tumor Volume (mm3)"]
    assert s.loc["Capomulin", "mean"] == pytest.approx(cap.mean())
    assert s.loc["Capomulin", "sem"] == pytest.approx(cap.std() / np.sqrt(len(cap)))
    per = mice_per_regimen(study.merged)
    assert per.sum() == 249 and per["Capomulin"] == 24
    assert sex_split(study.merged).to_dict() == {"Female": 128, "Male": 121}


def test_quartiles_match_numpy_and_flag_the_ceftamin_outlier(study) -> None:  # type: ignore[no-untyped-def]
    q = quartiles(study.final, "Ceftamin")
    v = study.final.loc[study.final["Drug Regimen"] == "Ceftamin", "Tumor Volume (mm3)"]
    assert q.q1 == pytest.approx(np.percentile(v, 25)) and q.q3 == pytest.approx(
        np.percentile(v, 75)
    )
    assert q.outliers == [pytest.approx(71.820674)]
    assert quartiles(study.final, "Capomulin").outliers == []
    with pytest.raises(ValueError, match="no mice"):
        quartiles(study.final, "Nope")


def test_regression_reports_no_relationship_in_these_files(study) -> None:  # type: ignore[no-untyped-def]
    r = weight_volume_regression(study.merged)
    assert r.n == 24
    assert abs(r.r) < 0.1 and r.p_value > 0.5
    assert r.r_squared == pytest.approx(r.r**2)
    tiny = study.merged[study.merged["Mouse ID"].isin(["a195", "a441"])]
    with pytest.raises(ValueError, match="at least 3"):
        weight_volume_regression(tiny)


def test_regression_recovers_a_known_line() -> None:
    rows = [
        {
            "Mouse ID": f"m{i}",
            "Drug Regimen": "X",
            "Weight (g)": float(i),
            "Tumor Volume (mm3)": 2.0 * i + 1,
        }
        for i in range(10)
    ]
    r = weight_volume_regression(pd.DataFrame(rows), "X")
    assert (
        r.slope == pytest.approx(2.0)
        and r.intercept == pytest.approx(1.0)
        and r.r == pytest.approx(1.0)
    )


def test_anova_and_tukey(study) -> None:  # type: ignore[no-untyped-def]
    a = anova_final_volume(study.final)
    assert a.groups == list(FOUR) and len(a.tukey) == 6
    assert a.f > 50 and a.p_value < 1e-10
    pairs = {frozenset((x, y)): rej for x, y, _, _, rej in a.tukey}
    assert pairs[frozenset(("Capomulin", "Ramicane"))] is False
    assert pairs[frozenset(("Capomulin", "Ceftamin"))] is True
    with pytest.raises(ValueError, match="at least 2"):
        anova_final_volume(study.final.head(3), ("Capomulin", "Ramicane"))


def test_survival_and_logrank(study) -> None:  # type: ignore[no-untyped-def]
    s = survival(study.final, "Capomulin")
    assert s.n == 24 and s.events == 4
    assert s.surv == sorted(s.surv, reverse=True) and all(0 < v <= 1 for v in s.surv)
    assert s.surv[-1] == pytest.approx(
        20 / 24, abs=1e-9
    )  # four events among 24, no censoring before them
    stat, p = logrank(study.final, "Capomulin", "Infubinol")
    assert stat > 0 and 0 <= p <= 1
    with pytest.raises(ValueError, match="no mice"):
        survival(study.final, "Nope")
