"""Static report — the Pages artefact. Every figure on the page is computed by this run from the
two study files and embedded as JSON for the Executive Shell."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .data import DATA_SOURCE, Study, load
from .stats import (
    FOUR,
    Anova,
    Quartiles,
    Regression,
    Survival,
    anova_final_volume,
    logrank,
    mice_per_regimen,
    quartiles,
    sex_split,
    summary_by_regimen,
    survival,
    weight_volume_regression,
)

PKG = Path(__file__).parent
SHELL_DIR = PKG / "shell"
TEMPLATES = PKG / "templates"
COLOURS = ["#58A6FF", "#3fb950", "#d29922", "#f85149"]


@dataclass(frozen=True)
class Result:
    study: Study
    quartiles: list[Quartiles]
    regression: Regression
    anova: Anova
    survival: list[Survival]
    logrank_vs_placebo: list[tuple[str, float, float]]


def run(meta: Path, results: Path) -> Result:
    s = load(meta, results)
    return Result(
        study=s,
        quartiles=[quartiles(s.final, r) for r in FOUR],
        regression=weight_volume_regression(s.merged),
        anova=anova_final_volume(s.final),
        survival=[survival(s.final, r) for r in FOUR],
        logrank_vs_placebo=[(r, *logrank(s.final, r, "Placebo")) for r in FOUR],
    )


def _row(cells: list[str], head: bool = False) -> str:
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _p(v: float) -> str:
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def _summary_table(s: Study) -> str:
    df = summary_by_regimen(s.merged)
    rows = [_row(["Regimen", "Mean", "Median", "Variance", "SD", "SEM", "Observations"], head=True)]
    for reg, r in df.iterrows():
        rows.append(
            _row(
                [
                    html.escape(str(reg)),
                    f"{r['mean']:.2f}",
                    f"{r['median']:.2f}",
                    f"{r['var']:.2f}",
                    f"{r['std']:.2f}",
                    f"{r['sem']:.3f}",
                    str(int(r["n"])),
                ]
            )
        )
    return f"<table>{''.join(rows)}</table>"


def _quartile_table(qs: list[Quartiles]) -> str:
    rows = [_row(["Regimen", "Mice", "Q1", "Q3", "IQR", "Bounds", "Outliers"], head=True)]
    for q in qs:
        rows.append(
            _row(
                [
                    q.regimen,
                    str(q.n),
                    f"{q.q1:.2f}",
                    f"{q.q3:.2f}",
                    f"{q.iqr:.2f}",
                    f"{q.lower:.2f} to {q.upper:.2f}",
                    ", ".join(f"{o:.2f}" for o in q.outliers) or "none",
                ]
            )
        )
    return f"<table>{''.join(rows)}</table>"


def _tukey_table(a: Anova) -> str:
    rows = [_row(["Pair", "Mean difference", "Adjusted p", "Different at 5%"], head=True)]
    for x, y, d, p, rej in a.tukey:
        rows.append(_row([f"{x} vs {y}", f"{d:+.2f}", _p(p), "yes" if rej else "no"]))
    return f"<table>{''.join(rows)}</table>"


def _survival_table(sv: list[Survival], lr: list[tuple[str, float, float]]) -> str:
    rows = [
        _row(
            [
                "Regimen",
                "Mice",
                "Left before day 45",
                "Survival at last event",
                "Log-rank vs placebo (p)",
            ],
            head=True,
        )
    ]
    by = {r: (stat, p) for r, stat, p in lr}
    for s in sv:
        last = f"{s.surv[-1]:.3f}" if s.surv else "1.000"
        rows.append(_row([s.regimen, str(s.n), str(s.events), last, _p(by[s.regimen][1])]))
    return f"<table>{''.join(rows)}</table>"


def _el(tag: str, inner: str = "", **attrs: str | float) -> str:
    """One SVG element; numeric attributes are rounded to a decimal."""
    parts = [
        f'{k.replace("_", "-")}="{v:.1f}"'
        if isinstance(v, float)
        else f'{k.replace("_", "-")}="{v}"'
        for k, v in attrs.items()
    ]
    return f"<{tag} {' '.join(parts)}>{inner}</{tag}>"


def _tick(x: float, y: float, text: str, anchor: str = "start") -> str:
    return _el("text", html.escape(text), **{"class": "pm-tick"}, x=x, y=y, text_anchor=anchor)


def _axis(pad: int, width: int, height: int) -> str:
    return _el(
        "path",
        **{"class": "pm-axis"},
        d=f"M{pad},{pad} L{pad},{height - pad} L{width - pad},{height - pad}",
    )


def _svg(width: int, height: int, label: str, body: str) -> str:
    return _el(
        "svg",
        body,
        **{"class": "pm-chart"},
        viewBox=f"0 0 {width} {height}",
        role="img",
        aria_label=label,
    )


def _box_chart(qs: list[Quartiles], width: int = 640, height: int = 260) -> str:
    pad = 40
    values = [v for q in qs for v in (q.lower, q.upper, *q.outliers)]
    lo, hi = min(20.0, *values), max(80.0, *values)

    def sy(v: float) -> float:
        return height - pad - (v - lo) / (hi - lo) * (height - 2 * pad)

    slot = (width - 2 * pad) / len(qs)
    parts = []
    for i, q in enumerate(qs):
        cx = pad + slot * (i + 0.5)
        colour = COLOURS[i % len(COLOURS)]
        parts.append(
            _el(
                "line",
                x1=cx,
                x2=cx,
                y1=sy(q.upper),
                y2=sy(q.lower),
                stroke=colour,
                stroke_width=1.5,
            )
        )
        title = _el("title", f"{q.regimen}: Q1 {q.q1:.2f}, Q3 {q.q3:.2f}")
        parts.append(
            _el(
                "rect",
                title,
                x=cx - 22,
                y=sy(q.q3),
                width=44,
                height=sy(q.q1) - sy(q.q3),
                fill=colour,
                fill_opacity=0.35,
                stroke=colour,
            )
        )
        for o in q.outliers:
            parts.append(
                _el(
                    "circle",
                    _el("title", f"{q.regimen} outlier {o:.2f}"),
                    cx=cx,
                    cy=sy(o),
                    r=4,
                    fill=colour,
                )
            )
        parts.append(_tick(cx, height - pad + 14, q.regimen, "middle"))
    ticks = _tick(pad - 4, pad + 4, f"{hi:.0f}", "end") + _tick(
        pad - 4, height - pad + 4, f"{lo:.0f}", "end"
    )
    label = "Final tumour volume by regimen: interquartile boxes, 1.5 IQR whiskers and outliers"
    return _svg(width, height, label, _axis(pad, width, height) + "".join(parts) + ticks)


def _km_chart(sv: list[Survival], width: int = 640, height: int = 240) -> str:
    pad = 40

    def sx(t: float) -> float:
        return pad + t / 45.0 * (width - 2 * pad)

    def sy(p: float) -> float:
        return height - pad - p * (height - 2 * pad)

    lines = []
    for i, s in enumerate(sv):
        pts = [(0.0, 1.0)]
        for t, p in zip(s.times, s.surv, strict=True):
            pts.append((t, pts[-1][1]))
            pts.append((t, p))
        pts.append((45.0, pts[-1][1]))
        d = " ".join(f"{sx(t):.1f},{sy(p):.1f}" for t, p in pts)
        colour = COLOURS[i % len(COLOURS)]
        lines.append(
            _el(
                "polyline",
                _el("title", s.regimen),
                fill="none",
                stroke=colour,
                stroke_width=2,
                points=d,
            )
        )
    legend = "".join(
        _el("tspan", f"{s.regimen}  ", fill=COLOURS[i % len(COLOURS)]) for i, s in enumerate(sv)
    )
    ticks = (
        _tick(pad - 4, pad + 4, "1.0", "end")
        + _tick(pad - 4, height - pad + 4, "0.0", "end")
        + _tick(width - pad, height - pad + 14, "Day 45", "end")
        + _el("text", legend, **{"class": "pm-tick"}, x=width - pad, y=pad - 8, text_anchor="end")
    )
    return _svg(
        width,
        height,
        "Kaplan-Meier time in study by regimen",
        _axis(pad, width, height) + "".join(lines) + ticks,
    )


def kpi_json(r: Result) -> dict[str, object]:
    s = r.study
    per = mice_per_regimen(s.merged)
    summary = summary_by_regimen(s.merged)
    return {
        "mice": int(s.merged["Mouse ID"].nunique()),
        "rows": len(s.merged),
        "dropped": s.dropped_mice,
        "regimens": int(per.size),
        "lowestMeanRegimen": str(summary.index[0]),
        "lowestMean": float(summary["mean"].iloc[0]),
        "anovaF": r.anova.f,
        "anovaP": r.anova.p_value,
        "regressionR": r.regression.r,
        "regressionP": r.regression.p_value,
        "outliers": {q.regimen: q.outliers for q in r.quartiles},
        "survival": {s_.regimen: {"n": s_.n, "events": s_.events} for s_ in r.survival},
        "sex": {str(k): int(v) for k, v in sex_split(s.merged).items()},
        "source": DATA_SOURCE,
    }


def render_html(r: Result, pages: str) -> str:
    s = r.study
    tpl = Template((TEMPLATES / "page.html").read_text(encoding="utf-8"))
    summary = summary_by_regimen(s.merged)
    reg = r.regression
    outl = [(q.regimen, o) for q in r.quartiles for o in q.outliers]
    return tpl.substitute(
        pages=html.escape(pages),
        source=html.escape(DATA_SOURCE),
        mice=str(s.merged["Mouse ID"].nunique()),
        rows=f"{len(s.merged):,}",
        dropped=", ".join(s.dropped_mice) or "none",
        regimens=str(mice_per_regimen(s.merged).size),
        lowest=html.escape(str(summary.index[0])),
        lowest_mean=f"{summary['mean'].iloc[0]:.2f}",
        highest=html.escape(str(summary.index[-1])),
        highest_mean=f"{summary['mean'].iloc[-1]:.2f}",
        summary_table=_summary_table(s),
        quartile_table=_quartile_table(r.quartiles),
        box_chart=_box_chart(r.quartiles),
        outliers=("; ".join(f"{reg_}: {o:.2f} mm³" for reg_, o in outl) or "none"),
        anova_f=f"{r.anova.f:.1f}",
        anova_p=_p(r.anova.p_value),
        tukey=_tukey_table(r.anova),
        reg_r=f"{reg.r:+.3f}",
        reg_r2=f"{reg.r_squared:.4f}",
        reg_p=_p(reg.p_value),
        reg_n=str(reg.n),
        reg_slope=f"{reg.slope:+.3f}",
        survival_table=_survival_table(r.survival, r.logrank_vs_placebo),
        km_chart=_km_chart(r.survival),
        report_json=json.dumps(kpi_json(r)),
    )


def write_report(
    out: Path,
    meta: Path,
    results: Path,
    pages: str = "https://freddricklogan.github.io/MedicalTreatment-Analytics/",
) -> Path:
    r = run(meta, results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    shutil.copy(SHELL_DIR / "exec-shell.css", out / "src" / "exec-shell.css")
    shutil.copy(SHELL_DIR / "exec-shell.js", out / "src" / "exec-shell.js")
    shutil.copy(PKG / "report.js", out / "src" / "report.js")
    shutil.copy(TEMPLATES / "report.css", out / "src" / "report.css")
    (out / "index.html").write_text(render_html(r, pages), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(kpi_json(r), indent=2), encoding="utf-8")
    return out / "index.html"
