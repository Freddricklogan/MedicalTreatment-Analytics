import json
from pathlib import Path

from pymaceuticals.report import write_report

DATA = Path(__file__).resolve().parents[1] / "data"


def test_write_report(tmp_path: Path) -> None:
    out = write_report(tmp_path / "dist", DATA / "Mouse_metadata.csv", DATA / "Study_results.csv")
    assert out.exists()
    page = out.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in page and "onclick" not in page and "style=" not in page
    assert "$" not in page.replace("$pages", "")
    data = json.loads((tmp_path / "dist" / "report.json").read_text())
    assert data["mice"] == 249 and data["dropped"] == ["g989"]
    assert data["outliers"]["Ceftamin"] and not data["outliers"]["Capomulin"]
    assert data["survival"]["Capomulin"]["events"] == 4
