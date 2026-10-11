"""Reliability of the three classifiers' probabilities on the test split, from the stored test predictions
(model.test_predictions, written once by `testrun score`). The spec lists a reliability plot among the
classifier metrics; this draws it after the frozen run from the stored probabilities, without refitting or
re-predicting anything.

Every (case, series) probability is one point, pooled over series. Points are binned by predicted probability
into ten equal-width bins; each bin's mean prediction is plotted against the share of its points that are true
labels. Bins with fewer than MIN_BIN points are left off the plot (they're in the table). Expected calibration
error (ECE) is the point-weighted mean gap between the two.

    uv run python scripts/reliability.py      # reports/reliability_test.md and reports/reliability_test.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MODELS = {"baseline": "TF-IDF baseline", "finetune": "Fine-tuned", "zeroshot": "Zero-shot"}
BINS = 10
MIN_BIN = 20
# categorical slots 1-3 of the reference palette (light, dark); the first three pass all-pairs colorblind checks
COLORS = {"baseline": ("#2a78d6", "#3987e5"), "finetune": ("#eb6834", "#d95926"), "zeroshot": ("#1baf7a", "#199e70")}


def table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    classes = [c.split(":", 1)[1] for c in df.columns if c.startswith("baseline:")]
    y = np.array([[int(c in (s or "").split("|")) for c in classes] for s in df["labels"]]).ravel()
    rows, ece = [], {}
    for m in MODELS:
        p = df[[f"{m}:{c}" for c in classes]].to_numpy().ravel()
        b = np.minimum((p * BINS).astype(int), BINS - 1)
        gap = 0.0
        for k in range(BINS):
            sel = b == k
            if sel.any():
                rows.append({"model": m, "bin": f"{k / BINS:.1f}-{(k + 1) / BINS:.1f}", "points": int(sel.sum()),
                             "mean_predicted": p[sel].mean(), "observed": y[sel].mean()})
                gap += sel.sum() * abs(p[sel].mean() - y[sel].mean())
        ece[m] = gap / len(p)
    return pd.DataFrame(rows), ece


def svg(t: pd.DataFrame, ece: dict[str, float]) -> str:
    w, h, left, top, size = 640, 470, 64, 24, 380
    x = lambda v: left + v * size
    yy = lambda v: top + (1 - v) * size
    style = "".join(
        f".s-{m}{{stroke:{lc}}}.f-{m}{{fill:{lc}}}" for m, (lc, _) in COLORS.items()) + "".join(
        f"@media (prefers-color-scheme: dark){{.s-{m}{{stroke:{dc}}}.f-{m}{{fill:{dc}}}}}" for m, (_, dc) in COLORS.items())
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           'font-family="system-ui, sans-serif" font-size="12" role="img" '
           'aria-label="Reliability of the three classifiers on the test split">',
           "<style>.ring{stroke:#fcfcfb}.bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.muted{fill:#52514e}.grid{stroke:#e4e3df}.diag{stroke:#8a8984}"
           "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.ink{fill:#f0efec}.muted{fill:#c3c2b7}"
           ".grid{stroke:#3a3a37}.diag{stroke:#7d7c76}.ring{stroke:#1a1a19}}" + style + "</style>",
           f'<rect class="bg" width="{w}" height="{h}"/>']
    for v in np.linspace(0, 1, 6):
        out += [f'<line class="grid" x1="{x(0)}" x2="{x(1)}" y1="{yy(v)}" y2="{yy(v)}" stroke-width="1"/>',
                f'<line class="grid" x1="{x(v)}" x2="{x(v)}" y1="{yy(0)}" y2="{yy(1)}" stroke-width="1"/>',
                f'<text class="muted" x="{x(0) - 8}" y="{yy(v) + 4}" text-anchor="end">{v:.1f}</text>',
                f'<text class="muted" x="{x(v)}" y="{yy(0) + 18}" text-anchor="middle">{v:.1f}</text>']
    out += [f'<line class="diag" x1="{x(0)}" y1="{yy(0)}" x2="{x(1)}" y2="{yy(1)}" stroke-width="1.5" '
            'stroke-dasharray="4 4"/>',
            f'<line class="diag" x1="{x(1) + 16}" x2="{x(1) + 36}" y1="{top + 14 + 3 * 36}" '
            f'y2="{top + 14 + 3 * 36}" stroke-width="1.5" stroke-dasharray="4 4"/>',
            f'<text class="ink" x="{x(1) + 42}" y="{top + 18 + 3 * 36}">Perfect calibration</text>',
            f'<text class="ink" x="{x(0.5)}" y="{yy(0) + 40}" text-anchor="middle">Predicted probability (bin mean)</text>',
            f'<text class="ink" transform="translate({left - 44},{yy(0.5)}) rotate(-90)" text-anchor="middle">'
            'Share that are true labels</text>']
    for i, m in enumerate(MODELS):
        d = t[(t.model == m) & (t.points >= MIN_BIN)]
        pts = " ".join(f"{x(a):.1f},{yy(b):.1f}" for a, b in zip(d.mean_predicted, d.observed))
        out.append(f'<polyline class="s-{m}" points="{pts}" fill="none" stroke-width="2" stroke-linejoin="round"/>')
        for a, b, n in zip(d.mean_predicted, d.observed, d.points):
            out.append(f'<circle class="f-{m} ring" cx="{x(a):.1f}" cy="{yy(b):.1f}" r="4.5" '
                       f'stroke-width="2"><title>{MODELS[m]}: predicted {a:.2f}, observed {b:.2f}, {n} points'
                       '</title></circle>')
        ly = top + 14 + i * 36
        out += [f'<line class="s-{m}" x1="{x(1) + 16}" x2="{x(1) + 36}" y1="{ly}" y2="{ly}" stroke-width="2"/>',
                f'<circle class="f-{m}" cx="{x(1) + 26}" cy="{ly}" r="4.5"/>',
                f'<text class="ink" x="{x(1) + 42}" y="{ly + 4}">{MODELS[m]}</text>',
                f'<text class="muted" x="{x(1) + 42}" y="{ly + 18}" font-size="11">ECE {ece[m]:.3f}</text>']
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--out", default="reports/reliability_test")
    a = ap.parse_args()
    con = duckdb.connect(a.db, read_only=True)
    df = con.execute("SELECT * FROM model.test_predictions").df()
    con.close()
    t, ece = table(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".svg").write_text(svg(t, ece), encoding="utf-8")
    wide = t.pivot(index="bin", columns="model", values=["points", "mean_predicted", "observed"])
    wide.columns = [f"{m} {v.replace('_', ' ')}" for v, m in wide.columns]
    wide = wide[[f"{m} {v}" for m in MODELS for v in ("points", "mean predicted", "observed")]]
    lines = ["# Reliability, test split", "",
             f"Every (case, series) probability pooled over series, binned by predicted probability; descriptive, "
             f"drawn from the stored test predictions after the frozen run. Bins under {MIN_BIN} points are left "
             f"off the plot.", "", f"![Reliability](reliability_test.svg)", "",
             "| Model | ECE |", "| --- | --- |", *[f"| {MODELS[m]} | {e:.3f} |" for m, e in ece.items()], "",
             wide.round(3).to_markdown(), ""]
    Path(a.out + ".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
