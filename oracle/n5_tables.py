#!/usr/bin/env python3
"""n5_tables.py — the ×2 bench tables (N5) from `vosrgate score` output: mean SSIMULACRA2 FR (vs the lossless 1024²
reference), mean gradient (detail), and the FidelityGuard worst-tile downsample consistency, per regime × class × arm,
plus per-image FR on the damaged (jpeg) regime and win counts for the route decision.

    python3 oracle/n5_tables.py oracle/reports/n5-x2-score.csv
"""
import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
ARMS = ["NERVE-fid4x-down", "NERVE-gan2x", "NERVE-rel2x", "RealESRGAN-4x-down", "bicubic-cg", "lanczos-mps", "REFERENCE"]


def f(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


agg = defaultdict(list)
for r in rows:
    agg[(r["regime"], r["class"], r["method"])].append(r)


def mean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


for metric, label in (("fr", "mean SSIMULACRA2 FR (higher = closer to the reference)"),
                      ("grad", "mean gradient (detail; reference in the REFERENCE column)"),
                      ("dc_worst", "FidelityGuard DC worst 64 px tile (higher = less invented)")):
    print(f"\n### {label}\n")
    print("| regime / class | " + " | ".join(ARMS) + " |")
    print("|---|" + "---:|" * len(ARMS))
    for regime in ("bicubic", "lanczos", "jpeg"):
        for cls in ("photo", "graphic"):
            cells = []
            for a in ARMS:
                v = mean([f(r[metric]) for r in agg[(regime, cls, a)]])
                cells.append("—" if v != v else f"{v:.2f}" if metric != "dc_worst" else f"{v:.4f}")
            print(f"| {regime} {cls} ({len(agg[(regime, cls, ARMS[0])])}) | " + " | ".join(cells) + " |")

print("\n### jpeg regime, per image FR\n")
print("| image | " + " | ".join(ARMS[:-1]) + " |")
print("|---|" + "---:|" * (len(ARMS) - 1))
images = sorted({r["image"] for r in rows})
for img in images:
    cells = []
    for a in ARMS[:-1]:
        v = [f(r["fr"]) for r in rows if r["image"] == img and r["regime"] == "jpeg" and r["method"] == a]
        cells.append(f"{v[0]:.2f}" if v else "—")
    print(f"| {img} | " + " | ".join(cells) + " |")

print("\n### route decision: `.fidelity` ×4 + downsample vs native 2x_NERVE_OTF_gan, per cell\n")
wins = defaultdict(int)
diffs = defaultdict(list)
for img in images:
    for regime in ("bicubic", "lanczos", "jpeg"):
        a = [f(r["fr"]) for r in rows if r["image"] == img and r["regime"] == regime and r["method"] == "NERVE-fid4x-down"]
        b = [f(r["fr"]) for r in rows if r["image"] == img and r["regime"] == regime and r["method"] == "NERVE-gan2x"]
        if a and b:
            wins[(regime, "fid4x-down" if a[0] > b[0] else "gan2x")] += 1
            diffs[regime].append(a[0] - b[0])
for regime in ("bicubic", "lanczos", "jpeg"):
    print(f"- {regime}: fid4x-down wins {wins[(regime, 'fid4x-down')]}/9, gan2x wins {wins[(regime, 'gan2x')]}/9; "
          f"mean FR(fid4x-down) − FR(gan2x) = {mean(diffs[regime]):+.2f}")
alld = [d for v in diffs.values() for d in v]
print(f"- all 27 cells: fid4x-down wins {sum(1 for d in alld if d > 0)}/27, mean Δ {mean(alld):+.2f}")
