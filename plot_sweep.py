#!/usr/bin/env python3
"""Recreate aie_pl/runs/latency_vs_crossings.png from sweep_results.csv."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "aie_pl", "runs", "sweep_results.csv")
PLOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "aie_pl", "runs", "latency_vs_crossings.png")

rows = [r for r in csv.DictReader(open(CSV_PATH)) if r["status"] == "pass"]
xs        = [int(r["crossings"])    for r in rows]
total_us  = [float(r["total_ns"]) / 1e3 for r in rows]
aie_us    = [float(r["aie_ns"])   / 1e3 for r in rows]
pl_us     = [float(r["pl_ns"])    / 1e3 for r in rows]

# Linear fit to total latency (in μs)
m, b  = np.polyfit(xs, total_us, 1)
x_fit = np.linspace(min(xs), max(xs), 200)
y_fit = m * x_fit + b

ss_res = sum((y - (m*x + b))**2 for x, y in zip(xs, total_us))
ss_tot = sum((y - np.mean(total_us))**2 for y in total_us)
r2 = 1 - ss_res / ss_tot

fig, ax = plt.subplots(figsize=(9, 5))

# Linear fit line drawn first so it sits behind the data lines
ax.plot(x_fit, y_fit, "--", color="red", lw=1.5, zorder=1, label="_nolegend_")

ax.plot(xs, total_us, "o-",  color="C0", lw=2.5, ms=8, zorder=3, label="Total Latency [AIE+PL]")
ax.plot(xs, aie_us,   "s-",  color="C1", lw=1.5, ms=6, zorder=2, label="Latency of AIE layers")
ax.plot(xs, pl_us,    "^-",  color="C2", lw=1.5, ms=6, zorder=2, label="Latency of PL layers")

# Annotation in the empty space above the total line (upper-left region)
label = f"linear fit: $R^2$ = {r2:.4f}\npenalty per crossing = {m*1e3:.0f} ns/crossing"
ax.text(5.3, max(total_us) - 0.03, label,
        color="red", fontsize=12,
        va="top", ha="left")

ax.set_xlabel("Number of PL↔AIE Crossings", fontsize=12)
ax.set_ylabel("Latency (μs)", fontsize=12)
ax.set_title("Latency Penalty of Crossing the PL-AIE Boundary", fontsize=12)
ax.set_xticks(xs)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"Saved: {PLOT_PATH}")
print(f"Fit:   {m*1e3:.1f} ns per crossing,  intercept={b*1e3:.1f} ns,  R²={r2:.4f}")
