#!/usr/bin/env python3
"""
结果分析与作图：复现 Figure_9_10.R 与 Figure_A5_A6.R 的图。
无需 R，用 Python + pandas + matplotlib 生成。
输出：actual_predicted.png, gain_hist.png, Gamma.png, tau.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- Figure 9 & 10 (from figure9.csv) ----------
print("Loading figure9.csv ...")
df = pd.read_csv("figure9.csv")
if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
    df = df.iloc[:, 1:]
G1 = df["G1"].values
G2 = df["G2"].values
# 数据量大时子采样以加快 2D 图
if len(G1) > 50000:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(G1), 50000, replace=False)
    G1_plot, G2_plot = G1[idx], G2[idx]
else:
    G1_plot, G2_plot = G1, G2

# 1) Predicted vs Actual (2D density style)
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.scatter(G1_plot, G2_plot, alpha=0.1, s=5, c="purple")
h = ax.hist2d(G1_plot, G2_plot, bins=25, cmap="Purples", cmin=1)
plt.colorbar(h[3], ax=ax)
ax.set_xlabel("Predicted Gain (Measured by Doubly Robust)")
ax.set_ylabel("Actual Gain (Measured by Field Experiment)")
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 4)
ax.set_facecolor("white")
for spine in ax.spines.values():
    spine.set_color("black")
ax.grid(True, axis="y", linestyle="-", linewidth=0.1, color="grey")
ax.grid(True, axis="x", linestyle="-", linewidth=0.1, color="grey")
plt.tight_layout()
plt.savefig("actual_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved actual_predicted.png")

# 2) Histogram of G2
fig, ax = plt.subplots(figsize=(3.9, 3))
ax.hist(G2, bins=50, color="purple", edgecolor="purple", linewidth=0.5)
ax.set_xlabel("CLV Gain (Measured by Field Experiment)")
ax.set_facecolor("white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="-", linewidth=0.1, color="grey")
plt.tight_layout()
plt.savefig("gain_hist.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved gain_hist.png")

# ---------- Figure A5 & A6 (from Gamma_tau.csv) ----------
print("Loading Gamma_tau.csv ...")
data = pd.read_csv("Gamma_tau.csv")
time_col = [c for c in data.columns if "Time" in c or "time" in c.lower()][0]
x = data[time_col].values

# 3) Gamma
gamma_cols = [c for c in data.columns if c.startswith("Gamma_")]
colors = ["indianred", "orange", "royalblue", "darkgreen"]
linestyles = ["--", ":", "-", "-."]
fig, ax = plt.subplots(figsize=(5, 3.5))
for i, col in enumerate(gamma_cols):
    ax.plot(x, data[col].values, label=col, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
ax.set_xlim(0, 10)
ax.set_xticks(range(1, 11))
ax.set_ylabel("Average Return")
ax.set_xlabel("Time Steps (1e6)")
ax.legend(loc="lower right", frameon=False)
ax.set_facecolor("white")
ax.yaxis.grid(True, linestyle="-", linewidth=0.1, color="black")
ax.xaxis.grid(False)
for spine in ax.spines.values():
    spine.set_color("black")
plt.tight_layout()
plt.savefig("Gamma.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved Gamma.png")

# 4) tau
tau_cols = [c for c in data.columns if c.startswith("tau_")]
colors_tau = ["indianred", "orange", "royalblue", "darkgreen", "pink"]
linestyles_tau = ["--", ":", "-", "-.", (0, (5, 1))]
fig, ax = plt.subplots(figsize=(5, 3.5))
for i, col in enumerate(tau_cols):
    ax.plot(x, data[col].values, label=col, color=colors_tau[i % len(colors_tau)], linestyle=linestyles_tau[i % len(linestyles_tau)])
ax.set_xlim(0, 10)
ax.set_xticks(range(1, 11))
ax.set_ylabel("Average Return")
ax.set_xlabel("Time Steps (1e6)")
ax.legend(loc="lower right", frameon=False)
ax.set_facecolor("white")
ax.yaxis.grid(True, linestyle="-", linewidth=0.1, color="black")
ax.xaxis.grid(False)
for spine in ax.spines.values():
    spine.set_color("black")
plt.tight_layout()
plt.savefig("tau.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tau.png")

print("Done. Figures in: " + os.path.abspath("."))
