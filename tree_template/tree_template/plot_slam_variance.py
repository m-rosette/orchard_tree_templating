"""Plot the landmark covariance from the row_fast_slam landmark debug CSV.

Usage: python3 plot_slam_variance.py [--csv PATH]
Default CSV: debug_data/fastslam_debug_landmarks.csv in the tree_template source folder.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm, colors


def plot_cov_ellipse(ax, mean, cov, color):

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    width = 2 * np.sqrt(eigvals[0])
    height = 2 * np.sqrt(eigvals[1])

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor='none',
        linewidth=1.5
    )

    ax.add_patch(ellipse)


DEFAULT_CSV = Path(__file__).resolve().parent.parent / "debug_data" / "fastslam_debug_landmarks.csv"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="landmark debug CSV (default: %(default)s)")
df = pd.read_csv(ap.parse_args().csv)

# compute convergence metric
areas = []

for _, row in df.iterrows():
    cov = np.array([
        [row["sigma_ss"], row["sigma_sd"]],
        [row["sigma_sd"], row["sigma_dd"]]
    ])

    areas.append(np.sqrt(np.linalg.det(cov)))

areas = np.array(areas)

# normalize for colormap
norm = colors.Normalize(vmin=areas.min(), vmax=areas.max())
cmap = cm.coolwarm

fig, ax = plt.subplots(figsize=(12, 6))

for i, row in df.iterrows():

    # s on x-axis, d on y-axis
    mean = [row["mu_s"], row["mu_d"]]

    cov = np.array([
        [row["sigma_ss"], row["sigma_sd"]],
        [row["sigma_sd"], row["sigma_dd"]]
    ])

    area = np.sqrt(np.linalg.det(cov))
    color = cmap(norm(area))

    plot_cov_ellipse(ax, mean, cov, color)


sm = cm.ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(sm, ax=ax, label="Covariance ellipse size")

ax.set_xlabel("s position")
ax.set_ylabel("d position")
ax.set_title("Landmark covariance convergence")

# plot landmark mean positions as scatter points
ax.scatter(df["mu_s"], df["mu_d"], s=5, color="red", label="landmark mean", zorder=3)

ax.axhline(0, linestyle="--", linewidth=0.5)
ax.axvline(0, linestyle="--", linewidth=0.5)
ax.grid(True)
ax.set_aspect('equal')

plt.legend()
plt.tight_layout()
plt.show()