#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = "/home/marcus/apple_harvest_ws/src/orchard_tree_templating/tree_template/debug_data/fastslam_debug.csv"

# 🔧 choose your output directory here
OUTPUT_DIR = Path(
    "/home/marcus/apple_harvest_ws/src/orchard_tree_templating/tree_template/debug_data"
)

SAVE_DPI = 150
SHOW_PLOTS = False   # set True if you also want interactive windows


def read_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out = {}
            for k, v in row.items():
                if v is None or v == "":
                    out[k] = math.nan
                    continue
                if k == "reject_reason_counts":
                    out[k] = v
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows


def series(rows, key):
    return [r.get(key, math.nan) for r in rows]


def save_fig(name):
    path = OUTPUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=SAVE_DPI)
    print(f"[saved] {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def main():
    p = Path(CSV_PATH)
    if not p.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(CSV_PATH)
    if not rows:
        raise SystemExit("CSV is empty.")

    meas_idx = series(rows, "meas_idx")

    # --- Plot 1: Particle degeneracy ---
    plt.figure()
    plt.plot(meas_idx, series(rows, "neff"))
    plt.xlabel("measurement index")
    plt.ylabel("Neff")
    plt.title("Effective sample size (degeneracy)")
    plt.grid(True)
    save_fig("neff")

    # --- Plot 2: Weight concentration ---
    plt.figure()
    plt.plot(meas_idx, series(rows, "w_max"), label="w_max")
    plt.plot(meas_idx, series(rows, "w_std"), label="w_std")
    plt.xlabel("measurement index")
    plt.ylabel("weight stats")
    plt.title("Weight concentration")
    plt.grid(True)
    plt.legend()
    save_fig("weight_concentration")

    # --- Plot 3: Likelihood terms (Mahalanobis) ---
    plt.figure()
    plt.plot(meas_idx, series(rows, "maha_mean"), label="maha_mean")
    plt.plot(meas_idx, series(rows, "maha_p95"), label="maha_p95")
    plt.xlabel("measurement index")
    plt.ylabel("Mahalanobis distance")
    plt.title("Innovation size (Mahalanobis)")
    plt.grid(True)
    plt.legend()
    save_fig("mahalanobis")

    # --- Plot 4: Measurement covariance (logdet) ---
    plt.figure()
    plt.plot(meas_idx, series(rows, "logdet_mean"), label="logdet_mean")
    plt.plot(meas_idx, series(rows, "logdet_p95"), label="logdet_p95")
    plt.xlabel("measurement index")
    plt.ylabel("logdet(S)")
    plt.title("Measurement covariance term (logdet)")
    plt.grid(True)
    plt.legend()
    save_fig("logdet")

    # --- Plot 5: Likelihood vs penalties ---
    plt.figure()
    plt.plot(meas_idx, series(rows, "logw_mean"), label="logw_mean")
    plt.xlabel("measurement index")
    plt.ylabel("log terms (mean over particles)")
    plt.title("Likelihood vs penalties (mean)")
    plt.grid(True)
    plt.legend()
    save_fig("likelihood_vs_penalties")

    # --- Plot 6: Measurement range ---
    plt.figure()
    plt.plot(meas_idx, series(rows, "range_r"))
    plt.xlabel("measurement index")
    plt.ylabel("range r (m)")
    plt.title("Measurement range")
    plt.grid(True)
    save_fig("measurement_range")


if __name__ == "__main__":
    main()
