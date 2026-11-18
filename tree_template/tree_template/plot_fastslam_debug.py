#!/usr/bin/env python3
"""Plot the row_fast_slam debug CSV.

Usage: python3 plot_fastslam_debug.py [--csv PATH] [--out_dir DIR] [--show]
Default CSV: debug_data/fastslam_debug.csv in the tree_template source folder (the row_fast_slam default).
Default output: the folder of the CSV.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "debug_data" / "fastslam_debug.csv"

SAVE_DPI = 150
OUTPUT_DIR = DEFAULT_CSV.parent   # main() sets it from the arguments
SHOW_PLOTS = False                # main() sets it from --show


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
    global OUTPUT_DIR, SHOW_PLOTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="debug CSV (default: %(default)s)")
    ap.add_argument("--out_dir", type=Path, default=None, help="PNG output folder (default: the CSV folder)")
    ap.add_argument("--show", action="store_true", help="also show the plots in windows")
    args = ap.parse_args()

    CSV_PATH = args.csv
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    OUTPUT_DIR = args.out_dir or CSV_PATH.parent
    SHOW_PLOTS = args.show

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
