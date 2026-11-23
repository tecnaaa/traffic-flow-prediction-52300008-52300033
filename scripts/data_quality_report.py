"""Data quality report BEFORE interpolation.

Rebuilds the same (T x N) table as scripts/prepare_pems_dataset.py (pivot by
road_name, 15-min resample) but keeps the NaNs, then reports:

1. Missing rate overall / per split / per node
2. Gap lengths (how long the interpolated stretches are)
3. Network-wide outages (most nodes missing at once) and worst days
4. Pearson hyperedge check: edges computed on interpolated data vs on really
   observed pairs only. If many edges exist only on interpolated data, the
   hyperedges partly encode shared data outages, not traffic.

Usage (from the project root):
    python scripts/data_quality_report.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLIT_RATIO = (0.6, 0.2, 0.2)  # must equal split_ratio in scripts/data_loader.py


def build_pivot(csv_path, value_col):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[df['timestamp'] != 'timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    piv = df.pivot_table(index='timestamp', columns='road_name', values=value_col)
    return piv.resample('15min').mean()


def gap_runs(isnan):
    """Return list of (length, touches_edge) for consecutive NaN runs of a 1-D bool array."""
    runs, i, n = [], 0, len(isnan)
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            runs.append((j - i, i == 0 or j == n))
            i = j
        else:
            i += 1
    return runs


def detrend(piv, train_end):
    """Subtract the time-of-day profile (fitted on train rows, NaN-aware)."""
    slot = (piv.index.hour * 60 + piv.index.minute) // 15
    profile = piv.iloc[:train_end].groupby(slot[:train_end]).mean()
    return piv - profile.reindex(slot).values


def edge_set(corr, tau):
    c = corr.values.copy()
    np.fill_diagonal(c, 0.0)
    c = np.nan_to_num(c, nan=0.0)
    iu = np.triu_indices_from(c, k=1)
    keep = c[iu] >= tau
    return set(zip(iu[0][keep], iu[1][keep]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=str(ROOT_DIR / 'outputs' / 'traffic_data_cleaned.csv'))
    p.add_argument('--value-col', default='current_avg_speed_kmh')
    p.add_argument('--npz', default=str(ROOT_DIR / 'outputs' / 'HCMC_Traffic_Data.npz'))
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--pre-len', type=int, default=12)
    p.add_argument('--tau', type=float, default=0.7)
    p.add_argument('--min-overlap', type=int, default=200,
                   help='min jointly observed train steps for a Pearson pair')
    args = p.parse_args()

    piv = build_pivot(args.csv, args.value_col)
    T, N = piv.shape
    isnan = piv.isna().values
    print(f"Range: {piv.index[0]} -> {piv.index[-1]} | T={T} steps ({T * 15 / 60 / 24:.1f} days), N={N}")
    if Path(args.npz).exists():
        t_npz = np.load(args.npz)['data'].shape[0]
        print(f"npz has T={t_npz} {'(matches)' if t_npz == T else '(MISMATCH: check the pipeline!)'}")

    n_win = T - args.seq_len - args.pre_len + 1
    b1 = int(n_win * SPLIT_RATIO[0]) + args.seq_len
    b2 = int(n_win * (SPLIT_RATIO[0] + SPLIT_RATIO[1])) + args.seq_len

    # 1. Missing rates
    print("\n== 1. Missing rate BEFORE interpolation ==")
    for name, sl in [('train', slice(0, b1)), ('val', slice(b1, b2)),
                     ('test', slice(b2, T)), ('ALL', slice(0, T))]:
        print(f"{name:>6}: {isnan[sl].mean() * 100:6.2f}%")
    node_missing = isnan.mean(axis=0) * 100
    print(f"Per node: min {node_missing.min():.1f}% | median {np.median(node_missing):.1f}% | "
          f"max {node_missing.max():.1f}% | nodes with >50% missing: {(node_missing > 50).sum()}")

    # 2. Gap lengths
    print("\n== 2. Gap lengths (consecutive missing steps, per node) ==")
    all_runs = [r for j in range(N) for r in gap_runs(isnan[:, j])]
    if all_runs:
        lengths = np.array([r[0] for r in all_runs])
        edge = np.array([r[1] for r in all_runs])
        total_missing = lengths.sum()
        print(f"gaps: {len(lengths)} | median {np.median(lengths):.0f} steps | "
              f"p90 {np.percentile(lengths, 90):.0f} | max {lengths.max()} steps ({lengths.max() * 15 / 60:.1f} h)")
        for steps, label in [(4, '> 1 hour'), (12, '> 3 hours'), (96, '> 1 day')]:
            share = lengths[lengths > steps].sum() / total_missing * 100
            print(f"missing steps inside gaps {label:>9}: {share:5.1f}% of all missing")
        print(f"missing steps at series start/end (filled with a CONSTANT by limit_direction='both'): "
              f"{lengths[edge].sum() / total_missing * 100:.1f}%")
    else:
        print("no gaps")

    # 3. Network-wide outages and worst days
    print("\n== 3. Network-wide outages (>50% of nodes missing at the same time) ==")
    out = isnan.mean(axis=1) > 0.5
    print(f"steps in outage: {out.sum()} ({out.mean() * 100:.1f}% of time)")
    outages, pos = [], 0
    while pos < T:
        if out[pos]:
            end = pos
            while end < T and out[end]:
                end += 1
            outages.append((end - pos, pos, end - 1))
            pos = end
        else:
            pos += 1
    if outages:
        print("Longest outages:")
        for length, s, e in sorted(outages, reverse=True)[:5]:
            print(f"  {piv.index[s]} -> {piv.index[e]}  ({length * 15 / 60:.1f} h)")
    daily = pd.Series(1 - isnan.mean(axis=1), index=piv.index).groupby(piv.index.date).mean() * 100
    print("Worst 5 days by observed coverage:")
    for day, cov in daily.nsmallest(5).items():
        print(f"  {day}: {cov:.1f}% observed")

    # 4. Pearson: interpolated vs observed-only (train period only, time-of-day removed)
    print(f"\n== 4. Pearson edges (rho >= {args.tau}), train period, daily profile removed ==")
    train = piv.iloc[:b1]
    interp = piv.interpolate(method='linear', limit_direction='both').iloc[:b1]
    corr_interp = detrend(piv.interpolate(method='linear', limit_direction='both'), b1).iloc[:b1].corr()
    corr_obs = detrend(piv, b1).iloc[:b1].corr(min_periods=args.min_overlap)
    e_interp, e_obs = edge_set(corr_interp, args.tau), edge_set(corr_obs, args.tau)
    union = len(e_interp | e_obs)
    print(f"edges on interpolated data : {len(e_interp)}")
    print(f"edges on observed pairs    : {len(e_obs)}  (pairs with >= {args.min_overlap} joint steps)")
    print(f"edges in both              : {len(e_interp & e_obs)} | Jaccard = "
          f"{len(e_interp & e_obs) / union if union else float('nan'):.2f}")
    only_interp = len(e_interp - e_obs)
    if e_interp:
        print(f"edges ONLY on interpolated data: {only_interp} ({only_interp / len(e_interp) * 100:.0f}% of them)")
    print("\nHow to read:")
    print("- High missing rate or many long gaps -> interpolated ramps dominate the series.")
    print("- Large share of missing in gaps > 3h, or outages shared by many nodes -> Pearson on interpolated")
    print("  data can reflect shared outages. A high 'ONLY on interpolated' share confirms it.")


if __name__ == '__main__':
    main()