"""Data diagnostics per split (numpy only).

For each of train / val / test it prints:
- mean and std of speed
- between-node std (std of node means)   -> spread of speed LEVELS across roads
- within-node std (mean over nodes of std over time)
- lag-1 mean |x_t - x_{t-1}|             -> MAE of 1-step persistence
- fraction of lag-1 differences that are exactly 0 (stale / repeated values)
- fraction of values equal to 0
- 'network-mean floor': MAE if every node predicted the network-average speed at
  that time. A model that over-smooths ends up near this number.

Usage (from the project root):
    python scripts/data_diagnostics.py
"""
import argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLIT_RATIO = (0.6, 0.2, 0.2)  # must equal split_ratio in scripts/data_loader.py


def describe(name, x):
    """x: (T, N)."""
    node_means = x.mean(axis=0)
    lag1 = np.abs(np.diff(x, axis=0))
    network_mean = x.mean(axis=1, keepdims=True)
    return {
        'split': name,
        'steps': x.shape[0],
        'mean': x.mean(),
        'std': x.std(),
        'between_node_std': node_means.std(),
        'within_node_std': x.std(axis=0).mean(),
        'lag1_mae': lag1.mean(),
        'zero_diff_pct': (lag1 < 1e-6).mean() * 100,
        'zero_value_pct': (x == 0).mean() * 100,
        'network_mean_floor': np.abs(x - network_mean).mean(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default=str(ROOT_DIR / 'outputs' / 'HCMC_Traffic_Data.npz'))
    p.add_argument('--feature', type=int, default=0)
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--pre-len', type=int, default=12)
    args = p.parse_args()

    data = np.load(args.npz)['data'][..., args.feature].astype(np.float64)  # (T, N)
    T, N = data.shape
    seq, pre = args.seq_len, args.pre_len

    # Approximate time boundaries of the window-based split in data_loader.py
    n_win = T - seq - pre + 1
    b1 = int(n_win * SPLIT_RATIO[0]) + seq
    b2 = int(n_win * (SPLIT_RATIO[0] + SPLIT_RATIO[1])) + seq
    parts = [('train', data[:b1]), ('val', data[b1:b2]), ('test', data[b2:])]

    print(f"T={T}, N={N} | boundaries: train [0,{b1}), val [{b1},{b2}), test [{b2},{T})\n")
    rows = [describe(name, x) for name, x in parts] + [describe('ALL', data)]

    cols = [('steps', 'steps', '{:.0f}'), ('mean', 'mean', '{:.2f}'), ('std', 'std', '{:.2f}'),
            ('between_node_std', 'btw_node', '{:.2f}'), ('within_node_std', 'in_node', '{:.2f}'),
            ('lag1_mae', 'lag1_mae', '{:.3f}'), ('zero_diff_pct', 'zero_diff%', '{:.1f}'),
            ('zero_value_pct', 'zero_val%', '{:.1f}'), ('network_mean_floor', 'net_floor', '{:.2f}')]
    print(f"{'split':>6}" + ''.join(f"{label:>12}" for _, label, _ in cols))
    for r in rows:
        print(f"{r['split']:>6}" + ''.join(f"{fmt.format(r[key]):>12}" for key, _, fmt in cols))
    print("\nbtw_node = std of node mean speeds | in_node = mean within-node std | "
          "net_floor = network-mean floor")

    print("\nHow to read:")
    print("- network_mean_floor close to the MAE of phys/data/dual (3.5-4.6) -> those models were over-smoothed.")
    print("- lag1_mae is the 1-step persistence MAE; if it is tiny, the series is very smooth.")
    print("- zero_diff_pct high (e.g. >50%) -> many repeated/stale/interpolated values.")
    print("- A large gap in lag1_mae or zero_diff_pct between val and test -> the splits differ in difficulty.")


if __name__ == '__main__':
    main()