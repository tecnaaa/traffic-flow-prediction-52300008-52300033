"""Naive baselines on the SAME split as scripts/data_loader.py (numpy only).

- persistence        : last observed value repeated for every horizon
- historical_average : mean of the same time-of-day slot, fitted on train period only

Metrics are in real units and use the same masking rules as utils/metrics.py
(label == 0 is ignored for MAE/RMSE, label <= 0 for MAPE).

Usage (from the project root):
    python scripts/naive_baselines.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
HORIZONS = (1, 2, 4, 12)  # in steps; minutes = step * --step-minutes
SPLIT_RATIO = (0.6, 0.2, 0.2)  # must equal split_ratio in scripts/data_loader.py


def masked_metrics(pred, truth, null_val=0.0):
    valid = truth != null_val
    diff = pred - truth
    mae = np.abs(diff)[valid].mean()
    rmse = np.sqrt((diff ** 2)[valid].mean())
    valid_pos = truth > null_val
    mape = (np.abs(diff)[valid_pos] / truth[valid_pos]).mean() * 100
    return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}


def summarize(name, pred, truth, args):
    overall = masked_metrics(pred, truth)
    horizon = {}
    for h in HORIZONS:
        if h <= args.pre_len:
            horizon[f'{h * args.step_minutes}min'] = masked_metrics(pred[..., h - 1], truth[..., h - 1])

    print(f"\n=== {name} ===")
    print(f"Overall | MAE {overall['MAE']:.4f} | RMSE {overall['RMSE']:.4f} | MAPE {overall['MAPE']:.2f}%")
    for key, m in horizon.items():
        print(f"{key:>7} | MAE {m['MAE']:.4f} | RMSE {m['RMSE']:.4f} | MAPE {m['MAPE']:.2f}%")

    out_dir = ROOT_DIR / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'results_naive_{name}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'tag': name, 'model': name, 'overall': overall, 'horizon': horizon}, f, indent=2)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default=str(ROOT_DIR / 'outputs' / 'HCMC_Traffic_Data.npz'))
    p.add_argument('--feature', type=int, default=0, help='feature index (0 = speed)')
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--pre-len', type=int, default=12)
    p.add_argument('--step-minutes', type=int, default=15)
    p.add_argument('--steps-per-day', type=int, default=96)
    args = p.parse_args()

    data = np.load(args.npz)['data'][..., args.feature].astype(np.float64)  # (T, N)
    T, N = data.shape
    seq, pre = args.seq_len, args.pre_len

    # Same window indexing and split as get_dataloader()
    n_win = T - seq - pre + 1
    train_idx = int(n_win * SPLIT_RATIO[0])
    val_idx = int(n_win * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))
    test_idx = np.arange(val_idx, n_win)
    print(f"T={T}, N={N}, windows={n_win} | train={train_idx}, "
          f"val={val_idx - train_idx}, test={len(test_idx)}")

    # Targets of each test window: shape (S, N, pre)
    target_times = test_idx[:, None] + seq + np.arange(pre)[None, :]      # (S, pre)
    truth = data[target_times].transpose(0, 2, 1)

    # 1) Persistence: last observed value of the input window
    last = data[test_idx + seq - 1]                                        # (S, N)
    persistence = np.repeat(last[:, :, None], pre, axis=2)
    summarize('persistence', persistence, truth, args)

    # 2) Historical average of each time-of-day slot (train period only)
    train_end = train_idx + seq + pre - 1   # exclusive: last train target time + 1
    spd = args.steps_per_day
    slots = np.arange(train_end) % spd
    profile = np.zeros((spd, N))
    for s in range(spd):
        profile[s] = data[:train_end][slots == s].mean(axis=0)
    hist = profile[target_times % spd].transpose(0, 2, 1)
    summarize('historical_average', hist, truth, args)


if __name__ == '__main__':
    main()