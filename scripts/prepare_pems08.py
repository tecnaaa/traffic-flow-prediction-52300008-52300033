"""Prepare PEMS08 for the existing pipeline (data_loader.py / train_*.py).

PEMS08 raw file has shape (17856, 170, 3): channel 0 = flow, 1 = occupancy,
2 = speed (this ordering is the standard ASTGCN/STSGCN convention). Most papers
report metrics on the FLOW channel, so that is the default here.

Output: outputs/PEMS08_Traffic_Data.npz with shape (T, N, 1), same layout as
the TP.HCM file (HCMC_Traffic_Data.npz), so get_dataloader() works unchanged.

Usage (from the project root):
    python scripts/prepare_pems08.py
    python scripts/prepare_pems08.py --channel 2   # use speed instead of flow
"""
import argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
CHANNEL_NAMES = {0: 'flow', 1: 'occupancy', 2: 'speed'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default=str(ROOT_DIR / 'data' / 'PEMS08' / 'pems08.npz'))
    p.add_argument('--channel', type=int, default=0, choices=[0, 1, 2],
                   help='0=flow (default, matches most papers), 1=occupancy, 2=speed')
    p.add_argument('--out', default=str(ROOT_DIR / 'outputs' / 'PEMS08_Traffic_Data.npz'))
    args = p.parse_args()

    data = np.load(args.raw)['data']  # (T, N, F)
    T, N, F = data.shape
    print(f"Raw PEMS08: T={T}, N={N}, F={F}")
    assert (T, N, F) == (17856, 170, 3), (
        f"Unexpected shape {data.shape}; expected (17856, 170, 3). "
        "Check that pems08.npz was downloaded correctly (not corrupted/truncated).")

    channel = data[:, :, args.channel:args.channel + 1].astype(np.float32)  # (T, N, 1)
    name = CHANNEL_NAMES[args.channel]

    print(f"\nUsing channel {args.channel} ({name})")
    print(f"mean={channel.mean():.2f}, std={channel.std():.2f}, "
          f"min={channel.min():.2f}, max={channel.max():.2f}")
    zero_pct = (channel == 0).mean() * 100
    print(f"exact-zero values: {zero_pct:.2f}% "
          f"({'normal for flow at night' if name == 'flow' else 'check if high'})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, data=channel)
    print(f"\nSaved {args.out}  shape={channel.shape}")
    print(f"5 min/step -> {T * 5 / 60 / 24:.1f} days total "
          f"({T} steps; with seq_len=pre_len=12, that is 60 min windows)")


if __name__ == '__main__':
    main()
