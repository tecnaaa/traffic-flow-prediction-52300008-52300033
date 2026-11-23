"""Create control graphs for the dual-graph ablation (numpy only).

control_identity.npy
    Identity matrix: dual = physical graph + a pure self path. Tests whether the
    gain of `dual` comes from simply keeping each node's own signal.
control_shuffled_<tag>.npy
    The data-driven hyperedge operator with rows AND columns permuted by the same
    random permutation. Same hyperedge sizes, degrees and density, but the groups
    are no longer aligned with the real correlated roads. Tests whether the
    Pearson structure itself matters.

Usage (from the project root):
    python scripts/make_control_graphs.py
    python scripts/make_control_graphs.py --operator outputs/hyperedge_operator_tau0.70.npy
"""
import argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / 'outputs'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--operator', default=str(OUT_DIR / 'hyperedge_operator_tau0.70.npy'))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    G = np.load(args.operator).astype(np.float32)
    N = G.shape[0]
    assert G.shape == (N, N), f"Operator must be square, got {G.shape}"

    np.save(OUT_DIR / 'control_identity.npy', np.eye(N, dtype=np.float32))

    perm = np.random.default_rng(args.seed).permutation(N)
    G_shuffled = G[perm][:, perm]
    tag = Path(args.operator).stem.replace('hyperedge_operator_', '')
    out_shuffled = OUT_DIR / f'control_shuffled_{tag}.npy'
    np.save(out_shuffled, G_shuffled)

    print(f"Operator: {args.operator} (N={N})")
    print(f"Density original : {np.count_nonzero(G) / N**2 * 100:.2f}%")
    print(f"Density shuffled : {np.count_nonzero(G_shuffled) / N**2 * 100:.2f}%  (must match)")
    print(f"Fraction of entries that moved: {(G != G_shuffled).mean() * 100:.1f}%")
    print(f"Saved: {OUT_DIR / 'control_identity.npy'}")
    print(f"Saved: {out_shuffled}")


if __name__ == '__main__':
    main()