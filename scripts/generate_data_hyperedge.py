"""Build data-driven hyperedges (E_data) from Pearson correlation of traffic speed.

Key points
- Correlation is computed on the TRAIN split only (no test-set leakage).
- The mean daily profile is removed first, otherwise every road looks correlated
  simply because all of them slow down at rush hour.
- Hyperedge i = node i + all nodes j with rho_ij >= tau (optionally top-k).
- Sweeps several tau values so the choice of tau can be justified (ablation).

Usage (from project root):
    python scripts/generate_data_hyperedge.py
    python scripts/generate_data_hyperedge.py --taus 0.6 0.7 --top-k 10
"""
import argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / 'outputs'


def detrend_time_of_day(x, steps_per_day=96, fit_len=None):
    """Subtract the mean daily profile. x: (T, N). Profile is fitted on the first
    `fit_len` steps only (train period). Assumes the series is continuous and
    starts at a day boundary; otherwise the profile is shifted but still valid."""
    T, N = x.shape
    fit_len = fit_len or T
    slot = np.arange(T) % steps_per_day
    profile = np.zeros((steps_per_day, N), dtype=np.float64)
    for s in range(steps_per_day):
        idx = np.where(slot[:fit_len] == s)[0]
        if len(idx):
            profile[s] = x[idx].mean(axis=0)
    return x - profile[slot]


def pearson_matrix(x):
    """x: (T, N) -> (N, N) Pearson matrix. Constant series (NaN) -> 0."""
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = np.corrcoef(x, rowvar=False)
    rho = np.nan_to_num(rho, nan=0.0)
    np.fill_diagonal(rho, 1.0)
    return rho


def build_incidence(rho, tau, top_k=None):
    """Return incidence matrix H of shape (N, E).

    Hyperedge i contains node i and every j with rho_ij >= tau
    (limited to the top_k strongest if top_k is set). Hyperedges with a single
    node are dropped and duplicates are merged.
    This function only needs a (N, N) correlation matrix, so it can also be
    called per sliding window for a dynamic version.
    """
    N = rho.shape[0]
    H = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        cand = np.where(rho[i] >= tau)[0]
        cand = cand[cand != i]
        if top_k is not None and len(cand) > top_k:
            cand = cand[np.argsort(rho[i, cand])[::-1][:top_k]]
        H[i, i] = 1.0
        H[cand, i] = 1.0
    H = H[:, H.sum(axis=0) >= 2]
    if H.shape[1] > 0:
        H = np.unique(H, axis=1)
    return H


def hypergraph_operator(H):
    """HGNN propagation matrix G = Dv^-1/2 H De^-1 H^T Dv^-1/2, shape (N, N).
    Nodes that belong to no hyperedge keep a self-loop."""
    N, E = H.shape
    if E == 0:
        return np.eye(N, dtype=np.float32)
    dv = H.sum(axis=1)
    de = H.sum(axis=0)
    dv_inv_sqrt = np.where(dv > 0, 1.0 / np.sqrt(np.maximum(dv, 1e-8)), 0.0)
    left = dv_inv_sqrt[:, None] * H / de[None, :]
    G = left @ (H.T * dv_inv_sqrt[None, :])
    isolated = dv == 0
    G[isolated, isolated] = 1.0
    return G.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default=str(OUT_DIR / 'HCMC_Traffic_Data.npz'))
    parser.add_argument('--feature', type=int, default=0, help='feature index (0 = speed)')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--taus', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--steps-per-day', type=int, default=96)
    parser.add_argument('--no-detrend', action='store_true')
    args = parser.parse_args()

    data = np.load(args.npz)['data']  # (T, N, F)
    T, N, F = data.shape
    x = data[:, :, args.feature].astype(np.float64)
    train_len = int(T * args.train_ratio)
    print(f'Data: T={T}, N={N}, F={F} | train steps used: {train_len}')

    if not args.no_detrend:
        x = detrend_time_of_day(x, args.steps_per_day, fit_len=train_len)
    rho = pearson_matrix(x[:train_len])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / 'data_correlation_train.npy', rho)

    off_diag = rho[~np.eye(N, dtype=bool)]
    print(f'Off-diagonal rho: mean={off_diag.mean():.3f}, '
          f'p90={np.percentile(off_diag, 90):.3f}, max={off_diag.max():.3f}\n')

    print(f'{"tau":>5} {"#hyperedges":>12} {"avg size":>9} {"G density":>10}')
    for tau in args.taus:
        H = build_incidence(rho, tau, args.top_k)
        G = hypergraph_operator(H)
        n_edges = H.shape[1]
        avg_size = H.sum(axis=0).mean() if n_edges else 0.0
        density = np.count_nonzero(G) / (N * N) * 100
        print(f'{tau:>5.2f} {n_edges:>12d} {avg_size:>9.2f} {density:>9.2f}%')
        np.save(OUT_DIR / f'hyperedge_incidence_tau{tau:.2f}.npy', H)
        np.save(OUT_DIR / f'hyperedge_operator_tau{tau:.2f}.npy', G)

    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
