"""Build the physical adjacency matrix for PEMS08 from distance.csv.

IMPORTANT: for PEMS03/04/07/08, distance.csv is NOT a dense pairwise distance
matrix (unlike METR-LA/PEMS-BAY). It is already a SPARSE, known list of real
road edges (PEMS08 has exactly 295 edges over 170 nodes). So the graph
structure comes directly from this edge list; this script does not apply any
further Gaussian-kernel sparsification (doing so would drop most of the
already-sparse real edges and isolate most nodes).

Two weighting schemes (--method):
    connectivity (default) : binary adjacency, 1.0 for every listed edge
    weighted                : Gaussian-kernel weight exp(-d_ij^2 / sigma^2) for
                              every listed edge (no thresholding), so closer
                              edges get a higher weight but every real edge is kept

distance.csv has columns (from, to, cost) with NODE IDS 0..N-1 for PEMS08.

Output: outputs/physical_adj_matrix.npy, shape (N, N), compatible with
train_stgcn.py and train_hgmwn.py (both apply their own D^-1(A+I) normalization
on top of this raw graph, so this script does NOT normalize).

Usage (from the project root):
    python scripts/build_physical_graph_pems08.py
    python scripts/build_physical_graph_pems08.py --method weighted
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--distance-csv', default=str(ROOT_DIR / 'data' / 'PEMS08' / 'distance.csv'))
    p.add_argument('--num-nodes', type=int, default=170)
    p.add_argument('--method', choices=['connectivity', 'weighted'], default='connectivity',
                   help='connectivity=binary (standard for PEMS03/04/07/08), '
                        'weighted=Gaussian kernel on the existing edges, no extra thresholding')
    p.add_argument('--out', default=str(ROOT_DIR / 'outputs' / 'physical_adj_matrix.npy'))
    args = p.parse_args()

    df = pd.read_csv(args.distance_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {'from', 'to', 'cost'}.issubset(df.columns), (
        f"Expected columns from/to/cost, got {list(df.columns)}")

    N = args.num_nodes
    max_id = max(df['from'].max(), df['to'].max())
    assert max_id < N, (
        f"distance.csv has node id {max_id} but --num-nodes={N}; "
        "ids are expected to already be 0-indexed 0..N-1 for PEMS08.")

    costs = df['cost'].to_numpy(dtype=np.float64)
    print(f"Edges in file: {len(df)} | cost stats: mean={costs.mean():.1f}, "
          f"std={costs.std():.1f}, min={costs.min():.1f}, max={costs.max():.1f}")

    adj = np.zeros((N, N), dtype=np.float32)
    if args.method == 'connectivity':
        for f, t in zip(df['from'], df['to']):
            adj[int(f), int(t)] = 1.0
            adj[int(t), int(f)] = 1.0  # undirected, matching ASTGCN's convention
    else:
        sigma = costs.std()
        weights = np.exp(-(costs ** 2) / (sigma ** 2))
        for f, t, w in zip(df['from'], df['to'], weights):
            adj[int(f), int(t)] = max(adj[int(f), int(t)], w)
            adj[int(t), int(f)] = max(adj[int(t), int(f)], w)

    density = np.count_nonzero(adj) / (N * N) * 100
    degree = adj.sum(axis=1)
    isolated = int((degree == 0).sum())
    print(f"method={args.method} -> density={density:.2f}%, "
          f"avg degree={np.count_nonzero(adj, axis=1).mean():.1f}, isolated nodes: {isolated}")
    if isolated > 0:
        print(f"NOTE: {isolated} node(s) have no road connection in distance.csv. "
              "train_hgmwn.py's row-normalization adds a self-loop, so these nodes still "
              "work (they just only see their own history through the physical graph).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, adj)
    print(f"Saved {args.out}  shape={adj.shape}")


if __name__ == '__main__':
    main()