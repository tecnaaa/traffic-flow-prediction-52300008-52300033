"""Create control graphs for the PEMS08 ablation (same size/density as the real operator)."""
import numpy as np
from pathlib import Path

OUT = Path('outputs')
G = np.load(OUT / 'hyperedge_operator_pems08_tau0.80.npy')
N = G.shape[0]

# Control 1: identity (no cross-node information, same extra parameters)
np.save(OUT / 'control_identity_pems08.npy', np.eye(N, dtype=np.float32))

# Control 2: shuffled node labels (same structure and density, wrong correspondence)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
np.save(OUT / 'control_shuffled_pems08_tau0.80.npy', G[np.ix_(perm, perm)].astype(np.float32))
print('saved controls', G.shape)