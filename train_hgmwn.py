"""Train / evaluate the proposed HGMWN model.

Examples (run from the project root):
    python train_hgmwn.py --mode dual --data-adj outputs/hyperedge_operator_tau0.70.npy
    python train_hgmwn.py --mode phys
    python train_hgmwn.py --mode data --data-adj outputs/hyperedge_operator_tau0.70.npy

Notes
- Labels (y) stay in raw units. For the training loss they are standardized
  with the train-fitted scaler; for evaluation ONLY the predictions are
  inverse-transformed, so MAE/RMSE/MAPE are in real km/h.
- The best epoch is chosen on the validation set (MAE) and restored for test.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.hgmwn import HGMWN, topk_sparsify

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.data_loader import get_dataloader
from models.hgmwn import HGMWN
from utils.metrics import masked_mae, masked_rmse, masked_mape

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HORIZONS = (1, 2, 4, 12)  # in steps; minutes = step * --step-minutes


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_normalize(adj, add_self_loop=True):
    """D^-1 (A + I): keeps the scale of graph propagation stable."""
    a = adj.astype(np.float32).copy()
    if add_self_loop:
        a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return a / deg


def compute_metrics(preds, truths):
    """preds, truths: float tensors in real units."""
    return {
        'MAE': masked_mae(preds, truths).item(),
        'RMSE': masked_rmse(preds, truths).item(),
        'MAPE': masked_mape(preds, truths).item(),
    }


def predict(model, loader, adj_phys, adj_data, scaler):
    """Return (preds, truths) in real units, shape (Samples, Nodes, PreLen)."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x, adj_phys, adj_data)
            # Inverse-transform predictions only; labels are already raw.
            preds.append(scaler.inverse_transform(output.cpu().numpy()))
            truths.append(batch_y[..., 0].permute(0, 2, 1).numpy())
    preds = torch.FloatTensor(np.concatenate(preds, axis=0))
    truths = torch.FloatTensor(np.concatenate(truths, axis=0))
    return preds, truths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['phys', 'data', 'dual'], default='dual')
    p.add_argument('--npz', default=str(ROOT_DIR / 'outputs' / 'HCMC_Traffic_Data.npz'))
    p.add_argument('--phys-adj', default=str(ROOT_DIR / 'outputs' / 'physical_adj_matrix.npy'))
    p.add_argument('--data-adj',
                   default=str(ROOT_DIR / 'outputs' / 'hyperedge_operator_tau0.70.npy'))
    p.add_argument('--normalize-data-adj', action='store_true',
                   help='row-normalize the data graph (not needed for the hypergraph operator)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--pre-len', type=int, default=12)
    p.add_argument('--step-minutes', type=int, default=15,
                   help='minutes per time step of the dataset (for horizon labels)')
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--topk', type=int, default=0,
            help='keep top-k neighbours per row of the physical graph (0 = off)')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"HGMWN | mode={args.mode} | device={DEVICE} | seed={args.seed}")

        # 1. Graphs
    adj_raw = np.load(args.phys_adj)
    if args.topk > 0:
        adj_raw = topk_sparsify(adj_raw, args.topk)  # sparsify before normalizing
    adj_phys = row_normalize(adj_raw)
    num_nodes = adj_phys.shape[0]
    adj_phys_t = torch.FloatTensor(adj_phys).to(DEVICE)

    if args.mode == 'phys':
        adj_data_t = adj_phys_t  # unused in this mode
        tag = 'phys'
    else:
        adj_data = np.load(args.data_adj)
        assert adj_data.shape == (num_nodes, num_nodes), (
            f"Data graph shape {adj_data.shape} != physical graph ({num_nodes}, {num_nodes})")
        if args.normalize_data_adj:
            adj_data = row_normalize(adj_data)
        adj_data_t = torch.FloatTensor(adj_data).to(DEVICE)
        tag = f"{args.mode}_{Path(args.data_adj).stem}"

    # 2. Data
    train_loader, val_loader, test_loader, scaler = get_dataloader(
        args.npz, seq_len=args.seq_len, pre_len=args.pre_len, batch_size=args.batch_size)
    mean, std = float(scaler.mean), float(scaler.std)
    print(f"Scaler: mean={mean:.4f}, std={std:.4f}")

    sample_x, _ = next(iter(train_loader))
    assert sample_x.shape[2] == num_nodes, (
        f"Data has {sample_x.shape[2]} nodes but graphs have {num_nodes}")
    in_dim = sample_x.shape[-1]

    # 3. Model
    model = HGMWN(num_nodes=num_nodes, in_dim=in_dim, out_dim=args.pre_len,
                  seq_len=args.seq_len, mode=args.mode).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ckpt_path = ROOT_DIR / 'outputs' / f'hgmwn_{tag}_seed{args.seed}.pt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Training with validation-based model selection
    best_val, best_epoch, bad_epochs = float('inf'), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, start = 0.0, time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            output = model(batch_x, adj_phys_t, adj_data_t)

            # Labels are raw -> standardize with the train scaler for the loss
            truth = batch_y[..., 0].permute(0, 2, 1)
            truth = (truth - mean) / std

            loss = criterion(output, truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_preds, val_truths = predict(model, val_loader, adj_phys_t, adj_data_t, scaler)
        val = compute_metrics(val_preds, val_truths)

        improved = val['MAE'] < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val['MAE'], epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss/len(train_loader):.4f} "
              f"| val MAE {val['MAE']:.4f} RMSE {val['RMSE']:.4f} MAPE {val['MAPE']:.2f}% "
              f"| {time.time()-start:.1f}s{' *' if improved else ''}")

        if bad_epochs >= args.patience:
            print(f"Early stopping (no val improvement for {args.patience} epochs).")
            break

    # 5. Test with the best checkpoint
    print(f"\nLoading best model (epoch {best_epoch}, val MAE {best_val:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    preds, truths = predict(model, test_loader, adj_phys_t, adj_data_t, scaler)

    overall = compute_metrics(preds, truths)
    horizon = {}
    for h in HORIZONS:
        if h <= args.pre_len:
            horizon[f'{h * args.step_minutes}min'] = compute_metrics(preds[..., h - 1], truths[..., h - 1])

    print(f"\n=== TEST RESULTS (HGMWN, mode={args.mode}) ===")
    print(f"Overall | MAE {overall['MAE']:.4f} | RMSE {overall['RMSE']:.4f} | MAPE {overall['MAPE']:.2f}%")
    for name, m in horizon.items():
        print(f"{name:>7} | MAE {m['MAE']:.4f} | RMSE {m['RMSE']:.4f} | MAPE {m['MAPE']:.2f}%")

    alphas = model.alphas()
    if alphas:
        print("Physical-graph weight per layer (sigmoid(alpha)): "
              + ", ".join(f"{a:.3f}" for a in alphas))

    # 6. Save results for the report tables
    result = {
        'tag': tag, 'mode': args.mode, 'seed': args.seed,
        'best_epoch': best_epoch, 'best_val_mae': best_val, 'n_params': n_params,
        'overall': overall, 'horizon': horizon, 'alphas': alphas,
    }
    result_path = ROOT_DIR / 'outputs' / f'results_hgmwn_{tag}_seed{args.seed}.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
