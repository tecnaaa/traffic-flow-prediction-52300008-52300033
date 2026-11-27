import torch
from train_stgcn import get_dataloader, compute_metrics

tr, va, te, scaler = get_dataloader('outputs/PEMS08_Traffic_Data.npz', 12, 12, 64)
x, y = next(iter(te))
print('x', x.shape, 'x range', x[..., 0].min().item(), x[..., 0].max().item())
print('y', y.shape, 'y range', y[..., 0].min().item(), y[..., 0].max().item())

preds, truths = [], []
for x, y in te:
    last = scaler.inverse_transform(x[:, -1, :, 0].numpy())   # (B, N), thang gốc
    last = torch.FloatTensor(last)
    preds.append(last.unsqueeze(-1).repeat(1, 1, 12))          # (B, N, 12)
    truths.append(y[..., 0].permute(0, 2, 1))                  # (B, N, 12)
preds, truths = torch.cat(preds), torch.cat(truths)

print('Overall', compute_metrics(preds, truths))
for h in (0, 5, 11):
    print('step', h + 1, compute_metrics(preds[..., h], truths[..., h]))