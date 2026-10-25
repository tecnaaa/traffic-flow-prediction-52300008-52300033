import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class StandardScaler:
    """
    Standardize the input data (Z-score normalization).
    Crucial for GNN stability and accurate MAE/RMSE inverse calculation.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def generate_dataset(data, seq_len, pre_len):
    """
    Generate sliding windows for time-series forecasting.
    data: Tensor of shape (Total_Time_Steps, Nodes, Features)
    seq_len: Historical window (e.g., 12 steps = 1 hour)
    pre_len: Prediction window (e.g., 12 steps = 1 hour)
    """
    x, y = [], []
    for i in range(len(data) - seq_len - pre_len + 1):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pre_len])
    
    return np.array(x), np.array(y)

def get_dataloader(file_path, seq_len=12, pre_len=12, batch_size=64, split_ratio=(0.6, 0.2, 0.2)):
    """
    Load .npz data, split into Train/Val/Test, and return PyTorch DataLoaders.
    """
    print(f"Loading dataset from: {file_path}")
    data = np.load(file_path)['data']
    
    # Extract structural dimensions
    total_steps, num_nodes, num_features = data.shape
    
    # 1. Generate sliding windows (X: Historical, Y: Future)
    x, y = generate_dataset(data, seq_len, pre_len)
    
    # 2. Split dataset sequentially (Train / Val / Test)
    train_idx = int(len(x) * split_ratio[0])
    val_idx = int(len(x) * (split_ratio[0] + split_ratio[1]))
    
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:], y[val_idx:]
    
    # 3. Fit StandardScaler ONLY on training data to prevent data leakage
    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    
    # Apply normalization to the specified feature channel (F=0 for speed)
    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    
    # 4. Convert to PyTorch Tensors
    x_train_t = torch.FloatTensor(x_train)
    y_train_t = torch.FloatTensor(y_train)
    x_val_t = torch.FloatTensor(x_val)
    y_val_t = torch.FloatTensor(y_val)
    x_test_t = torch.FloatTensor(x_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 5. Build PyTorch DataLoaders
    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(x_train)}, Val samples: {len(x_val)}, Test samples: {len(x_test)}")
    return train_loader, val_loader, test_loader, scaler

# Example Usage:
# train_loader, val_loader, test_loader, scaler = get_dataloader('outputs/HCMC_Traffic_Data.npz')