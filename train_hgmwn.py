import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

from scripts.data_loader import get_dataloader
from models.hgmwn import HGMWN
from utils.metrics import masked_mae, masked_rmse, masked_mape

# Hyperparameters Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEQ_LEN = 12
PRE_LEN = 12

def main():
    print(f"Starting Proposed HGMWN Training on device: {DEVICE}")
    
    # 1. Load Data-driven Adjacency matrix
    adj_data = np.load('outputs/adj_matrix.npy')
    adj_data_t = torch.FloatTensor(adj_data).to(DEVICE)
    num_nodes = adj_data.shape[0]
    
    # 2. Load Physical Adjacency matrix (Replacing fallback identity matrix)
    adj_physical = np.load('outputs/physical_adj_matrix.npy')
    adj_physical_t = torch.FloatTensor(adj_physical).to(DEVICE)
    
    # 3. Initialize DataLoaders
    train_loader, val_loader, test_loader, scaler = get_dataloader(
        'outputs/HCMC_Traffic_Data.npz', 
        seq_len=SEQ_LEN, pre_len=PRE_LEN, batch_size=BATCH_SIZE
    )
    
    # 4. Initialize Proposed Model and Optimizer
    model = HGMWN(num_nodes=num_nodes, in_dim=1, out_dim=PRE_LEN, seq_len=SEQ_LEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE) 
            batch_y = batch_y.to(DEVICE) 
            
            optimizer.zero_grad()
            
            # Forward pass with dual graphs
            output = model(batch_x, adj_physical_t, adj_data_t)
            
            # Reshape label (extract speed feature F=0)
            truth = batch_y[..., 0].permute(0, 2, 1)
            
            loss = criterion(output, truth)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Time: {epoch_time:.2f}s")
        
    # 6. Evaluation on Test Set
    print("\nEvaluating Proposed HGMWN model on Test set...")
    model.eval()
    preds, truths = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x, adj_physical_t, adj_data_t)
            
            # Inverse transform back to real values
            pred = scaler.inverse_transform(output.cpu().numpy())
            truth = scaler.inverse_transform(batch_y[..., 0].permute(0, 2, 1).cpu().numpy())
            
            preds.append(pred)
            truths.append(truth)
            
    # Concatenate results
    preds = torch.FloatTensor(np.concatenate(preds, axis=0))
    truths = torch.FloatTensor(np.concatenate(truths, axis=0))
    
    # Compute metrics
    mae = masked_mae(preds, truths).item()
    rmse = masked_rmse(preds, truths).item()
    mape = masked_mape(preds, truths).item()
    
    print("\n=== EVALUATION RESULTS (PROPOSED HGMWN) ===")
    print(f"MAE:  {mae:.4f} (Mean Absolute Error)")
    print(f"RMSE: {rmse:.4f} (Root Mean Square Error)")
    print(f"MAPE: {mape:.4f}% (Mean Absolute Percentage Error)")

if __name__ == "__main__":
    main()