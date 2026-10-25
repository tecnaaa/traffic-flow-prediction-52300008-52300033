import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

# Import modules from your project
from scripts.data_loader import get_dataloader
from models.stgcn import STGCN
from utils.metrics import masked_mae, masked_rmse, masked_mape

# Hyperparameters Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEQ_LEN = 12  # Past 60 minutes (12 steps x 5 mins)
PRE_LEN = 12  # Future 60 minutes prediction

def main():
    print(f"Starting training on device: {DEVICE}")
    
    # 1. Load Adjacency Matrix
    adj_matrix = np.load('outputs/adj_matrix.npy')
    adj = torch.FloatTensor(adj_matrix).to(DEVICE)
    num_nodes = adj_matrix.shape[0]
    
    # 2. Initialize DataLoaders
    train_loader, val_loader, test_loader, scaler = get_dataloader(
        'outputs/HCMC_Traffic_Data.npz', 
        seq_len=SEQ_LEN, pre_len=PRE_LEN, batch_size=BATCH_SIZE
    )
    
    # 3. Initialize Model and Optimizer
    model = STGCN(num_nodes=num_nodes, in_dim=1, out_dim=PRE_LEN, seq_len=SEQ_LEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Use MSE for training loss calculation
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE) 
            batch_y = batch_y.to(DEVICE) 
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_x, adj) # output shape: (Batch, Nodes, PRE_LEN)
            
            # Reshape label (extract speed feature F=0) to match model output shape
            truth = batch_y[..., 0].permute(0, 2, 1) # truth shape: (Batch, Nodes, PRE_LEN)
            
            # Compute loss and backpropagation
            loss = criterion(output, truth)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Time: {epoch_time:.2f}s")
        
    # 5. Evaluation on Test Set
    print("\nEvaluating model on the Test set...")
    model.eval()
    preds, truths = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x, adj)
            
            # Inverse transform back to real km/h values for error calculation
            pred = scaler.inverse_transform(output.cpu().numpy())
            truth = scaler.inverse_transform(batch_y[..., 0].permute(0, 2, 1).cpu().numpy())
            
            preds.append(pred)
            truths.append(truth)
            
    # Concatenate all batches
    preds = torch.FloatTensor(np.concatenate(preds, axis=0))
    truths = torch.FloatTensor(np.concatenate(truths, axis=0))
    
    # Compute core evaluation metrics
    mae = masked_mae(preds, truths).item()
    rmse = masked_rmse(preds, truths).item()
    mape = masked_mape(preds, truths).item()
    
    print("\n=== EVALUATION RESULTS (BASELINE STGCN) ===")
    print(f"MAE:  {mae:.4f} (Mean Absolute Error in km/h)")
    print(f"RMSE: {rmse:.4f} (Root Mean Square Error)")
    print(f"MAPE: {mape:.4f}% (Mean Absolute Percentage Error)")

if __name__ == "__main__":
    main()