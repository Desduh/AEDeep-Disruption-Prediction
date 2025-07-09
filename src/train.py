# train.py
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from model import PeakPredictorLSTM
from pmodel_generation import generate_and_prepare_series, create_sequences

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model, Loss, and Optimizer
    model = PeakPredictorLSTM(config['model']).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    print("Model Architecture:")
    print(model)
    
    # --- Training Loop ---
    model.train()
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        epoch_losses = []
        
        # Generate new data for each epoch by creating multiple series
        for series_num in range(config['training']['num_series_per_epoch']):
            # 1. Generate and prepare a new time series
            df = generate_and_prepare_series(
                length=config['p_model']['series_length_train'],
                p_value=config['p_model']['p_value'],
                peak_threshold=config['data']['peak_threshold'],
                norm_percentile=config['data']['normalization_percentile']
            )

            # 2. Create sequences from the series
            X_train, y_train = create_sequences(
                df,
                window_size=config['data']['window_size'],
                lookahead=config['data']['lookahead']
            )
            
            if len(X_train) == 0:
                print(f"Skipping series {series_num+1} due to insufficient length.")
                continue

            # 3. Create DataLoader
            dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
            dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
            
            # Progress bar for the current series
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {epoch+1}, Series {series_num+1}/{config['training']['num_series_per_epoch']}"
            )

            for X_batch, y_batch in progress_bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"End of Epoch {epoch+1}. Average Loss: {avg_epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), config['files']['model_save_path'])
    print(f"\nTraining complete. Model saved to {config['files']['model_save_path']}")


if __name__ == '__main__':
    main()