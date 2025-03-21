import torch
import os
import numpy as np
import pandas as pd
import random
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import LSTM
from data_preprocessing_LSTM import preprocess_data, load_config


# 设置随机种子
seed = 65
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set up logging
logging.basicConfig(level=logging.INFO)

def evaluate_model(model, test_loader, target_scaler, test_time_indices, output_dir="outputs"):
    """Evaluate the model and save results."""
    model.eval()
    test_predictions, test_targets = [], []

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")
    evaluation_metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")

    with torch.no_grad():
        for i, (sequences, targets) in enumerate(test_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            # node_features = node_features_tensor[nodes]
            output = model(sequences)
            test_predictions.append(output.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
            # Extract time indices (hour of the day) based on batch index
            batch_size = targets.shape[0]
            batch_time_indices = test_time_indices[i * batch_size:(i + 1) * batch_size]

    # Concatenate results
    test_predictions = np.concatenate(test_predictions).squeeze()
    test_targets = np.concatenate(test_targets).squeeze()

    # Inverse transform predictions and targets to original scale
    test_predictions = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    test_targets = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_targets, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    smape = np.mean(2 * np.abs(test_targets, - test_predictions) / (np.abs(test_targets) + np.abs(test_predictions))) * 100
    r2 = r2_score(test_targets, test_predictions)
    corr_coef, _ = pearsonr(test_targets, test_predictions)

    # Save evaluation metrics
    with open(evaluation_metrics_path, "w") as f:
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
        f.write(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape}%\n")
        f.write(f"R-squared (R2): {r2}\n")
        f.write(f"Pearson Correlation Coefficient: {corr_coef}\n")
    logging.info(f"Saved evaluation metrics to {evaluation_metrics_path}")

    return test_targets, test_predictions, mae, rmse, smape, r2, corr_coef

# Main function to run testing
if __name__ == "__main__":
    # Load configuration and preprocess data
    config = load_config()
    train_seq, train_tgt, val_seq, val_tgt, test_seq, test_tgt, target_scaler = preprocess_data(config)

    # Extract time indices for test dataset
    test_time_indices = np.arange(len(test_tgt))  # Replace this with actual time indices if available

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(
        sequence_feature_dim=test_seq.shape[2],
        lstm_hidden_dim=128,
        lstm_layers=4,
    ).to(device)

    model_path = os.path.join(config['output_dir'], "lstm_model.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    logging.info(f"Loaded model from {model_path}")

    # Move tensors to device
    test_seq, test_tgt = test_seq.to(device), test_tgt.to(device)
    # Prepare DataLoader
    batch_size = 27
    test_dataset = TensorDataset(test_seq, test_tgt)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate the model
    output_dir = config['output_dir']  # Directory to save outputs
    test_targets, test_predictions, mae, rmse, smape, r2, corr_coef = evaluate_model(
        model=model,
        test_loader=test_loader,
        target_scaler=target_scaler,
        test_time_indices=test_time_indices,
        output_dir=output_dir
    )

    logging.info("Testing completed successfully.")
