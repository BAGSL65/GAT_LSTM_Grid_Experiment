import torch
import os
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb
from data_preprocessing_XGBoost import preprocess_data, load_config, create_graph, load_and_prepare_data

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


def evaluate_xgboost(model, test_seq, test_tgt,target_scaler, output_dir="outputs",):
    """Evaluate the XGBoost model and save results."""
    # Convert data to DMatrix format
    dtest = xgb.DMatrix(test_seq, label=test_tgt)
    # Make predictions
    test_predictions = model.predict(dtest)
    # Inverse transform predictions and targets to original scale
    test_predictions = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    test_tgt = target_scaler.inverse_transform(test_tgt.reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_tgt, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_tgt, test_predictions))
    smape = np.mean(2 * np.abs(test_tgt - test_predictions) / (np.abs(test_tgt) + np.abs(test_predictions))) * 100
    r2 = r2_score(test_tgt, test_predictions)
    corr_coef, _ = pearsonr(test_tgt, test_predictions)
    # Save evaluation metrics
    evaluation_metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(evaluation_metrics_path, "w") as f:
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
        f.write(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape}%\n")
        f.write(f"R-squared (R2): {r2}\n")
        f.write(f"Pearson Correlation Coefficient: {corr_coef}\n")
    logging.info(f"Saved evaluation metrics to {evaluation_metrics_path}")

    return test_tgt, test_predictions, mae, rmse, smape, r2, corr_coef


# Main function to run testing
if __name__ == "__main__":
    # Load configuration and preprocess data
    config = load_config()
    train_seq, train_tgt, val_seq, val_tgt, test_seq, test_tgt, target_scaler, node_to_state = preprocess_data(
        config)

    # Load the model

    model_path = os.path.join(config['output_dir'], "xgboost_model.model")
    model = xgb.Booster()
    model.load_model(model_path)
    logging.info(f"Loaded model from {model_path}")

    # Evaluate the model
    output_dir = config['output_dir']  # Directory to save outputs
    # Evaluate the model
    test_tgt, test_predictions, mae, rmse, smape, r2, corr_coef = evaluate_xgboost(model, test_seq, test_tgt,target_scaler, output_dir)

    logging.info("Testing completed successfully.")
