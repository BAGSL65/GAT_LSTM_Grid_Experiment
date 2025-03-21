import os
import xgboost as xgb
import logging
import matplotlib.pyplot as plt
from data_preprocessing_XGBoost import preprocess_data, load_config
# Set up logging
logging.basicConfig(level=logging.INFO)

def train_xgboost(train_seq, train_tgt, val_seq, val_tgt, output_dir="outputs"):
    """Train an XGBoost model."""
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(train_seq, label=train_tgt)
    dval = xgb.DMatrix(val_seq, label=val_tgt)
    params = {
        'objective': 'reg:squarederror',  # Regression task
        'max_depth': 6,  # Maximum depth of a tree
        'eta': 0.1,  # Learning rate
        'subsample': 0.8,  # Subsample ratio of the training instances
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
        'seed': 65,  # Random seed
        'eval_metric': 'rmse',  # Evaluation metric
        'tree_method': 'hist',  # Use GPU for training
        'device':'cuda'
    }
    evals_result = {}
    evals = [(dtrain, 'train'), (dval, 'val')]
    # Define XGBoost parameters

    # Train the model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        evals_result=evals_result
    )
    # Save the model
    model_save_path = os.path.join(output_dir, "xgboost_model.model")
    model.save_model(model_save_path)
    logging.info(f"Saved XGBoost model to {model_save_path}")

    # Plot loss curves
    train_losses = evals_result['train']['rmse']
    val_losses = evals_result['val']['rmse']
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('XGBoost Training and Validation Loss')
    plt.legend()
    plt.grid()
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    logging.info(f"Loss curve saved at {loss_curve_path}")

    return train_losses,val_losses

# Main function to run training
if __name__ == "__main__":
    # Load configuration and preprocess data
    config = load_config()
    train_seq, train_tgt,val_seq, val_tgt, test_seq, test_tgt,target_scaler,node_to_state = preprocess_data(config)
    # Train the model
    output_dir = config['output_dir']  # Directory to save outputs
    train_losses,val_losses = train_xgboost(train_seq, train_tgt, val_seq, val_tgt, output_dir)

    save_train_losses_path = os.path.join(output_dir, "XGBoost_train_losses.pkl")
    save_val_losses_path = os.path.join(output_dir, "XGBoost_val_losses.pkl")
    import pickle
    # 保存到文件
    with open(save_train_losses_path, 'wb') as f:
        pickle.dump(train_losses, f)
    with open(save_val_losses_path, 'wb') as f:
        pickle.dump(val_losses, f)

    logging.info("Training completed successfully.")
