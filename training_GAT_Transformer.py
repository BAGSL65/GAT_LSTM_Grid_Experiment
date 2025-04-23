import torch
import os
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from GAT_Transformer_model import GAT_Transformer
from data_preprocessing_GAT_Transformer import preprocess_data, load_config
# Set up logging
logging.basicConfig(level=logging.INFO)
# Set random seed
seed = 65
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=10, patience=10, output_dir="outputs"):
    """Train the model and save the results."""
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    trigger_times = 0

    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "gat_transformer_model.pth")
    training_log_path = os.path.join(output_dir, "training_log.txt")

    # Open a log file to write training progress
    with open(training_log_path, 'w') as log_file:
        log_file.write("Epoch, Train Loss, Validation Loss\n")

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for sequences, targets, nodes in train_loader:
                sequences, targets, nodes = sequences.to(device), targets.to(device), nodes.to(device)
                optimizer.zero_grad()
                # node_features = node_features_tensor[nodes]
                output = model(sequences, edge_index_tensor, edge_attr_tensor, node_features_tensor, nodes)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets, nodes in val_loader:
                    sequences, targets, nodes = sequences.to(device), targets.to(device), nodes.to(device)
                    # node_features = node_features_tensor[nodes]
                    output = model(sequences, edge_index_tensor, edge_attr_tensor, node_features_tensor, nodes)
                    loss = criterion(output, targets)
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            val_losses.append(val_loss)


            # Log results for this epoch
            log_msg = (f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
            logging.info(log_msg)
            log_file.write(f"{epoch+1},{train_loss},{val_loss}\n")
            
            # Adjust learning rate
            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()
            print(f"Current Learning Rate: {current_lr}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Saved best model at {model_save_path}")
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logging.info("Early stopping triggered!")
                    break

    # Save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAT-Transformer Training and Validation Loss')
    plt.legend()
    plt.grid()
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    logging.info(f"Loss curve saved at {loss_curve_path}")

    return train_losses, val_losses

# Main function to run training
if __name__ == "__main__":
    # Load configuration and preprocess data
    config = load_config()
    train_seq, train_tgt, train_nodes, val_seq, val_tgt, val_nodes, test_seq, test_tgt, test_nodes, node_features_tensor, edge_index_tensor, edge_attr_tensor, target_scaler,node_to_state = preprocess_data(config)

    # Move tensors to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_features_tensor = node_features_tensor.to(device)
    edge_index_tensor = edge_index_tensor.to(device)
    edge_attr_tensor = edge_attr_tensor.to(device)

    # Prepare DataLoader
    batch_size = 1024
    train_dataset = TensorDataset(train_seq, train_tgt, train_nodes)
    val_dataset = TensorDataset(val_seq, val_tgt, val_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                              persistent_workers=True 
                              )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            persistent_workers=True
                            )

    # Model parameters
    sequence_feature_dim = train_seq.shape[2]
    node_feature_dim = node_features_tensor.shape[1]
    gat_out_channels = 64
    gat_heads = 8
    edge_dim = edge_attr_tensor.shape[1]
    d_model = 256
    # Initialize model
    model = GAT_Transformer(node_feature_dim, sequence_feature_dim, config['sequence_length'], gat_out_channels, gat_heads, edge_dim,d_model).to(device)

    # Define optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15,min_lr=1e-7)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    criterion = torch.nn.MSELoss()
    # Train the model
    output_dir = config['output_dir']  # Directory to save outputs
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=200,
        patience=20,
        output_dir=output_dir
    )
    save_train_losses_path = os.path.join(output_dir, "GAT_Transformer_train_losses.pkl")
    save_val_losses_path = os.path.join(output_dir, "GAT_Transformer_val_losses.pkl")
    import pickle
    # 保存到文件

    with open(save_train_losses_path, 'wb') as f:
        pickle.dump(train_losses, f)
    with open(save_val_losses_path, 'wb') as f:
        pickle.dump(val_losses, f)

    logging.info("Training completed successfully.")
