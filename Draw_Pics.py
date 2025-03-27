
import os
import logging
import pickle

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
output_dir= "outputs/Pics"
read_dir= "outputs/Transformer_Plots"
save_train_losses_path = os.path.join(read_dir, "Transformer_train_losses.pkl")
save_val_losses_path = os.path.join(read_dir, "Transformer_val_losses.pkl")


with open(save_train_losses_path, 'rb') as f:
    train_losses=pickle.load(f)

with open(save_val_losses_path, 'rb') as f:
    val_losses=pickle.load(f)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Loss',fontsize=18)
plt.tick_params(axis='x', labelsize=15)     # x轴刻度
plt.tick_params(axis='y', labelsize=15)     # y轴刻度
plt.title('Transformer Training and Validation Loss',fontsize=18)
plt.legend(fontsize=18)
plt.grid()
loss_curve_path = os.path.join(output_dir, "Transformer_loss_curve.png")
plt.savefig(loss_curve_path, dpi=300)
plt.close()
logging.info(f"Loss curve saved at {loss_curve_path}")