# gat_lstm_model_early_fusion.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
seed = 65
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class GCN_LSTM(nn.Module):
    def __init__(self, node_feature_dim, sequence_feature_dim, gcn_out_channels, lstm_hidden_dim, lstm_layers):
        super(GCN_LSTM, self).__init__()

        self.gcn = GCNConv(node_feature_dim, gcn_out_channels)
        self.gcn_dropout = nn.Dropout(0.2)

        # The input to LSTM is the sequence feature dimension + the output of the GAT layers
        combined_input_dim = sequence_feature_dim + gcn_out_channels
        self.lstm = nn.LSTM(combined_input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.3)

        # Fully connected layer for final prediction
        self.fc = nn.Linear(lstm_hidden_dim, 1)  # Output layer to predict target consumption

    def forward(self, sequences, edge_index, edge_attr, node_features, node_indices):

        # GAT layer 1-hop_1 neighbors
        gcn_out = self.gcn(node_features, edge_index)
        gcn_out = self.gcn_dropout(gcn_out)

        # Expand the GAT output to match the sequence length and concatenate with the sequence data
        gcn_out = gcn_out.unsqueeze(1).repeat(1, sequences.size(1), 1)  # Repeat for each time step
        combined_input = torch.cat((sequences, gcn_out), dim=-1)  # [batch_size, seq_len, seq_feat_dim + 2 * gat_out_channels]

        # LSTM layer processes the combined sequence and GAT data
        lstm_out, _ = self.lstm(combined_input)
        lstm_out = self.lstm_dropout(lstm_out)

        # Take the last output of LSTM
        lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]

        # Fully connected layer to predict the next hour's consumption
        out = self.fc(lstm_out)
        
        return out
