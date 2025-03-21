import torch
import torch.nn as nn
seed = 65
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class LSTM(nn.Module):
    def __init__(self,sequence_feature_dim,lstm_hidden_dim, lstm_layers):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(sequence_feature_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden_dim, 1)  # Output layer to predict target consumption

    def forward(self, sequences):
        # LSTM layer processes the combined sequence and GAT data
        lstm_out, _ = self.lstm(sequences)
        lstm_out = self.lstm_dropout(lstm_out)

        # Take the last output of LSTM
        lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]

        # Fully connected layer to predict the next hour's consumption
        out = self.fc(lstm_out)
        
        return out
