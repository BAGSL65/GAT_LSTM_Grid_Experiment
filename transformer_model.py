import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不参与训练但会保存到模型参数

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        pe = self.pe[:x.size(0)]
        pe = pe.unsqueeze(1)
        return x + pe


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.d_model = 64
        self.seq_len = 24
        self.pred_len = 1
        self.nhead=4
        self.num_layers=3


        # 输入编码
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.dropout = nn.Dropout(0.1)

        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,  # 通常设置为d_model的4倍
            dropout=0.1,
            activation='gelu',  # 效果通常优于relu
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(self.d_model * self.seq_len, 4 * self.d_model),
            nn.ReLU(),
            nn.Linear(4 * self.d_model, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            output: Tensor of shape [batch_size, pred_len, output_dim]
        """
        # 输入投影和位置编码
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Transformer处理
        x = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]

        # 输出预测
        x = x.reshape(x.size(0), -1)  # [batch, seq_len * d_model]
        return self.fc(x)  # [batch, pred_len, output_dim]