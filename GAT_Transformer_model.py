# gat_lstm_model_early_fusion.py
import torch
import torch.nn as nn
from ST_Fusion import SpatioTemporalFusion
from torch_geometric.nn import GATConv
from transformer_model import TimeSeriesTransformer

seed = 65
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GAT_Transformer(nn.Module):
    def __init__(self, node_feature_dim, sequence_feature_dim, seq_len, gat_out_channels, gat_heads, edge_dim, d_model):
        super().__init__()
        # Edge attribute conditioning
        self.edge_attr_transform = nn.Linear(edge_dim, gat_out_channels)

        # Parallel GAT layers
        self.gat_1hop_1 = GATConv(node_feature_dim, gat_out_channels, heads=gat_heads, concat=False,
                                  edge_dim=gat_out_channels)
        
        self.gat_1hop_2 = GATConv(node_feature_dim, gat_out_channels, heads=gat_heads, concat=False,
                                  edge_dim=gat_out_channels)
        
        self.relu = nn.ReLU()
        
        self.gat_dropout = nn.Dropout(0.2)

        # self.SpatioTemporalFusion = SpatioTemporalFusion(2*gat_out_channels, sequence_feature_dim,seq_len)
        self.fusion = nn.Sequential(
            nn.Linear(2*gat_out_channels + sequence_feature_dim, d_model),
            # nn.LayerNorm(d_model),
            # nn.Dropout(0.2),
            
        )
        
        self.transformer = TimeSeriesTransformer(d_model, d_model, seq_len)
        # self.transformer = TimeSeriesTransformer(d_model, 2*sequence_feature_dim, seq_len)
        self.output = nn.Sequential(
            # nn.Linear(d_model, d_model//2),
            # nn.LayerNorm(d_model),
            # nn.Dropout(0.2),
            # nn.Linear(d_model//2, 1),
            
            # nn.Dropout(0.3),
            nn.Linear(d_model, 1)
        )

    def forward(self, sequences, edge_index, edge_attr, node_features, node_indices):
        # Edge attribute conditioning
        transformed_edge_attr = self.edge_attr_transform(edge_attr)

        # GAT layer 1-hop_1 neighbors
        gat_1hop_out_1 = self.gat_1hop_1(node_features, edge_index, transformed_edge_attr)
        gat_1hop_out_1 = self.relu(gat_1hop_out_1)
        gat_1hop_out_1 = self.gat_dropout(gat_1hop_out_1)

        # GAT layer 1-hop_2 neighbors
        gat_1hop_out_2 = self.gat_1hop_2(node_features, edge_index, transformed_edge_attr)
        gat_1hop_out_2 = self.relu(gat_1hop_out_2)
        gat_1hop_out_2 = self.gat_dropout(gat_1hop_out_2)

        # Gather GAT output for each sequence's node index (both 1-hop and 2-hop)
        gat_1hop_out_1 = gat_1hop_out_1[node_indices]
        gat_1hop_out_2 = gat_1hop_out_2[node_indices]

        gat_combined_out = torch.cat((gat_1hop_out_1, gat_1hop_out_2), dim=-1)  # [batch_size, 2 * gat_out_channels]
        gat_combined_out = gat_combined_out.unsqueeze(1).repeat(1, sequences.size(1), 1)
        # gat_combined_out = gat_combined_out.unsqueeze(1).repeat(1, sequences.size(1), 1)  # Repeat for each time step
        # combined_input = torch.cat((sequences, gat_combined_out), dim=-1)  # [batch_size, seq_len, seq_feat_dim + 2 * gat_out_channels]
        
        # combined_input = self.SpatioTemporalFusion(gat_combined_out,sequences)
        combined_input = self.fusion(torch.cat([
            gat_combined_out, 
            sequences
        ], dim=-1))
        
        # Transformer处理
        output = self.transformer(combined_input)
        return self.output(output)
