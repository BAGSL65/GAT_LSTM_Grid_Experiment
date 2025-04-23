import torch
import torch.nn as nn

class SpatioTemporalFusion(nn.Module):
    def __init__(self, gat_dim, time_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        # 投影层确保维度匹配
        self.gat_proj = nn.Linear(gat_dim, time_dim)

        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=time_dim,
            num_heads=5,
            batch_first=True
        )
        
    def forward(self, gat_feat, time_feat):
        """
        Args:
            gat_feat: [B, T, D_gat]
            time_feat: [B, T, D_time]
        Returns:
            [B, T, 2*D_time] 
        """
        # 投影GAT特征
        gat_feat = self.gat_proj(gat_feat)  # [B, D_time]
        
        # 交叉注意力
        attn_out, _ = self.cross_attn(
            query=time_feat,  # 用时间特征作为query
            key=gat_feat,  # [B, 1, D_time]
            value=gat_feat
        )  # [B, T, D_time]
        
        # 残差连接+特征拼接
        enhanced_time = time_feat + attn_out  # [B, T, D_time]
        
        return torch.cat([enhanced_time, gat_feat], dim=-1)  # [B, T, 2*D_time]