import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class EEGConformer(nn.Module):
    def __init__(self, n_outputs, n_chans, n_times, n_filters_time=40, filter_time_length=25,
                 pool_time_length=75, pool_time_stride=15, drop_prob=0.5, att_depth=6, 
                 att_heads=10, att_drop_prob=0.5, final_fc_length=2440, return_features=False):
        super(EEGConformer, self).__init__()
        self.patch_embedding = PatchEmbedding(n_chans, n_filters_time, filter_time_length,
                                              pool_time_length, pool_time_stride, drop_prob)
        self.transformer_encoder = TransformerEncoder(n_filters_time, att_depth, att_heads, att_drop_prob)
        self.fc = FullyConnected(final_fc_length, n_outputs, return_features)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # Add one extra dimension
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, n_channels, n_filters_time, filter_time_length, pool_time_length, 
                 stride_avg_pool, drop_prob):
        super(PatchEmbedding, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(n_filters_time),
            nn.ELU(),
            nn.AvgPool2d((1, pool_time_length), stride=(1, stride_avg_pool)),
            nn.Dropout(drop_prob),
            nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
            Rearrange("b d_model 1 seq -> b seq d_model")
        )

    def forward(self, x):
        return self.conv_layers(x)

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, depth, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(emb_size, num_heads, dropout) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attention(x2)
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        Q = rearrange(self.queries
