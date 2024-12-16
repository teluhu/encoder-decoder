import torch.nn as nn
from utils import MultiHeadAttention, PositionWiseFeedForward
class EncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, d_hidden, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model,d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output,_ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.dropout(x)
        f_output = self.feed_forward(x)
        x = self.norm2(x + f_output)
        return x