import torch.nn as nn
from utils import MultiHeadAttention, FeedForward
class EncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, d_hidden,is_drop = True, drop = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model,d_hidden,is_drop, drop)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        f_output = self.feed_forward(x)
        x = self.norm2(x + f_output)
        return x