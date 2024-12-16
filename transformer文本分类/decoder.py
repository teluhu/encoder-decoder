from utils import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding
import torch.nn as nn
import math

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, vocab_size):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, block1 = self.self_attn(x, x, x, tgt_mask)  
        x = self.norm1(x + self.dropout(attn_output))  
        attn_output, block2 = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output)) 
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output)) 
        return x, block1, block2


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, vocab_size, nclass, max_len=5000):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, vocab_size)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, nclass) 

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attention_weights = {}
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        ## 遍历 self.dec_layers 列表中的每一层
        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, src_mask,
                                          tgt_mask) 
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2
        final_output = self.fc_out(x)  
        return final_output, attention_weights