# transformer_decoder/decoder.py
from .utils import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding
import torch.nn as nn
import math

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, vocab_size):
        super(DecoderLayer, self).__init__()

        # 自注意力层（Causal-Attention）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 编码器-解码器注意力层（Cross-Attention）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # 前馈网络（Feed-Forward）
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # LayerNorm 层（Layer Normalization）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):

        # 自注意力：输入为解码器自身的输入（x），q,k,v全部都是x
        attn_output, block1 = self.self_attn(x, x, x, tgt_mask)  # [bs, seq_len, emb] [bs, 1, seq_len, seq_len]
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和 LayerNorm

        # 交叉注意力：输入为编码器输出（enc_output）和解码器输入（x）
        attn_output, block2 = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))  # 残差连接和 LayerNorm

        # 前馈网络（Feed-Forward）
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # 残差连接和 LayerNorm

        return x, block1, block2


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, vocab_size, max_len=5000):
        super(Decoder, self).__init__()

        # 建立模型的 emb 层  创建一个形状为 (vocab_size, d_model) 的可训练参数矩阵
        # 初始化：嵌入矩阵的权重通常会在初始化时随机分配，或者从预训练模型中加载。
        # 训练：在训练过程中，嵌入矩阵的权重会通过反向传播进行更新，以优化模型性能。
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # 解码器层列表
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, vocab_size)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attention_weights = {}
        seq_len = x.size(1)

        # 添加嵌入和位置编码
        ## 嵌入的扩大：乘以嵌入维度的平方根 嵌入向量的初始值通常是从一个较小的标准差分布中随机抽取的，比如标准正态分布或均匀分布。如果直接将这些小数值输入到后续的线性变换、激活函数等操作中，可能会导致信号逐渐衰减，特别是当网络层数较多时。通过乘以嵌入维度的平方根，可以放大这些初始值，使得它们在整个网络中的传播更为稳定。
        # 在 Transformer 模型中，位置编码（positional encoding）会直接加到词嵌入（token embeddings）上。位置编码的设计通常是基于正弦和余弦函数，其幅度大致为 1。如果不调整词嵌入的规模，那么位置编码的影响可能会过大或过小，从而破坏了两者之间的相对比例。乘以 sqrt(d_model) 可以使词嵌入的均方根（RMS, Root Mean Square）与位置编码相近，维持两者在一个相似的数量级上，避免一方压倒另一方。
        # 这种做法源自 Vaswani 等人在他们的论文《Attention Is All You Need》中提出的建议。他们指出，这样的缩放有助于保持模型各部分的输出具有相似的方差，进而促进更有效的学习。
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        ## 遍历 self.dec_layers 列表中的每一层
        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, src_mask,
                                          tgt_mask)  # [bs, seq_len, emb] [bs, seq_len, emb], [bs, heads, 1, seq_len], [bs, 1, seq_len, seq_len]
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # 最后通过全连接层映射到词汇表大小
        final_output = self.fc_out(x)  # (batch_size, target_seq_len, vocab_size)

        return final_output, attention_weights