import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # 定义线性变换层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(self.head_dim)
    def split_head(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return x
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 线性变换
        q = self.q_linear(q)  # (batch_size, seq_len_q, hidden_size)  对 emb 的值或者维度做了改变
        k = self.k_linear(k)  # (batch_size, seq_len_k, hidden_size)
        v = self.v_linear(v)  # (batch_size, seq_len_v, hidden_size)
        # 分割头
        q = self.split_head(q, batch_size)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = self.split_head(k, batch_size)  # (batch_size, num_heads, seq_len_k, head_dim)
        v = self.split_head(v, batch_size)  # (batch_size, num_heads, seq_len_v, head_dim)

        # 每头独立计算
        ## 多头计算注意力分数 QK^T
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            # scores 张量中对应于 mask 中为0的位置（即不应该被关注的位置），将其值设置为 -1e9
            # 这样-1e9在经过softmax以后得到的概率接近于零
            # 实现了 mask 中掩码为0的位置是没有score分数的
            scores = scores.masked_fill(mask == 0, -1e9)
        ## 减少计算使用过设置为负无穷大，然后负无穷大的对应函数值为0，0不参与计算实现的
        ## softmax 作用与最后一维，负无穷大对应的是 0，对应到 v 就不会有最后向量
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        ## 应用注意力权重到值向量
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, head_dim)
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_len_q, hidden_size)
        # (batch_size, num_heads, seq_len, head_dim) 变化 transpose(1, 2)
        # (batch_size, seq_len, num_heads, head_dim)
        # view 之前用 contiguous() 是为了确保张量在内存中的布局是连续的
        # view 进行重塑，将 num_heads 和 head_dim 合并在一起
        # 最后一个线性变换
        output = self.o_linear(output)  # (batch_size, seq_len_q, hidden_size)
        return output, attention_weights
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model,d_hidden):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_hidden)
        self.W2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.W1(x))
        return self.W2(x)

class PositionalEncoding(nn.Module):#维度是奇数也不会报错
    def __init__(self, d_model, max_len):
        super().__init__()#继承父类nn.Module
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 只考虑偶数位置，确保 div_term 的长度匹配
        div_term = torch.exp(2 * torch.arange(0, (d_model + 1) // 2).float()  * -(math.log(10000.0) / d_model))
        #其中2 * torch.arange(0, (d_model + 1) // 2).float()是2i
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2]) 
        self.register_buffer('pe', pe.unsqueeze(0))#pe是[batch_size, seq_length, d_model]，注册后会成为self.pe
    def forward(self, x):#这里的x是[batch_size, seq_length, d_model]
        return x + self.pe[:, :x.size(1)]