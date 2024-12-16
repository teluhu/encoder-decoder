import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):#维度是奇数也不会报错
    def __init__(self, d_model, max_seq_length):
        super().__init__()#继承父类nn.Module
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # 只考虑偶数位置，确保 div_term 的长度匹配
        div_term = torch.exp(2 * torch.arange(0, (d_model + 1) // 2).float()  * -(math.log(10000.0) / d_model))
        #其中2 * torch.arange(0, (d_model + 1) // 2).float()是2i
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2]) 
        self.register_buffer('pe', pe.unsqueeze(0))#pe是[batch_size, seq_length, d_model]，注册后会成为self.pe

    def forward(self, x):#这里的x是[batch_size, seq_length, d_model]
        return x + self.pe[:, :x.size(1)]#也可以尝试下除了相加的方式，但是感觉乘法的话就会有权重为0的可能性
    
class MultiHeadAttention(nn.Module):#x是[batch_size, seq_length, d_model]### 注意！这里的MultiHeadAttention不是完全体，因为没有加上mask！
    def __init__(self, d_model, num_heads):
        super().__init__()#继承父类nn.Module
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model//self.num_heads #因为要整除
        self.Qw = nn.Linear(d_model, d_model)#加载Q权重
        self.Kw = nn.Linear(d_model, d_model)#加载K权重
        self.Vw = nn.Linear(d_model, d_model)#加载V权重
        self.Ow = nn.Linear(d_model, d_model)#加载V权重
        
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    #把第一列和第二列交换，也就是seq_length和self.num_heads
    #变成了(batch_size, self.num_heads, seq_length, self.d_k)

    def concat_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(2,1).reshape(batch_size, seq_length, self.d_model)
    
    def dot_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)#首先是K转置，所以要交换最后两列，矩阵相乘
        #得到格式为[batch_size,num_heads,seq_length,seq_length]

        attn_probs = torch.softmax(attn_scores, dim=-1)#对最后一维做softmax
        output = torch.matmul(attn_probs, V)
        return output#得到了(batch_size, self.num_heads, seq_length, self.d_k)

    
    def forward(self,Q,K,V):#首先第一步是分头，把d_model分解成num_heads
        Qa=self.split_heads(self.Qw(Q))
        Ka=self.split_heads(self.Kw(K))
        Va=self.split_heads(self.Vw(V))
        attn_output = self.dot_attention(Qa, Ka, Va)
        output= self.Ow(self.concat_heads(attn_output))
        return output
        
class PositionWiseFeedForward(nn.Module): #之前想过在这里加上dropout，但是效果一般，所以这里改正常了
    def __init__(self, d_model,d_hidden):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_hidden)
        self.W2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.W1(x))
        return self.W2(x)
