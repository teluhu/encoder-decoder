# transformer 的 forward 函数运行：
 
## 提前的设置
### 参数设置
    vocab_size = tokenizer.vocab_size # tokenizer 的词汇表的大小
    d_model = 1536 # token 对应的 embedding 的维度
    nhead = 4 # 多头注意力头的数目
    num_decoder_layers = 2 # decoder 的层数
    dim_feedforward = 256 # 
    max_seq_length = 5 # 生成最多的 token 个数
    dropout = 0.1 # 训练时候的丢弃率
### 模型初始化
    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
    
    model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length,
            dropout=dropout
    )
## 传递参数
    output = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask) # [batch_size, seq_len, vocab_size]
**注意**：
一个 batch_size 的数据通过 tokenizer 分词后的格式问题

* src_input_ids：掩码多头注意力的输入 [batch_size, seq_len] 扩展为 [batch_size, seq_len, emb] 是在Decoder的forward 
* tgt_input_ids：encoder 的输出， 赋值是在 transformer 的 forward 函数中，生成 [batch_size, seq_len, emb] 
* src_mask：padding mask 用在交叉注意力上面，赋值在 test [batch_size, num_heads, 1, seq_len]
* tgt_mask：causal mask 用在自注意力上卖弄，赋值在 test [batch_size, 1, seq_len, seq_len]


## 运行过程
* test: src_input_ids, src_mask, tgt_mask 得到 transformer forward 的输出 
* Transformer: encoder_output, decoder_output 整合 encoder_output 到 decoder_output 得到 output 
* Decoder: embedding, pos_encoding, dec_layer 得到所有层的输出 
* DecoderLayer： MultiHeadAttention, PositionWiseFeedForward, LayerNorm. 构建 dec_layer 的每一层 得到 层 的输出

## 总结
batch_size != 1 的文本序列完成 decoder 的整个过程。

## 待完成
加入encoder部分的代码（已完成）