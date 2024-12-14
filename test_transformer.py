# test_transformer.py
# 前向传播测试 测试 Transformer 的 forward 函数成功

import torch
from model import TransformerModel
from transformers import AutoTokenizer, AutoModel


# test Transformer's forward function
def simple_test_transformer_model():

    # tokenizer 目录路径
    local_tokenizer_dir = "./tokenizer_files/"  # 替换为你的实际路径
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        local_tokenizer_dir,
        trust_remote_code=True,  # 如果使用的分词器有自定义代码，需要启用此选项
        truncation_side='right',  # 设置分词器的截断侧
        padding_side='right'  # 设置分词器的填充侧
    )
    # 参数设置
    vocab_size = tokenizer.vocab_size
    d_model = 1536
    num_heads = 8
    num_decoder_layers = 2
    dim_feedforward = 256
    max_seq_length = 5
    dropout = 0.1
    batch_size = 2

    # 初始化模型
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    # batch_size = 2
    source_texts = ["Translate this sentence.", "Another example sentence, final example sentence."]

    # 分词
    tokenized_source = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)
    src_input_ids = tokenized_source["input_ids"]  # [batch, seq_len]
    src_attention_mask = tokenized_source["attention_mask"]

    ## 目标序列掩码（decoder self-attention） causal mask 不看当前位置之后
    seq_len = src_input_ids.size(1)
    # 下三角包括对角线全为1
    tgt_mask = torch.torch.tril(torch.ones(seq_len, seq_len))
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1).to(src_input_ids.device)  # [1, 1, seq_len, seq_len]
    tgt_mask = tgt_mask.expand(batch_size, -1, -1, -1)  # [batch_size, 1, seq_len, seq_len]

    ## padding mask 不看padding
    src_mask = src_attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    src_mask = src_mask.expand(-1, num_heads, -1, -1)  # [batch_size, num_heads, 1, seq_len]


    # print(f"src_input_ids_size: {src_input_ids.size()}")
    # print(f"tgt_mask_size: {tgt_mask.size()}")
    # print(f"src_mask: {src_mask}")
    # print(f"src_mask_size:{src_mask.size()}")

    # 前向传播测试 测试 Transformer 的 forward 函数成功
    # 第一个参数是decoder的input。第二个参数不重要。第三个参数是防止padding参与运算，第四个参数是自回归掩码
    output, attn_weights = model(src_input_ids, src_input_ids, src_mask, tgt_mask)
    print("Forward pass output shape:", output.size())  # 应为 (batch_size, seq_length, vocab_size)


# 运行测试函数
simple_test_transformer_model()
