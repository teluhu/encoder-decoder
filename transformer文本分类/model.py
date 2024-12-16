from Encoder import EncoderLayer
from decoder import Decoder
import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_decoder_layers, dim_feedforward, max_seq_length,nclass, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = EncoderLayer(d_model, num_heads, dim_feedforward, dropout)# 编码器
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, dim_feedforward, dropout, vocab_size,nclass, max_seq_length) # 解码器层
    def forward(self, src_input_ids, encoder_outputs, src_mask, tgt_mask):
        # encoder_outputs = torch.randn(src_input_ids.size(0), src_input_ids.size(1), 1536)  # [batch_size, seq_len_src, d_model]
        # 编码器的输入2：调用编码器函数，输入是 [batch_size, seq_len, d_model]
        x = self.embedding(src_input_ids) * math.sqrt(self.embedding.embedding_dim)
        encoder_outputs = self.encoder(x)
        output, attn_weights = self.decoder(src_input_ids, encoder_outputs, src_mask, tgt_mask)
        return output, attn_weights

    def generate(self, start_token, max_len, src_input_ids, src_mask, tgt_mask):
        with torch.no_grad():
            # 确保 start_token 是整数类型，并初始化生成序列
            if not isinstance(start_token, int):
                raise ValueError("start_token must be an integer representing a valid token ID.")
            generated_sequence = [start_token]

            for i in range(max_len - 1):  # 减一因为已经包含了一个 start_token
                # 构造当前的目标序列张量，确保所有元素都是整数类型
                tgt_tensor = torch.tensor(generated_sequence, dtype=torch.long).unsqueeze(0).to(
                    next(self.parameters()).device)

                # 检查生成的 tgt_tensor 是否包含有效的 token 索引
                if tgt_tensor.max() >= self.embedding.num_embeddings:
                    raise ValueError(
                        f"Generated token index {tgt_tensor.max().item()} exceeds embedding size {self.embedding.num_embeddings}.")

                # 通过 forward 函数获取解码器输出
                output = self(src_input_ids, tgt_tensor, src_mask, tgt_mask)  # 使用 self(...) 而不是 self.forward(...)
                print(output.size())
                # 从输出中选择概率最大的 token ID，并确保它是整数类型
                next_token = int(output.argmax(dim=-1)[:, -1].item())

                # 将下一个 token 添加到生成序列中
                generated_sequence.append(next_token)

                # 如果遇到了结束标记，则停止生成
                if next_token == self.tokenizer.eos_token_id:
                    break

            return generated_sequence

    @classmethod
    def generate_square_subsequent_mask(cls, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask