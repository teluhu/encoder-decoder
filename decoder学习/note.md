encoder：
tokenizer是进行分词：得到input_ids，attention_mask。不同的分词器会得到不同的结果
tokenizer_config.json 分词配置文件


embedding是对分词映射到一个固定维度的稠密向量空间：
嵌入矩阵已知，
根据tokenizer分词后的结果到嵌入矩阵找对应向量表示，
添加位置编码到词嵌入上


model(**inputs) 这样的代码时，意味着正在将 inputs 字典中的每个键值对作为单独的关键字参数传递给 model 的调用方法。
model(**inputs)  等价于
model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])


decoder：自回归在训练的时候可能会用到教师强制（teacher forcing）
encoder和decoder的调用函数是不一样的


baseline：
数据的格式转换问题：
文件的格式：jsonl，json，parquet，csv
gb18030 与 gb2312：
GB2312：包含约 6763 个汉字和 682 个符号，主要用于简体中文。
GB18030：是 GB2312 的超集，包含更多的字符，包括繁体中文、少数民族文字等。它支持超过 27000 个汉字和大量其他字符。


ValueError: Expected input batch_size (511) to match target batch_size (1).
tokenizer 在处理输入文本时可能会产生不同的序列长度，特别是当使用 padding=True 时。如果输入文本和输出文本的长度差异很大，可能会导致批量大小不匹配。

面临的问题是输入可能会很长，但是输出很短
手动填充：padding='max_length',  # 确保填充到最大长度

除了数据的填充以外，还需要考虑传递的id和mask，也就是给模型传的是什么（看师兄的代码）
从 AutoTokenizer和AutoModelForCausalLM都是from_pretrained()可以看到是先加载模型
然后再使用模型，传递的参数是在使用的时候传递的


transformer:
多头注意力的思想是一个比较有意思的思想，多头注意力，self-consistency
多头注意力：拼接？？？
自注意力：被用来获取同一序列不同位置的依赖关系。
交叉注意力：这个多头注意力的query来自解码器第一个多头自注意力的输出，但是它的key和value来自解码器的输出memory。这种query、key和value不同源的注意力叫做交叉注意力。解码器使用交叉注意力来处理输入序列和输出序列之间的依赖关系。


推理需要有的是：
开始的token
结束的token是根据模型生成结束字符和达到最大的生成长度决定的

训练需要添加的是：训练数据集、criterion、optimizer

为什么最后输出的output是 torch.Size([2, 4, 151643])
因为：final_output = self.fc_out(x) 全连接映射到词汇表大小


generated_ids = self.decoder_model.generate(
                                        inputs_embeds=input_embs, 
                                        attention_mask=input_attn_mask,
                                        do_sample=False,
                                        max_new_tokens=max_new_tokens,
                                      )