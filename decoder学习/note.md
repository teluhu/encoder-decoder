# 一些零碎的笔记

encoder：
tokenizer是进行分词：得到input_ids，attention_mask。不同的分词器会得到不同的结果
embedding是对分词映射到一个固定维度的稠密向量空间：

model(**inputs) 这样的代码时，意味着正在将 inputs 字典中的每个键值对作为单独的关键字参数传递给 model 的调用方法。
model(**inputs)  等价于
model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

decoder：自回归在训练的时候可能会用到教师强制（teacher forcing）

推理需要有的是：
开始的token
结束的token是根据模型生成结束字符和达到最大的生成长度决定的