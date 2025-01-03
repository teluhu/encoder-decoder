# 位置编码的公式

![{A3A4481B-4D2C-4C1B-A2B8-C9BF739E55A9}](https://github.com/user-attachments/assets/1c4bb270-e81c-4b4e-893b-e806d7b1ed31)

# 前置知识
在 PyTorch 中，当你调用一个模块实例时：**output = module(input)**
实际上是调用了模块的 `__call__`方法，而 `__call__` 方法会进一步调用 forward 方法：

`__call__`方法的机制：

```python
def __call__(self, *input, **kwargs):
    # 执行一些前置操作（如 hooks 等）
    return self.forward(*input, **kwargs)
```

因此，以下两种写法等价：

```python
output = module(input)  # 实际上调用了 module.forward(input)
output = module.forward(input)  # 直接调用 forward 方法
```

通常直接使用 module(input) 更为规范，因为它可以确保 PyTorch 的其他功能（如 hooks 或自动求导）正常工作。

# 实现流程

要把embedding层和position层相加，那么就需要`__init__`和forward两个部分。

## `__init__`部分

在`__init__`部分，首先创建一个位置编码（句子长度的格式，维度）格式

ps：因为位置编码是（句子长度的格式，维度），两个部分要相加，所以这里的位置编码矩阵也是同样的维度

之后创建位置index，从公式中可以看到，每个pos都要被计算，所以要生成max_length个pos

接着是缩放因子，计算如下

![{1A597B9F-C44D-4270-8113-56021EBBFDBF}](https://github.com/user-attachments/assets/0af1c02c-d76f-4753-9b5b-77f10de3a1a7)

然后计算每个pos的位置编码，奇数维度用sin，偶数维度用cos

最后注册pe，注册后，pe 成为 self 对象的一个属性，可以通过 self.pe 访问。pe 不会出现在 model.parameters() 中（即不会被优化）。但它会包含在 model.state_dict() 中，因此可以随模型一起保存和加载。


## `forward`部分

因为embedding的输出是[batch_size, length, dmodel]，所以叠加在一起的时候是x+pe[:,:x.size(1)]，【max_length>= length】，输出为[batch_size, length, dmodel]
