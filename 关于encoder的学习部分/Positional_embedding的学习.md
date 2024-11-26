# 位置编码的公式

<img src="C:\Users\16864\AppData\Roaming\Typora\typora-user-images\image-20241126112658932.png" alt="image-20241126112658932" style="zoom: 80%;" />

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

<img src="C:\Users\16864\AppData\Roaming\Typora\typora-user-images\image-20241126113657471.png" alt="image-20241126113657471" style="zoom:67%;" />

最后计算每个pos的位置编码，奇数维度用sin，偶数维度用cos



## `forward`部分

因为embedding的输出是[batch_size, length, dmodel]，所以叠加在一起的时候是x+pe[:,:x.size(1)]，【max_length>= length】，输出为[batch_size, length, dmodel]