# Scaled Dot Product Attention
![scaled dot product attention](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/scaled_dot_product_attn.svg)

# `d_model`
在 Transformer 模型中，\($d_{\text{model}}$\)（也称为模型的维度）是指输入和输出的嵌入维度，以及注意力机制中注意力权重的维度。它是模型中的关键参数之一。

具体来说，\($d_{\text{model}}$\) 表示输入和输出嵌入向量的维度，通常在 Transformer 的编码器和解码器中是相同的。在注意力机制中，每个注意力头产生的注意力权重的维度也等于 \($d_{\text{model}}$\)。

> 一个嵌入向量长

在 Transformer 模型的原始论文 "Attention is All You Need" 中，\($d_{\text{model}}$\) 是作为模型的一个超参数进行调整的。该值的选择影响了模型的表示能力和计算复杂性。通常，较大的 \($d_{\text{model}}$\) 值可以提高模型的表示能力，但会增加计算成本。

例如，在 PyTorch 中，如果使用 `nn.Transformer` 模块，可以通过参数 `d_model` 来设置 \($d_{\text{model}}$\) 的值。以下是一个简单的示例：

```python
import torch.nn as nn

# 设置 d_model 的值为 512
d_model = 512

# 创建 Transformer 模型
transformer_model = nn.Transformer(d_model=d_model, ...)
```

在上述示例中，`d_model` 被设置为 512，表示模型中嵌入向量和注意力权重的维度为 512。