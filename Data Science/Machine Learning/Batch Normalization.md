
### Batch Normalization（批量归一化）：加速深度学习训练的利器

Batch Normalization（BN）由 **Sergey Ioffe** 和 **Christian Szegedy** 于 2015 年提出，通过对神经网络中间层的输入进行归一化，显著提升训练速度和模型稳定性，已成为现代深度学习架构的标配。

---

### 一、核心思想与作用

#### 1. 解决的问题：Internal Covariate Shift  
- **ICS 现象**：前层参数更新导致后层输入分布不断变化，训练效率低。  
- **BN 目标**：让每层输入分布保持稳定，加速收敛。

#### 2. 带来的优势  
- 允许更大的学习率，缓解梯度消失/爆炸。  
- 降低对初始化的敏感度。  
- 正则化效果（部分替代 Dropout）。  
- 提高模型泛化能力。

---

### 二、数学原理与算法流程

#### 1. 归一化公式  
给定批次输入 \( $\mathbf{x} = \{x_1,\dots,x_m\}$ \)，BN 三步计算：

1. **批次均值与方差**  
   $$
   \mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i,\quad
   \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_B)^2
   $$

2. **标准化**  
   $$
   \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
   $$

3. **可学习缩放与偏移**  
   $$
   y_i = \gamma \hat{x}_i + \beta
   $$  
   其中 \( $\gamma, \beta$ \) 为可训练参数，恢复表达能力。

#### 2. 应用位置  
- 卷积层：按通道独立归一化（BatchNorm2d）。  
- 全连接层：按神经元归一化（BatchNorm1d）。  
- 通常放在 **激活函数之前**。

---

### 三、训练 vs 推理

| 阶段     | 统计来源                           | 输出公式                                                                                              |
| ------ | ------------------------------ | ------------------------------------------------------------------------------------------------- |
| **训练** | 当前批次 \( $\mu_B, \sigma_B^2$ \) | 按上文三步计算                                                                                           |
| **推理** | 训练时维护的滑动平均                     | \( $y = \gamma \frac{x - \text{running\_mean}}{\sqrt{\text{running\_var} + \epsilon}} + \beta$ \) |

滑动平均更新：  
$$
\text{running\_mean} \leftarrow \alpha \cdot \text{running\_mean} + (1-\alpha)\mu_B  
$$
$$
\text{running\_var} \leftarrow \alpha \cdot \text{running\_var} + (1-\alpha)\sigma_B^2
$$  
\( $\alpha$ \) 通常取 0.9/0.99。

---

### 四、PyTorch 代码示例

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.bn1(self.fc1(x))
        x = nn.ReLU()(x)
        x = self.bn2(self.fc2(x))
        x = nn.ReLU()(x)
        return self.fc3(x)

model = Net()
model.train()   # 训练模式（使用批次统计）
model.eval()    # 推理模式（使用滑动平均）
```

---

### 五、归一化家族

| 方法 | 归一化维度 | 适用场景 |
|---|---|---|
| **BatchNorm** | 批次维度 | 图像、大 batch |
| **LayerNorm** | 样本维度（整层） | NLP、RNN |
| **InstanceNorm** | 样本+通道 | 风格迁移 |
| **GroupNorm** | 通道分组 | 小 batch |
| **SwitchableNorm** | 学习权重融合 | 自动选择 |

---

### 六、使用技巧

- **BatchNorm + Dropout**：可能冲突；现代网络常用 BN 单独正则化。  
- **小 batch**（<16）：BN 统计不稳定，可改用 GroupNorm。  
- **迁移学习**：微调时冻结部分 BN 参数需重新估计统计量。  
- **RNN**：一般使用 LayerNorm；直接 BN 效果不佳。

---

### 七、总结

Batch Normalization 通过归一化中间层输入，显著缓解了 Internal Covariate Shift，使网络训练更快、更稳。尽管出现众多变体，BN 仍是图像任务的首选归一化技术。掌握其原理与使用细节，是构建高性能模型的关键一步。