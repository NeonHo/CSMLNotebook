Word2Vec 的核心流程确实涉及 **将 One-Hot 编码后的词向量与可训练参数矩阵相乘，得到低维稠密表示，再通过上下文预测任务训练这些参数**。以下详细解释：

---

### **1. 从 One-Hot 到低维稠密向量的转换**

Word2Vec 中，每个词首先被表示为 One-Hot 向量（假设词汇表大小为 $V$，则向量长度为 $V$，仅对应词的位置为 1，其余为 0）。但 One-Hot 向量无法捕捉语义关系，因此需要通过一个 **可训练的参数矩阵 $W$** 将其映射到低维空间：

- **输入层 → 隐藏层**  
  设 One-Hot 向量为 $x$（维度 $V \times 1$），参数矩阵 $W$（维度 $N \times V$，$N$ 为目标向量维度，如 300），则低维向量 $h$ 为：  
  $$
  h = W \cdot x
  $$  
  由于 $x$ 中仅一位为 1，$h$ 等价于 $W$ 中对应词的行向量，即 **矩阵 $W$ 的每一行就是一个词的嵌入向量**。

- **几何意义**  
  One-Hot 向量是高维空间中的孤立点（彼此正交），而 $W$ 将这些点投影到 $N$ 维空间，使语义相关的词在低维空间中距离接近。

---

### **2. 训练参数矩阵 $W$ 的两种模型**

Word2Vec 通过 **Skip-gram** 和 **CBOW** 利用上下文信息训练 $W$。

#### **Skip-gram 模型**
- **目标**：已知中心词 $w_t$，预测上下文词 $w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$（窗口大小为 2）。
- **流程**  
  1. 中心词 $w_t$ 的 One-Hot 向量 $x$ 输入模型。  
  2. 通过 $W$ 得到低维向量 $h$（$w_t$ 的嵌入）。  
  3. $h$ 与输出矩阵 $W'$（维度 $V \times N$）相乘得预测向量 $y$（维度 $V \times 1$）。  
  4. softmax 将 $y$ 转为概率分布，与真实上下文词的 One-Hot 向量对比，计算交叉熵损失。  
  5. 反向传播更新 $W$ 和 $W'$，使模型更好预测上下文词。

#### **CBOW 模型**
- **目标**：已知上下文词 $w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$，预测中心词 $w_t$。
- **流程**  
  1. 所有上下文词的 One-Hot 向量分别经 $W$ 映射为低维向量。  
  2. 对这些向量取平均/求和得上下文表示 $h$。  
  3. $h$ 与 $W'$ 相乘预测中心词 $w_t$。  
  4. 同样通过损失函数和反向传播更新参数。

---

### **3. 优化训练效率的技巧**

#### **负采样（Negative Sampling）**
- **思想**：每次随机选 $k$ 个负样本（非上下文词），将多分类转为二分类。  
- **优势**：复杂度从 $O(V)$ 降至 $O(k)$，大幅提升速度。

#### **层次 Softmax（Hierarchical Softmax）**
- **思想**：用霍夫曼树组织词汇表，每个内部节点为二分类器。  
- **优势**：只需遍历 $\log V$ 个节点，适用于大词汇表。

---

### **4. 最终词向量的获取**

训练完成后：
- **矩阵 $W$**（输入层 → 隐藏层）：每行对应词的嵌入，常用作最终词向量。  
- **矩阵 $W'$**（隐藏层 → 输出层）：也可作为词向量，实际中常取 $W$ 或两者平均。

---

### **5. 代码示例（简化版 Skip-gram）**

```python
import numpy as np

# 词汇表大小 5，嵌入维度 3
vocab_size = 5
embedding_dim = 3

# 初始化参数
W = np.random.randn(embedding_dim, vocab_size)   # 3×5
W_prime = np.random.randn(vocab_size, embedding_dim)  # 5×3

# 中心词索引 0 → One-Hot 向量
center_idx = 0
x = np.zeros(vocab_size)
x[center_idx] = 1

# 1. One-Hot → 低维向量
h = np.dot(W, x)  # 等价于 W[:, center_idx]

# 2. 预测上下文词
y_pred = np.dot(W_prime, h)

# 3. softmax 概率
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
p = softmax(y_pred)

# 4. 交叉熵损失示例
target_idx = 1
target = np.zeros(vocab_size)
target[target_idx] = 1
loss = -np.sum(target * np.log(p))

# 5. 反向传播（省略梯度计算）
```

---

### **总结**

Word2Vec 的核心是  
**“One-Hot 向量 × 参数矩阵” → 低维向量**  
再通过 **上下文预测任务** 训练参数，使语义相关的词在低维空间中距离接近。该方法将离散词语转为连续语义向量，为后续 NLP 任务奠定基础。