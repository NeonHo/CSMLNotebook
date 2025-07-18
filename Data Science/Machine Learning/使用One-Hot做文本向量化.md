
文本向量化的经典流程，属于**离散表示**中的**One-Hot编码**实现方式。这一过程将原始文本转换为计算机可处理的数值向量，具体步骤如下：

---

### **1. Tokenization（分词）**

将文本分割为独立的“[[token]]”（通常是单词、子词或字符）。

**示例**  
- 原始文本：`"Hello world! This is a test."`  
- 分词后：`["Hello", "world", "!", "This", "is", "a", "test", "."]`

**实现细节**  
- 英文：按空格、标点分割，可能需要去除停用词（如“is”“a”）或进行词干提取（如“running”→“run”）。  
- 中文：需使用分词工具（如 jieba、THULAC），例如：  
  `"我爱自然语言处理"` → `["我", "爱", "自然语言", "处理"]`

---

### **2. 统计频率**

统计每个 token 在**训练语料库**中出现的次数。

**示例**  
假设语料库包含多个句子，统计后得到词频：

```python
{"the": 1000, "a": 800, "apple": 500, "computer": 300, "hello": 200, ...}
```

---

### **3. 频率排序**

按词频从高到低对所有 token 进行排序。

**示例**  
排序结果：`["the", "a", "apple", "computer", "hello", ...]`

---

### **4. 频率转 Index**

为每个 token 分配唯一的整数索引（通常从 0 或 1 开始）。

**示例**

```python
{"the": 0, "a": 1, "apple": 2, "computer": 3, "hello": 4, ...}
```

---

### **5. 保留前 10k 个词**

选择词频最高的前 10,000 个 token，丢弃低频词。低频词通常用特殊 token `<UNK>`（未知词）表示。

**示例**  
- 原始文本：`"I like to play football with my friends."`  
- 若 “football” 不在前 10k 中，则转换为：`"I like to play <UNK> with my friends."`

---

### **6. One-Hot 编码**
[[One-Hot编码]]
将每个 token 转换为长度为 10,000 的二进制向量，仅对应索引位置为 1，其余为 0。

**示例**  
- 词汇表大小为 10k，“apple” 的索引为 2 → `[0, 0, 1, 0, 0, ..., 0]`（仅第 3 位为 1）。  
- 句子编码：将每个 token 的 One-Hot 向量拼接或求和，例如：  
  `["apple", "is", "red"]` → 三个 One-Hot 向量的组合（维度：3 × 10,000）。

---

### **完整代码示例**

下面是一个完整实现上述流程的 Python 代码：

```python
from collections import Counter
import numpy as np

# 示例语料库
corpus = [
    "Hello world!",
    "This is a test.",
    "Hello Python!",
    "Python is great.",
    "Natural language processing is interesting."
]

# 1. Tokenization（分词）
def tokenize(corpus):
    tokens = []
    for sentence in corpus:
        # 简单分词：按空格分割，转小写，去除标点
        sentence_tokens = [word.strip('.,!?"\'').lower() for word in sentence.split()]
        tokens.extend(sentence_tokens)
    return tokens

all_tokens = tokenize(corpus)
print("分词结果:", all_tokens[:10])
# ['hello', 'world', 'this', 'is', 'a', 'test', 'hello', 'python', 'python', 'is']

# 2. 统计频率
word_freq = Counter(all_tokens)
print("词频统计:", dict(list(word_freq.items())[:3]))
# {'hello': 2, 'world': 1, 'this': 1}

# 3. 频率排序
sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
print("频率排序:", sorted_words[:3])
# [('is', 3), ('hello', 2), ('python', 2)]

# 4. 频率转 Index
vocab_size = 10000  # 假设保留前 10k 个词
vocab = {"<PAD>": 0, "<UNK>": 1}  # 预定义特殊 token
for i, (word, freq) in enumerate(sorted_words[:vocab_size - 2]):
    vocab[word] = i + 2  # 从 2 开始分配索引

print("词汇表:", dict(list(vocab.items())[:3]))
# {'<PAD>': 0, '<UNK>': 1, 'is': 2}

# 5. 文本转索引
def text_to_indices(sentence, vocab):
    tokens = [word.strip('.,!?"\'').lower() for word in sentence.split()]
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return indices

test_sentence = "Hello TensorFlow!"
indices = text_to_indices(test_sentence, vocab)
print("索引表示:", indices)
# [2, 1]（'hello' 存在，'tensorflow' 不存在 → <UNK>）

# 6. One-Hot 编码
def indices_to_onehot(indices, vocab_size):
    onehot_matrix = np.zeros((len(indices), vocab_size))
    for i, idx in enumerate(indices):
        onehot_matrix[i, idx] = 1
    return onehot_matrix

onehot = indices_to_onehot(indices, len(vocab))
print("One-Hot 形状:", onehot.shape)
# (2, 10)（假设词汇表大小为 10）
print("One-Hot 示例:", onehot)
```

---

### **关键细节与注意事项**

1. **词汇表大小**  
   - 保留前 10k 个词是常见做法，但需根据具体语料调整。例如，大规模语料可能需要 20k–50k，而小数据集可能只需 5k。

2. **特殊 token**  
   - `<PAD>`：用于填充句子至固定长度（如 Batch 训练时）。  
   - `<UNK>`：处理未登录词（不在词汇表中的词）。  
   - `<SOS>` / `<EOS>`：序列开始 / 结束标记（常用于机器翻译、生成任务）。

3. **计算效率**  
   - One-Hot 向量是高维稀疏矩阵（如 10k 维中仅 1 位为 1），实际应用中常转换为**稀疏矩阵**存储，或通过矩阵乘法直接转换为低维嵌入（如 [[Word2Vec]]）。

4. **局限性**  
   - 忽略语义：如 “apple” 和 “banana” 的向量正交，无法体现相关性。  
   - 维度灾难：词汇表增大时，计算和存储成本剧增。

---

### **应用场景**

- 简单文本分类（如垃圾邮件检测）。  
- 作为更复杂模型（如神经网络）的输入预处理。  
- 信息检索（如 TF-IDF 的基础）。  

这种方法是文本向量化的基础，理解其流程有助于掌握更高级的分布式表示（如 Word2Vec、BERT）。