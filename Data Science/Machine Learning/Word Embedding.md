**Word Embedding（词嵌入）** 是自然语言处理（NLP）中至关重要的技术，其核心是将文本中的词转换为**低维稠密向量**，并使语义相似的词在向量空间中距离相近。与传统的 One-Hot 编码（高维稀疏、语义无关）不同，Word Embedding 能够捕捉词之间的语义关联（如“国王”与“王后”、“苹果”与“香蕉”的向量接近），极大提升了模型对语言的理解能力。

---

### **一、Word Embedding 的核心思想**

1. **分布式表示（Distributed Representation）**  
   每个词由多个维度的实数向量表示（通常 50–300 维），向量的每个维度捕获词的不同语义特征（如词性、主题、情感等）。  
   - “苹果”的向量可能在“水果”维度上值较高，在“电子产品”维度上值较低；  
   - “iPhone”则相反，在“电子产品”维度上值较高。

2. **语义相似度**  
   通过向量间距离（如余弦相似度）衡量词的语义相似性：  
   - $\text{余弦相似度}(\text{“猫”}, \text{“狗”}) \approx 0.8$（语义相近）  
   - $\text{余弦相似度}(\text{“猫”}, \text{“电脑”}) \approx 0.1$（语义无关）

3. **词向量的代数运算**  
   体现语义类比关系：  
   - $\text{“国王”} - \text{“男人”} + \text{“女人”} \approx \text{“王后”}$  
   - $\text{“北京”} - \text{“中国”} + \text{“美国”} \approx \text{“华盛顿”}$

---

### **二、主流 Word Embedding 方法**

#### 1. 静态词嵌入（Static Word Embedding）  
为每个词分配固定向量，不考虑词在上下文中的歧义。

- **Word2Vec（2013）**  
  Google 开发的经典模型，通过预测上下文学习词向量，含两种训练模式：  
  - **CBOW（Continuous Bag-of-Words）**：用上下文词预测中心词。  
  - **Skip-gram**：用中心词预测上下文词。  

- **GloVe（2014）**  
  结合全局统计信息（词共现矩阵）和局部上下文，通过最小化“词共现概率的对数差”学习向量，语义类比任务表现更优。  

- **FastText（2016）**  
  Facebook 扩展模型，将词拆分为子词（如 “apple” → “ap”“pp”“ple”），向量为子词加权和，解决稀有词和未登录词问题。

#### 2. 动态词嵌入（Contextual Word Embedding）  
根据上下文动态生成向量，解决一词多义。

- **ELMo（2018）**  
  双向 LSTM 预训练语言模型，词向量是上下文动态表示。  

- **BERT（2018）**  
  基于 Transformer 的双向预训练模型，使用 MLM 与 NSP 任务，上下文敏感。  

- **GPT 系列**  
  OpenAI 单向自回归预训练模型，擅长生成，也可提取上下文相关向量。

---

### **三、Word Embedding 的训练与应用**

#### 1. 训练方式  
- **无监督学习**：大规模无标注文本（Word2Vec、GloVe）。  
- **预训练 + 微调**：先预训练，再在特定任务上微调（BERT、GPT）。

#### 2. 应用场景  
- 文本分类  
- 语义相似度计算  
- 机器翻译  
- 命名实体识别（NER）  
- 问答系统

---

### **四、Word Embedding vs. 传统文本表示**

| 特性             | One-Hot / TF-IDF          | Word Embedding             |
|------------------|---------------------------|----------------------------|
| **向量维度**     | 高维（词汇表大小）        | 低维（50–300）             |
| **向量类型**     | 稀疏（仅一位为 1）        | 稠密（所有维度有值）       |
| **语义捕捉**     | 无（词间正交）            | 强（语义相近向量接近）     |
| **一词多义**     | 不支持                    | 动态嵌入可区分             |
| **计算效率**     | 高                        | 低（需矩阵乘法）           |
| **典型应用**     | 简单任务                  | 复杂任务                   |

---

### **五、代码示例：使用预训练 Word2Vec**

```python
from gensim.models import KeyedVectors
import numpy as np

# 加载预训练模型
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# 获取词向量
vector = model['apple']  # 300 维
print("'apple'的向量维度:", vector.shape)  # (300,)

# 计算相似度
print("'apple'与'banana'的相似度:", model.similarity('apple', 'banana'))
print("'apple'与'computer'的相似度:", model.similarity('apple', 'computer'))

# 语义类比
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("king - man + woman =", result)  # [('queen', 0.71)]

# 不匹配词
print("['dog', 'cat', 'apple', 'bird']中不匹配的词:",
      model.doesnt_match(['dog', 'cat', 'apple', 'bird']))  # apple
```

---

### **六、局限性与挑战**

1. **一词多义**：静态嵌入无法区分上下文含义，需动态模型。  
2. **领域适应性**：医学、法律等专业领域需微调或重训。  
3. **计算资源**：BERT 等大模型需大量 GPU 与数据。  
4. **可解释性**：向量维度语义难以直观解释。

---

### **总结**

Word Embedding 通过低维稠密向量有效捕捉词义，从静态 Word2Vec 到动态 BERT 持续演进。实际应用中根据任务选择合适方法，或组合多种方法以获得更佳效果。