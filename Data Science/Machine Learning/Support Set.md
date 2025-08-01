在小样本学习（[[few-shot learning]]）中，**支持集（Support Set）** 是一个核心概念，它是模型在训练或推理阶段用于学习新类别知识的关键数据集合。以下从定义、作用、与其他集合的关系、典型应用场景等方面详细介绍：

---

### **1. 支持集的定义**

支持集是由少量（通常为 1~5 个）带有标签的样本组成的集合，这些样本属于模型需要快速学习的**新类别**。

- 例如，在“5-shot 3-way”任务中，支持集包含 3 个新类别，每个类别有 5 个带标签的样本（共 15 个样本）。
- 这些样本是模型在处理当前任务时的“参考依据”，帮助模型理解新类别的特征分布。

---

### **2. 支持集的核心作用**

在小样本学习中，模型通常缺乏大量标注数据，支持集的作用是为模型提供**少量先验知识**，使其能够：

- **快速归纳新类别特征**：通过少量样本捕捉类别内的共性（如“猫”的耳朵形状、毛发特征）和类别间的差异（如“猫”与“狗”的体型区别）。
- **建立类别判断标准**：支持集的样本标签为模型提供了“锚点”，模型通过对比查询样本（Query Set）与支持集样本的相似度，判断查询样本的类别。

---

### **3. 支持集与其他集合的关系**

在小样本学习任务中，数据通常被划分为三个部分，支持集与其他集合的关系如下：

| 集合 | 定义 | 与支持集的关系 |
|---|---|---|
| **支持集（Support Set）** | 少量带标签的新类别样本，用于模型学习新类别特征。 | 核心参考集，模型的“学习素材”。 |
| **查询集（Query Set）** | 与支持集同属新类别、但标签未知的样本，需要模型预测其类别。 | 模型的“测试对象”，通过与支持集样本对比完成分类。 |
| **训练集（Base Set）** | 大量带标签的基础类别样本，用于预训练模型的特征提取能力（非新类别）。 | 为模型提供通用特征提取能力，支持集则基于此能力快速适配新类别（类似“举一反三”中的“一”）。 |

**举例**：

- 基础集：包含 1000 张“汽车”“自行车”的图片（用于预训练模型识别交通工具的通用特征）。
- 支持集：3 个新类别（“摩托车”“卡车”“电动车”），每个类别 5 张图片（共 15 张，带标签）。
- 查询集：10 张“摩托车”“卡车”“电动车”的图片（无标签，需模型预测类别）。

---

### **4. 支持集在典型小样本学习方法中的应用**

不同小样本学习方法对支持集的利用方式不同，核心是通过支持集构建“类别原型”或“相似度度量”：

1. **原型网络（Prototype Networks）**
   - 模型将支持集中每个类别的样本特征取平均，得到该类别的“原型（Prototype）”（如“猫”的平均特征向量）。
   - 查询样本的类别由其特征与各原型的距离（如欧氏距离）决定（距离最近的原型对应的类别即为预测结果）。
   - **支持集的作用**：直接决定“原型”的位置，是类别判断的核心依据。

2. **匹配网络（Matching Networks）**
   - 模型通过支持集中的样本及其标签，学习一个“注意力机制”或“相似度函数”，衡量查询样本与每个支持集样本的相似度。
   - 查询样本的类别由支持集样本的标签通过相似度加权投票决定（相似度高的支持集样本权重更大）。
   - **支持集的作用**：为相似度计算提供“参照样本”，标签作为投票依据。

1. **元学习（[[Meta Learning]]）方法**
   - 模型在元训练阶段（使用基础集）学习“如何利用支持集快速学习新任务”的能力（即“学习如何学习”）。
   - 元测试阶段，支持集作为新任务的“训练数据”，模型通过几轮梯度更新（如 MAML 方法）或特征调整，快速适配新类别。
   - **支持集的作用**：模拟新任务的“训练样本”，帮助模型完成“快速适配”。


---

### **5. 支持集的关键特性**

- **样本量少**：通常每个类别 1~5 个样本（对应 1-shot、5-shot 任务），这也是“小样本”的核心体现。
- **类别新颖性**：支持集的类别是模型在训练阶段（基础集）未见过的新类别，考验模型的泛化能力。
- **标签必要性**：支持集样本必须带标签，否则模型无法学习新类别的特征或建立判断标准（区别于无监督学习）。

---

### **总结**

支持集是小样本学习中模型“快速学习新类别”的核心依据，通过少量带标签的新类别样本，为模型提供特征参考和类别锚点，使模型能够在缺乏大量数据的情况下完成对查询样本的分类。其设计直接影响小样本学习的性能，合理利用支持集的特征分布和标签信息，是提升模型泛化能力的关键。