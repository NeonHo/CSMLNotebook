### F-Measure（F 分数）

F-Measure 是**精确率（[[Precision]]）** 和**召回率（Recall）** 的加权调和平均，用于综合评估模型性能（尤其适用于不平衡数据集）。

#### 1. 精确率（Precision）

先明确精确率的定义（F-Measure 的基础）：

$$  
\text{精确率} = \frac{TP}{TP + FP}
$$
表示模型预测为正例的样本中，实际为正例的比例（“预测正确的正例占预测正例的比例”）。

#### 2. F-Measure 的计算

F-Measure 的一般形式为：

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$
其中 $\beta$ 是权重参数：

  

- $\beta > 1$：更侧重召回率（如疾病诊断，避免漏诊）。
- $\beta < 1$：更侧重精确率（如垃圾邮件识别，避免误判正常邮件）。

#### 3. 最常用：F1 分数
[[F-1 Score]]
当 $\beta = 1$ 时，F-Measure 称为**F1 分数**，此时精确率和召回率权重相等：

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- 意义：F1 分数平衡了精确率和召回率，适用于两者都需要关注的场景（如推荐系统，既需准确推荐用户感兴趣的内容，又需覆盖足够多的兴趣点）。
- 局限：当数据集极度不平衡时，F1 可能无法反映模型对少数类的识别能力，需结合灵敏度、特异度等指标。