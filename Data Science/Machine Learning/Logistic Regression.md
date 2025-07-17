因变量: 二元分布[[二项分布]]

模型: $E[y|x;\theta]$ 学习到超参数后, 给定自变量, 得到因变量的数学期望.



与 [[Linear Regression]] 之间的区别


逻辑回归（Logistic Regression）是一种广泛应用于**二分类问题**的统计学习方法，尽管名称中包含“回归”，但其本质是分类算法。它通过逻辑函数（Sigmoid 函数）将线性回归的输出映射到概率空间，从而进行分类决策。以下是其核心原理、数学基础及应用特点的详细介绍：

### 1. 基本原理与模型形式

#### (1) 线性回归的局限性  
对于二分类问题（标签 $y \in \{0, 1\}$），直接使用线性回归 $z = w^T x + b$ 的输出可能超出 $[0, 1]$ 范围，无法表示概率。

#### (2) 逻辑函数（Sigmoid 函数）  
逻辑回归引入 Sigmoid 函数将线性输出映射到 $[0, 1]$ 区间：  
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$  
其中 $z = w^T x + b$，$\sigma(z)$ 表示样本 $x$ 属于正类（$y=1$）的概率 $P(y=1|x)$。

#### (3) 模型输出与决策规则  
逻辑回归模型可表示为：  
$$ P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}} $$  
决策规则为：若 $P(y=1|x) \geq 0.5$，则预测为正类；否则为负类。

---

### 2. 损失函数：对数损失（Log Loss）
[[LogLoss]]
逻辑回归通过最大化对数似然函数（等价于最小化对数损失）来估计参数 $w$ 和 $b$。对于单个样本 $(x, y)$，对数损失为：  
$$ L(y, \hat{y}) = -\bigl[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \bigr] $$  
其中 $\hat{y} = \sigma(w^T x + b)$。

**直观解释**  
- 当 $y=1$ 时，损失为 $-\log(\hat{y})$；若 $\hat{y} \to 1$，损失趋于 $0$；若 $\hat{y} \to 0$，损失趋于 $+\infty$。  
- 当 $y=0$ 时，损失为 $-\log(1-\hat{y})$；若 $\hat{y} \to 0$，损失趋于 $0$；若 $\hat{y} \to 1$，损失趋于 $+\infty$。

---

### 3. 参数优化：梯度下降

通过最小化对数损失的平均值（经验风险）来求解参数：  
$$ J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \Bigl[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \Bigr] $$  

使用梯度下降迭代更新参数：  
$$ w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \quad b \leftarrow b - \alpha \frac{\partial J}{\partial b} $$  

其中梯度为：  
$$ \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) $$

---

### 4. 多分类扩展：Softmax 回归
[[SoftMax]]
逻辑回归可扩展到多分类问题，称为 **Softmax 回归**。对于 $K$ 个类别，模型输出每个类别的概率：  
$$ P(y=k|x) = \frac{e^{w_k^T x}}{\sum_{j=1}^{K} e^{w_j^T x}}, \quad k=1,2,\dots,K $$  

损失函数为[[交叉熵损失]]：  
$$ L(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log(\hat{y}_k) $$  
其中 $y_k$ 为 [[One-Hot编码]]标签（第 $k$ 类为 1，其余为 0）。

---

### 5. 正则化
[[Ridge Regression & Lasso Regression]]

为防止过拟合，常加入 L1 或 L2 正则化：  
- **L2 正则化（Ridge）**：$J(w, b) + \frac{\lambda}{2} \|w\|^2$  
- **L1 正则化（Lasso）**：$J(w, b) + \lambda \|w\|_1$  
- **弹性网络（Elastic Net）**：结合 L1 和 L2 正则化  

---

### 6. 优缺点

- **优点**  
  - 计算高效，适合大规模数据。  
  - 输出概率可解释性强（如疾病风险、违约概率）。  
  - 可通过正则化控制过拟合。  
  - 对线性可分数据效果好。  

- **缺点**  
  - 只能建模线性决策边界（需通过特征变换引入非线性）。  
  - 对异常值敏感。  
  - 对多分类问题需扩展为 Softmax 回归。  

---

### 7. 应用场景

- **医学诊断**：根据症状预测疾病概率。  
- **金融风控**：客户违约概率评估。  
- **市场营销**：用户购买意愿预测。  
- **自然语言处理**：文本分类（如垃圾邮件识别）。  

---

### 8. 代码示例（Python 实现）

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 加载乳腺癌数据集（二分类）
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建逻辑回归模型（默认 L2 正则化）
clf = LogisticRegression(max_iter=1000)

# 训练模型
clf.fit(X_train, y_train)

# 预测并评估
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# 查看系数（特征重要性）
print("Coefficients:", clf.coef_)
```

---

### 9. 与其他模型的对比

| 模型      | 适用场景        | 决策边界  | 输出形式  |
| ------- | ----------- | ----- | ----- |
| 逻辑回归    | 二分类、概率预测    | 线性    | 概率值   |
| [[SVM]] | 二分类、小样本高维数据 | 最大间隔  | 类别标签  |
| [[决策树]] | 多分类、非线性关系   | 分段线性  | 类别/概率 |
| 神经网络    | 复杂非线性问题     | 任意非线性 | 概率/标签 |

---

### 总结
逻辑回归是处理二分类问题的经典方法，通过 Sigmoid 函数将线性回归输出映射为概率，具有计算简单、可解释性强的优点。其核心是对数损失函数和梯度下降优化，可通过正则化增强泛化能力。对于多分类问题，可扩展为 Softmax 回归。尽管它只能表示线性决策边界，但在许多实际场景中仍是首选基线模型。
