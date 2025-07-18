
线性回归（Linear Regression）是机器学习中最基础、应用最广泛的**回归任务**算法，用于建模自变量（特征）与因变量（目标值）之间的**线性关系**。它通过拟合一条最优直线（或超平面），实现对连续型目标值的预测。

---

### 一、核心思想与数学定义

#### 1. 简单线性回归（单特征）
当只有一个自变量 $x$ 和一个因变量 $y$ 时，线性回归模型假设二者存在线性关系，可表示为  
$$
\hat{y} = w x + b
$$  
其中  
- $\hat{y}$ 是模型对 $y$ 的预测值；  
- $w$ 是**权重**（斜率），表示 $x$ 对 $y$ 的影响程度；  
- $b$ 是**偏置**（截距），表示当 $x=0$ 时 $y$ 的基准值。  

模型的目标是找到最优的 $w$ 和 $b$，使预测值 $\hat{y}$ 尽可能接近真实值 $y$。

#### 2. 多元线性回归（多特征）
当有多个自变量（特征）$x_1, x_2, \dots, x_n$ 时，模型扩展为  
$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$  
用向量形式可简化表示为  
$$
\hat{y} = \boldsymbol{w}^T \boldsymbol{x} + b
$$  
其中  
- $\boldsymbol{x} = [x_1, x_2, \dots, x_n]^T$ 是特征向量；  
- $\boldsymbol{w} = [w_1, w_2, \dots, w_n]^T$ 是权重向量；  
- $\boldsymbol{w}^T \boldsymbol{x}$ 表示向量内积（即加权求和）。

---

### 二、损失函数：衡量预测误差
为找到最优的 $\boldsymbol{w}$ 和 $b$，需定义**损失函数**（Loss Function）来衡量预测值与真实值的差异，线性回归中最常用的是**均方误差（Mean Squared Error, MSE）**：

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2
$$  
其中 $m$ 是样本数量，$\hat{y}_i$ 是第 $i$ 个样本的预测值，$y_i$ 是真实值。

模型训练的目标是**最小化均方误差**，即  
$$
\min_{\boldsymbol{w}, b} \text{MSE} = \min_{\boldsymbol{w}, b} \frac{1}{m} \sum_{i=1}^m (\boldsymbol{w}^T \boldsymbol{x}_i + b - y_i)^2
$$

---

### 三、参数求解：如何找到最优的 $\boldsymbol{w}$ 和 $b$

#### 1. 正规方程（解析解）
对于线性回归，均方误差的最小化问题存在**解析解**（通过数学推导直接求出最优参数），称为**正规方程**：

$$
\hat{\boldsymbol{\theta}} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}
$$  
其中  
- $\hat{\boldsymbol{\theta}} = [w_1, w_2, \dots, w_n, b]^T$ 是包含权重和偏置的参数向量；  
- $\boldsymbol{X}$ 是**增广特征矩阵**（在原始特征矩阵后添加一列全为 1 的向量，用于表示偏置 $b$）；  
- $(\boldsymbol{X}^T \boldsymbol{X})^{-1}$ 是矩阵 $\boldsymbol{X}^T \boldsymbol{X}$ 的逆矩阵。  

**适用场景**：样本数量 $m$ 较小、特征数量 $n$ 较少（逆矩阵计算复杂度为 $O(n^3)$）。

#### 2. 梯度下降（数值解）
[[Section 5 Stochastic Gradient Descent#3.2. 小批量梯度下降]]
当样本量或特征数较大时，正规方程计算成本过高，此时常用**梯度下降**（Gradient Descent）求解参数：

1. **初始化参数**：随机设置 $\boldsymbol{w}$ 和 $b$ 的初始值。  
2. **计算梯度**：求均方误差对 $\boldsymbol{w}$ 和 $b$ 的偏导数（梯度），确定参数更新方向：  
   $$
   \frac{\partial \text{MSE}}{\partial w_j} = \frac{2}{m} \sum_{i=1}^m (\hat{y}_i - y_i) x_{ij}
   $$  
   $$
   \frac{\partial \text{MSE}}{\partial b} = \frac{2}{m} \sum_{i=1}^m (\hat{y}_i - y_i)
   $$  
3. **更新参数**：沿梯度负方向（减小误差的方向）迭代更新参数：  
   $$
   w_j \leftarrow w_j - \alpha \cdot \frac{\partial \text{MSE}}{\partial w_j}
   $$  
   $$
   b \leftarrow b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
   $$  
   其中 $\alpha$ 是**学习率**（控制更新步长，需手动设置）。  
4. **收敛条件**：当梯度接近 0（参数变化小于阈值）或达到最大迭代次数时，停止更新。

**适用场景**：大规模数据（如 $m > 10^4$），通过批量梯度下降（BGD）、随机梯度下降（SGD）或小批量梯度下降（Mini-Batch GD）优化效率。

---

### 四、线性回归的扩展与优化

#### 1. 多项式回归（处理非线性关系）
当自变量与因变量存在非线性关系（如二次关系）时，可通过**特征变换**将非线性关系转化为线性关系。例如，对特征 $x$ 进行平方变换：  
$$
\hat{y} = w_1 x + w_2 x^2 + b
$$  
令 $x_1 = x$，$x_2 = x^2$，则模型回归为线性形式  
$$
\hat{y} = w_1 x_1 + w_2 x_2 + b
$$  
可沿用线性回归的求解方法。

#### 2. 正则化（防止过拟合）
当模型复杂度较高（如特征过多）时，线性回归可能出现**过拟合**（训练误差小，测试误差大）。通过添加正则化项约束参数大小，可缓解过拟合：

- **岭回归（Ridge Regression）**：添加 L2 正则化，损失函数为  
  $$
  \text{Loss} = \text{MSE} + \lambda \sum_{j=1}^n w_j^2
  $$  
  其中 $\lambda \geq 0$ 是正则化强度，控制对参数的惩罚力度。

- **Lasso 回归**：添加 L1 正则化，损失函数为  
  $$
  \text{Loss} = \text{MSE} + \lambda \sum_{j=1}^n |w_j|
  $$  
  特点是会使部分参数变为 0，实现“特征选择”（适合高维稀疏数据）。

- **弹性网络（Elastic Net）**：结合 L1 和 L2 正则化，平衡特征选择和参数平滑性。
[[特征选择#3. 嵌入法（Embedded）]]

---

### 五、优缺点与适用场景

#### 优点
- **简单易解释**：参数 $w_j$ 直接表示特征 $x_j$ 对目标值的影响程度（正负表示方向，绝对值表示强度）。  
- **计算高效**：无论是正规方程还是梯度下降，求解速度都快于复杂模型。  
- **可扩展性强**：通过特征变换（如多项式）或正则化，可处理非线性和高维数据。

#### 缺点
- **只能建模线性关系**：对非线性关系（如指数、对数关系）需手动进行特征变换，否则拟合效果差。  
- **对异常值敏感**：均方误差会放大异常值的影响，导致模型偏移（可通过稳健回归如 Huber 损失缓解）。

#### 适用场景
- **预测连续值**：如房价预测、销售额预测、温度预测等。  
- **特征重要性分析**：通过权重 $w_j$ 判断哪些特征对目标值影响最大（如影响用户消费的关键因素）。  
- **作为基准模型**：在复杂任务中，先用线性回归建立基准，再与复杂模型（如神经网络）对比。

---

### 六、代码示例（Python 实现）

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 生成示例数据（y = 2x + 3 + 噪声）
np.random.seed(42)
x = np.random.rand(100, 1) * 10  # 特征（100 个样本，1 个特征）
y = 2 * x + 3 + np.random.randn(100, 1) * 2  # 目标值（带噪声）

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 训练线性回归模型
model = LinearRegression()
model.fit(x_train, y_train)

# 预测与评估
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

print(f"权重 w: {model.coef_[0][0]:.2f}")  # 接近真实值 2
print(f"偏置 b: {model.intercept_[0]:.2f}")  # 接近真实值 3
print(f"测试集 MSE: {mse:.2f}")
```

---

### 总结
线性回归是通过拟合线性关系预测连续目标值的经典算法，核心是最小化均方误差，可通过正规方程或梯度下降求解参数。其优点是简单易解释、计算高效，缺点是对非线性关系拟合能力有限。通过特征变换（如多项式回归）或正则化（如岭回归），可扩展其适用范围，是解决回归问题的基础工具和重要基准。
