
在时间序列分析中，**偏自相关函数（Partial Autocorrelation Function, PACF）** 是衡量时间序列中两个观测值之间“直接相关性”的工具，它剔除了中间变量的间接影响，专注于变量自身的滞后项与当前项的净关联。PACF与自相关函数（[[ACF自相关性]]）共同构成了识别[[AR自回归]]模型阶数的核心依据。


### 一、PACF的定义与核心思想
偏自相关函数用于衡量**在控制所有中间滞后项（即介于两个时间点之间的观测值）影响后，时间序列中第t期观测值$x_t$与第$t-k$期观测值$x_{t-k}$之间的相关系数**。

- 对于滞后阶数$k$，PACF的记为$\phi_{kk}$（也可简写为$PACF(k)$）。
- 直观理解：例如$PACF(3)$衡量的是$x_t$与$x_{t-3}$的相关性，且已排除$x_{t-1}$和$x_{t-2}$对两者的间接影响。


### 二、PACF与ACF的区别
- **自相关函数（ACF）**：衡量$x_t$与$x_{t-k}$的**总相关性**，包括直接相关和通过中间滞后项（如$x_{t-1}, x_{t-2}, ..., x_{t-k+1}$）的间接相关。
- **偏自相关函数（PACF）**：仅衡量$x_t$与$x_{t-k}$的**直接相关性**，剔除了中间滞后项的干扰。

举例：  
在时间序列$x_t \rightarrow x_{t-1} \rightarrow x_{t-2}$中，$x_t$与$x_{t-2}$的ACF可能因$x_{t-1}$的中介作用而显示显著相关，但PACF会剔除$x_{t-1}$的影响，若两者无直接关联，则$PACF(2)$会接近0。


### 三、PACF的计算原理
PACF的计算基于**自回归（AR）模型的系数**。对于滞后阶数$k$，$\phi_{kk}$是$k$阶AR模型中$x_{t-k}$的系数，即：

$$
x_t = \phi_{k1}x_{t-1} + \phi_{k2}x_{t-2} + ... + \phi_{kk}x_{t-k} + \varepsilon_t
$$

其中，$\phi_{kk}$就是$k$阶滞后的偏自相关系数。

#### 计算步骤（以Yule-Walker方程为例）：
1. 计算序列的自相关系数$r_1, r_2, ..., r_k$（$r_i$为滞后$i$的ACF值）。  
2. 构建Yule-Walker方程组：
   $$
   \begin{cases} 
   r_1 = \phi_{k1}r_0 + \phi_{k2}r_1 + ... + \phi_{kk}r_{k-1} \\
   r_2 = \phi_{k1}r_1 + \phi_{k2}r_0 + ... + \phi_{kk}r_{k-2} \\
   ... \\
   r_k = \phi_{k1}r_{k-1} + \phi_{k2}r_{k-2} + ... + \phi_{kk}r_0 
   \end{cases}
   $$
   （注：$r_0 = 1$，因变量与自身的自相关系数为1）  
3. 求解方程组，得到$\phi_{kk}$，即$PACF(k)$的值。


### 四、PACF的图形表示（偏自相关图）
偏自相关图（PACF图）是横轴为滞后阶数$k$、纵轴为$PACF(k)$值的条形图，通常会添加置信区间（如95%置信区间，以虚线表示）。

- 若$PACF(k)$落在置信区间内，说明该滞后阶数的偏自相关不显著（可视为0）；
- 若超出置信区间，则说明$x_t$与$x_{t-k}$存在显著的直接相关性。


### 五、PACF的主要应用：识别AR模型的阶数
在时间序列建模中，PACF是识别**自回归模型（AR(p)）阶数$p$**的关键工具，核心规则是：  
- **AR(p)模型的PACF在滞后$p$处会“截尾”**（即当$k > p$时，$PACF(k)$迅速趋近于0，并落在置信区间内）。

举例：  
- 若序列适合AR(1)模型，则PACF在滞后1处显著不为0，滞后2及以上的PACF均接近0；  
- 若适合AR(2)模型，则PACF在滞后1和2处显著，滞后3及以上接近0。


### 六、总结
1. PACF衡量时间序列中变量与滞后$k$项的**直接相关性**，剔除中间项的间接影响。  
2. 与ACF的核心区别在于是否包含间接相关。  
3. 主要用途是确定AR模型的阶数（通过PACF的截尾特性）。  
4. 计算依赖AR模型系数，可通过Yule-Walker方程求解。

在实际分析中，PACF常与ACF配合使用（如ACF识别MA模型阶数，PACF识别AR模型阶数），是时间序列建模的基础工具之一。