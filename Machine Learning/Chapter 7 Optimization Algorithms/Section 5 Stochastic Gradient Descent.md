# 1. 为什么要使用随机梯度下降？
传统的梯度下降算法要将整个训练集都过一遍才能迭代一次，对于庞大的数据集来说计算量太大。
所以我们需要应对这个每一个迭代就要用到一个训练集的情况，所以使用随机梯度下降。
# 2. 随机梯度下降是做什么的？
## 2.1. 经典梯度下降算法
[[Section 3 Classical Optimization Algorithms#3.2.1. 一阶法]] 中所讲的
$$
\theta_{t+1}=\theta_{t} -\alpha\nabla L(\theta_t)
$$
中的$L(\theta)$可以表示为：
$$
L(\theta)=\mathbb{E}_{(x,y)\sim P_{data}}L(f(x,\theta),y)=\frac{1}{M}\sum_{i=0}^{n}L(f(x_i,\theta),y_i)
$$
这就是最经典最原始的梯度下降算法。
但是$M$通常很大，因为它代表着整个训练集的样本量。
所以，人们走向的另一个极端：让每一个迭代仅仅使用一个样本进行参数优化，即直接用单个样本计算得到所损失看作当前平均损失。即SGD 随机梯度下降。
# 3. 随机梯度下降怎么用？
## 3.1. 随机梯度下降
$$
\begin{array}{lr}
L(\theta;x_i,y_i)=L(f(x_i,\theta),y_i)\\
\nabla{L(\theta;x_i,y_i)}=\nabla L(f(x_i,\theta),y_i)
\end{array}
$$
加快了收敛速度，适应于模型在线更新的场景。
## 3.2. 小批量梯度下降
但是这种迭代算法的方差很大，为了减小方差，让迭代算法更加稳定，在计算量能够承受的前提下，尽可能利用矩阵并行优化的优势，使用训练集中的$m$个训练数据进行参数更新。即小批量梯度下降（Mini-Batch Gradient Descent）。
$$
\begin{array}{lr}
L(\theta)=\frac{1}{m}\sum_{j=0}^m L(f(x_{b_j},\theta),y_{b_j})\\
\nabla L(\theta)=\frac{1}{m}\sum_{j=0}^m \nabla L(f(x_{b_j},\theta),y_{b_j})
\end{array}
$$
注意除了$m$需要基于经验调参以外，如何选取这$m$个样本也是有讲究的：
每进行一个epoch：
1. 对整个训练集进行随机排序一次。
2. 要进行一个batch迭代时：
	1. 按照当前训练集的顺序挑选$m$个样本。
	2. 将这个$m$个样本组成当前batch。
3. 执行2.直到所有的训练样本都被用过一次为止。


