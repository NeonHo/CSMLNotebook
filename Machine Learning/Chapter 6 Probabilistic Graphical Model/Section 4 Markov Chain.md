The future is independent of the past given the present.
**过去所有的信息都已经被保存到了现在的状态，基于现在就可以预测未来。**
- 极端的思想却能够降低复杂度。
- 时间序列模型应用：
	- RNN [[Section 1 RNN v.s. CNN]]
	- HMM
- 随机过程
	- 统计模型
	- 对事物的过程
	- 进行预测和处理
	- ![[7794b60e7bc5ec8be7d5aac07df5a2d.jpg]]
- Markov Chain
	- 用数学方法解释自然变化的一般规律模型
	- 马尔可夫性质——无记忆性：
		- 下一状态的概率分布只由当前状态决定。
		- 此前的状态均与下一状态无关。
	- 数学
		- $P(X_{t+1}|...,X_{t-2},X_{t-1},X_{t})=P(X_{t+1}|X_t)$
		- 