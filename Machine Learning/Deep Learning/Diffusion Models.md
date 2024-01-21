# Reference
DDPM(Denoise Diffusion Probabilistic Model) 理论梳理(含公式推导，小白) - Jiff的文章 - 知乎
https://zhuanlan.zhihu.com/p/560603623

# Background
- 图像生成领域常见模型
	- [[Section 1 GAN]]
	- [[Variational Autoencoder]]
- Denoising Diffusion Probabilistic Model (DDPM)
	- called Diffusion Models
	- for image synthesis
	- OpenAI, Stability AI, Google Brain
# Theory Introduction

![[Pasted image 20240120181026.png]]


- From noise to objective data sample
	- noise from simple distribution
- 2 procedures (Both are [[Section 4 Markov Chain]]):
	- forward (diffusion process)
	- reverse
		- to generate data samples (as the generator from [[Section 1 GAN]])
			- the generator of GAN will change the dimension.
			- DDPM will keep the dimension
	- ![[319a03cc4de41836a325f926f525d56.jpg]]
		- $x_0 \rightarrow x_T$ is forward
			- original image is added noise gradually to pure noise.
			- [[Section 4 Markov Chain]]  $q(x_{1:T}|x_o)=\prod_{t=1}^Tq(x_t|x_{t-1})$  
			- $q(x_t|x_{t-1})=N(x_t, \mu=\sqrt{1-\beta_t}x_{t-1}, \sigma^2=\beta_tI)$ 
			- $\beta_t$ is set previously.
				- increase as Linear or Cosine.
			- $x_t$ 是从非标准的正态分布中采样得到的。
			- 通过[[Reparameterization Trick]]
			- $x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon$
				- $\epsilon\sim N(0, I)$
				- 用 $\epsilon$ 代替采样时没有梯度的变量$x_t$ 从而有利于反向传播
		- $x_T\rightarrow x_0$ is reverse
			- random noise is recovered to input image.
			- learn a denoising process.
			- reverse of the $q(x_t|x_{t-1})$ is $q(x_{t-1}|x_t,x_0)$
				- 没有$x_0$, 马尔科夫链也没办法以一个高置信度重构出输入图像。
				- 将$q(x_{t-1}|x_t,x_0)$ 这个后验概率用高斯分布的概率密度函数展开
					- ![[Pasted image 20240120190405.png]]
				- DDPM use [[Section 1 Multi-layer Perceptron & Boolean Function#Neural Network]] to model the reverse process.
					- $p_\theta(x_{t-1}|x_t)\sim q(x_{t-1}|x_t)$
						- ![[Pasted image 20240120190738.png]]
						- ![[Pasted image 20240120190809.png]]
						- ![[e6541d259be4430a834bfe81174dd58.jpg]]
						- ![[6883dc7a8b5ad565d399715286e8c39.jpg]]
						- 方差不重要，关键在于均值，但是均值的拟合很重要。
							- ![[Pasted image 20240120190405.png]]
							- 方差只与 $\beta_t$ 有关， 无需预测。
							- $p_\theta(x_{t-1}|x_t)\sim N(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$
								- $\Sigma_\theta(x_t,t)=\frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}$
							- 高斯分布$p(\mu_1,\sigma_1)$ vs $q(\mu_2,\sigma_2)$ 的 KL散度
								- $D_{KL}(p,q)=\log \frac{\sigma_2}{\sigma_1}+\frac{\sigma^2_1+(\mu_1-\mu_2)^2}{2\sigma^2_2}-\frac{1}{2}$
								- 因为方差相同
								- 所以 $D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))=\frac{1}{2\Sigma^2_t}||\mu_t(x_t,x_0)-\mu_\theta(x_t,t)||^2 + C$
								- 所以我们最后的目标是最小化$||\mu_t(x_t,x_0)-\mu_\theta(x_t,t)||^2$
								- $\mu$ 分布不确定，值域较广，我们希望预测正态分布的采样。
									- $\mu_\theta(x_t,t)=\frac{1}{\sqrt{a_t}}(x_t+\frac{\beta}{})$