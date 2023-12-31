- Quantization: network compression
- Re-parameterize: emerging technique to improve model performance
- both is individually.
- Combine is lack
- RepQ
	- simultaneous application
	- apply quantization to re-parameterized network.
	- the test stage weights
		- ==present a differentiable function of trainable parameters==
	- apply QAT on top of this function
	- better than baseline method LSQ
# Introduction
Paragraph 1st
- focus on re-parametrization and quantization as our main research fields.

Paragraph 2nd
- Faster than resnet-101
- ==Channel Pruning== 
- part of YOLOv7
- The idea behind re-parametrization
	- represented in different math form
	- help gradient descent to reach a better local minimum.
		- Improve performance
	- re-parameterization involves replacing linear layers (FC or Conv).
		- Replace Conv with a block consisting of multiple Conv Layers with:
			- various kernel sizes
			- channel numbers
			- residual connections
			- BN
	- training with blocks
	- inference with a single Conv
	- boosting model quality but without burden at inference.
- Efficiency
	- re-parameterization
	- quantization $\ge75\%$ cut

Paragraph 3rd
- apply re-parameterization to practical applications
- to quantize it.
	- Only 2 work:
		- QARepVGG
		- RepOpt
	- eliminates the improvements introduced by the re-parameterization.
- to solve problem:
	- quality
	- generalization
- introduce QAT for Re-Parameterized NN.
- challenge:
	- difference between testing weights and training weights
	- prohibit the standard QAT application
	- if QAT on training, can not merge into one branch without increasing the quantizer bit width.
- Propose
	- to compute the inference parameters of a Conv 
	- as a differentiable function of the trainable parameters of the re-parameterized block
	- to apply a pseudo-quantization function on top of it.
	- enables end2end QA Re-parameterized training.

Paragraph 4th
- first method allows QAT with arbitrary re-parametrization schemes.
	- extensive experiments showing that our RepQ consistent quality enhancement on SOTA architectures.
	- lossless 8-bit quantization of Rep
- BN contained in the rep blocks 
	- challenge for differentiable merging and hence for QAT.
	- how to compute differentiable weight functions via BN folding
	- enhance BN-folding by introducing BN statistics estimation
	- reduce the computational complexity in the theory and practice.
# Related works
- Re-parameterization
	- train deep networks without explicit residual connections
	- single-branch architectures to reach a decent quality.
	- a larger number of conv, diverse branches in their blocks.
	- object detection : yolov6 v7
	- faster convergence in a simple convex problem.
	- rep equal to a certain gradient scaling strategy.
- Quantization
	- 32bit float point numbers is redundant to remain quality.
	- search the optimal low-bit integer parameters.
	- quantization
		- rounding
			- zero gradient everywhere
			- QAT address this issue by simulating quantization in a differentiable manner.
				- allowing the subsequent quantization.
					- injecting pseudo-quantization noise into full-precision network training.
					- straight-through estimator
					- a smooth approximation of stair-step function
				- improvement
					- knowledge distillation
					- progressive quantization
					- stochastic precision
					- BN re-estimation
					- additional regularization
					- various non-uniform quantization extensions.
# Background

Quantization
- reduce inference time
- power consumption
- decreasing the precision of mat mul in Conv and FC
- QAT recover the quality
	- the original conv: $X * W$ 
	- transformed into $Q(X)*Q(W)$
		- $Q$ is pseudo-quantization func
- LSQ is used

Options
- Apply Rep and PTQ successively
	- unfriendly distributions
	- big drop after PTQ
- Apply Rep and QAT successively
	- Rep train and convert into single layer
	- std QAT
- Apply Rep and QAT simultaneously
	- quantize each layer inside a rep block independently
	- merge those layers into a single quantized layer only after QAT
	- ==what is rep in this==
		- Rep is not only diverse branch, it can also have multiple-conv layers in one branch
		- $f(x)=xw=xw_1w_2$
		- lower bit quantization is much more challenging.
		- Merge into one is a big drop.
- QAT and Rep simultaneously
	- performing pseudo-quantization on top of merged rep block.

# Proposed Methods

1. RepQ training framework Rep Block without BN layers
2. For BN layers
	1. RepQ-BN
	2. RepQ-BNEst
## RepQ: QAT with Rep
- in Online Convolutional Re-parameterization
	- reduce training time
		- by merging Rep blocks ==without== BN into a single Conv
	- still optimizing the extended set of weights introduced by Rep
		- the summary of weighted weights.
- Illustrate the Rep
	- In a simple example: 
		- Training $R(X, W)=X*W_1*W_2+X*W_3$
		- Inference $R(X,W)=X*(W_1*W_2+W_3)=X*M(W_1,W_2,W_3)$
	- In a broader sense
		- $M$ is a differentiable function.
			- From the block's trainable parameters,
			- to the final converted convolution.
		- Generalize to all novel Rep strategies without BN:
			- $R(X,W)=X*M(W_1,...,W_n)$

- $M(W_1, W_2, W_3)$ and $X*W_1*W_2+X*W_3$ is equally in gradient flow in backward and forward.

- Pseudo-quantization
	- $X*M(W_1,...,W_n)\rightarrow Q(X)*Q(M(W_1,...,W_n))$
	- apply on top of $M$
	- $\because$ $Q$ and $M$ are differentiable function.
	- $\therefore$ the gradient propagates smoothly to the weights $W_1,...,W_n$.
	- End-to-end QAT $RepQ$ = Rep with $M$ $+$ pseudo-quantization function $Q$.
## RepQ-BN
- use BN
	- some papers: BN is essential
	- removal leads drop
	- handle BN in QAT
- First option is fusing BN with the preceding Conv during training.
	- Folding BN reduces the task to the no-BN case.
	- $BN(X*W)=\frac{X*W-\mathbb{E}[X*W]}{\sqrt{\mathbb{V}[X*W]+\varepsilon}}\gamma+\beta$
	- $BN(X*W)=X*\frac{W}{\sqrt{\mathbb{V}[X*W]+\varepsilon}}\gamma-\frac{\mathbb{E}[X*W]}{\sqrt{\mathbb{V}[X*W]+\varepsilon}}\gamma+\beta$
	- $BN(X*W)=X*M(X,W)+b(X,W)$
- A Conv followed by BN is equivalent to a single Conv operator.
	- Dependent on the input X through the batch statistics: mean and variance.
	- Rep with BN: $R(X, W)=X*M(X, W_1,...,W_n)$
	- Algorithm 1st To compute $M$ and apply quantization in practice for a simple case of $R(X, W)=BN(X*W)$
		- $Y=X*W$
## RepQ-BNEst
- Conv is computed twice in Algorithm 1st.
	- The 1st Conv is used to calculate the BN statistics $\mu$ and $V$.
	- Not necessary.
		- A novel method of estimating BN running statistics based on inputs and weights without extra Conv.
		- A simple example:
			- $X (Batch\ Size, Height, Width, Channel_{in})$
			- $W (Channel_{in}, Channel_{out})$
			- $\mathbb{E}[X^{(b,h,w,i)}W^{(i,o)}]=\mathbb{E}[X^{(b,h,w,i)}]W^{(i,o)}$
				- calculate $i$ means on $b\times h\times w$ input values. $O(b\times h\times w)\times C_{in}$
				- Multiply $i\times o$ weight parameters. $O(1\times C_{in}\times C_{out})$
				- Avoiding feature map $XW$ storing. and $O(b\times h\times w\times 1\times 1\times C_{in}\times C_{out})$
				- The complexity is no different (I think).
- A similar reduction is not possible due to the need to calculate the input covariance matrix.
	- Approximate the covariance matrix with a diagonal form.
		- The variance is substituted with
			- quadratic statistic of the weight $W*W$ (element-wise square of $W$)
			- output that estimates variance
		- computationally more efficient for QAT.
		- $\mathbb{E}[X*W]\approx \mathbb{E}[X]\cdot \sum_{h,w}W_{h,w}$
		- $\mathbb{V}[X*W]\approx \mathbb{V} [X]\cdot \sum_{h,w}W_{h,w}^2$

# Exp
## setup
### For Resnet-18
#### ACNet
[1908.03930v3.pdf (arxiv.org)](https://arxiv.org/pdf/1908.03930v3.pdf)
![[Pasted image 20231203200006.png]]
- ![[Pasted image 20231203194833.png]]
- ![[Pasted image 20231203195223.png]]
#### OREPA
[2204.00826v1.pdf (arxiv.org)](https://arxiv.org/pdf/2204.00826v1.pdf)
![[Pasted image 20231203195952.png]]
aiming to reduce the huge training overhead by squeezing the complex training-time block into a single convolution.
![[Pasted image 20231203195916.png]]
![[Pasted image 20231203200942.png]]
### For VGG
#### RepVGG A0 & B0
![[Pasted image 20231203201458.png]]
![[Pasted image 20231203201445.png]]
- compare with QARepVGG
- Baseline:
	- two stage:
		- FP pretrained for initialization.
		- QAT
	- w/o Rep during the QAT
	- Plain: no Rep
	- Merged: trained in the FP stage, Rep merged back into Conv before QAT.
	- RepQ: don't merge after training, build new gradient flow for these weights and apply QAT.
	- A simple example for $R(X)=BN(X*W)+X$
		- if BNEst
			- FP training with BN estimation
			- Q: Why need FP training with BN estimation?
			- A: W is related the first training stage, so the dataflow is needed be consistent with quantization stage. (RepQ-BN don't need because the extra Conv can get away from the training flow passes.)
		- else:
			- FP training
		- if BNEst
			- estimate the BN statistics.
			- fuse BN
		- else:
			- an extra Conv to calculate BN statistics.
			- fuse BN
		- quantization training
## results
- 8-bit RepQ exceeds the FP result
- RepQ-BN is better than RepQ-BNEst in 8 bits
- RepQ-BNEst is better in 4 bits
- RepQ-BNEst allows for a 25% training time reduction compared to RepQ-BN.
# Discussion
4-bit RepVGG-B0 has 2 times less bit-operations than 8-bit RepVGG-A0, while its acc is higher.
## Limitations
RepQ is the increase in training time (as same as Rep)
RepQ-BNEst mitigates this issue.