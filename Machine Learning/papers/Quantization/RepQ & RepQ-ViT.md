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
- faster than resnet-101
- ==Channel Pruning== 
- part of YOLOv7
- The idea behind re-parametrization
	- represented in different math form
	- help gradient descent to reach a better local minimum.
		- improve performance
	- re-parameterization involves replacing linear layers (FC or Conv).
		- replace Conv with a block consisting of multiple Conv Lyaers with:
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
	- only 2 work:
		- QRepVGG
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
- Propose to compute the inference
	- 
