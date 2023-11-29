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
focus on re-parametrization and quantization as our main research fields.
- faster than resnet-101
- ==Channel Pruning== 
- part of YOLOv7
- The idea behind re-parametrization
	- represented in different math form
	- help gradient descent to reach a better local minimum.
	- 