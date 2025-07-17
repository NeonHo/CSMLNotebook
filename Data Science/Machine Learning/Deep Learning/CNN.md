multi-layer: make the different scale feature mix into the only one output feature map.
stride: control the size of the output feature map
padding: influence the output size, keep the feature map has the same size with the input shape.
pooling: squeeze the information of the input feature map.

# receptive field 感受野
- The closer neural unit can only see the feature in a limited corner.
- The far neural unit can see the global sight.

# Classical CNNs

Paddle Paddle = PyTorch + Keras

- AlexNet
	- 5 Conv Layers
	- 3 Fully-Connection Layers
	- Parameters 60 M
	- 2 GPUs training parallel.
- VGG
	- $3\times 3$ Conv Kernels
	- $2\times 2$ Pooling Kernels
	- activation functions: ReLU
	- several small kernels perform better than 1 big kernel.
		- 设输入通道数和输出通道数都为C， 3个步长为1的$3\times3$卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个$3\times3$连续卷积相当于一个$7\times7$卷积），其参数总量为$3\times (9\times C^2)$ ，如果直接使用$7\times7$卷积核，其参数总量为 $49\times C^2$ 。很明显，$27\times C^2$ 小于$49\times C^2$，即减少了参数；而且$3\times3$卷积核有利于更好地保持图像性质。
		- 使用了3个$3\times3$卷积核来代替$7\times7$卷积核，使用了2个$3\times3$卷积核来代替$5\times5$卷积核。这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度（因为多层非线性层可以增加网络深度来保证学习更复杂的模式），在一定程度上提升了神经网络的效果。
- Resnet
	- avoid the gradient vanishing caused by increasing the layers.
	- transmit the information from the deep layer to the shallow layer.
	- The residual connection can make the residual block into a identity mapping in properly occasions.
[[卷积]]