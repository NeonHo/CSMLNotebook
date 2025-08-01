在卷积神经网络（Convolutional Neural Network，[[CNN]]）中，一次卷积操作的乘法次数取决于以下几个因素：

1. **输入特征图大小**：输入特征图的大小会影响一次卷积操作的乘法次数。如果输入特征图的大小是 $(H \times W)$，其中 \($H$\) 是高度（行数），\($W$\) 是宽度（列数），那么乘法操作的次数可以表示为 \($H \times W$\)。

2. **卷积核大小**：卷积核的大小也会影响乘法操作的次数。如果卷积核的大小是 \($K_h \times K_w$\)，其中 \($K_h$\) 是卷积核的高度，\($K_w$\) 是卷积核的宽度，那么一次卷积操作的乘法次数可以表示为 \($K_h \times K_w$\)。

3. **通道数**：如果输入特征图和卷积核都有多个通道，那么每个通道之间的乘法操作也需要计算。通常，对于输入特征图的通道数为 \($C_{\text{in}}$\)，卷积核的通道数为 \($C_{\text{out}}$\)，一次卷积操作的乘法操作次数可以表示为 \($C_{\text{in}} \times C_{\text{out}}$\)。

4. **步幅（stride）和填充（padding）**：步幅和填充的设置也会影响卷积操作的乘法次数。较大的步幅和较多的填充会减少输出特征图的大小，从而降低乘法操作的次数。

# 2D Convolution

Each output unit need $C_{in}\times K^2$ multiplication and $C_{in}\times K^2-1$ addition.
If we have bias, we need an extra addition for each output unit.
And the total number of output units is $C_{out}\times W\times H$.
So the total number of computation for 1 layer is 
$$
2\times C_{in}\times K^2 \times C_{out}\times W\times H
$$

# 3D Convolution

3D convolution's kernel is 3D, so it has an extra dimension compared with 2D convolution, and the output feature matrix has an extra dimension $T$ to present "time".

So the total number of computation for 1 layer is:
$$
2\times C_{in}\times K^3 \times C_{out}\times W\times H \times T
$$
