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



### 卷积神经网络（CNN）详解

卷积神经网络（Convolutional Neural Network, CNN）是深度学习的核心模型之一，特别适合处理具有网格结构的数据（如图像、音频）。它通过卷积层、池化层和全连接层的组合，自动提取数据特征，在计算机视觉领域取得了突破性进展。

---

### 一、CNN 的核心结构与组件

#### 1. 基本组件
- **卷积层（Convolutional Layer）**：通过卷积核提取局部特征，实现权值共享，大幅减少参数。  
- **激活函数（Activation Function）**：引入非线性，常用 ReLU（Rectified Linear Unit）。  
- **池化层（Pooling Layer）**：降低特征维度，增强特征的平移不变性，常用 Max Pooling。  
- **全连接层（Fully Connected Layer）**：将提取的特征映射到分类或回归结果。

#### 2. 数据流动
输入图像 → 卷积层 + 激活函数 → 池化层 → 重复多次 → 全连接层 → 输出

---

### 二、卷积层：特征提取的核心
[[卷积]]
#### 1. 卷积计算
- **局部连接**：卷积核（滤波器）在输入上滑动，每次处理一个局部区域。  
- **权值共享**：同一卷积核在不同位置使用相同参数，提取相同类型的特征（如边缘、纹理）。  
- **多通道**：输入通道（如 RGB 三通道）与卷积核通道对应，卷积后求和得到单通道输出。多个卷积核生成多个特征图。

#### 2. 关键参数
- **卷积核大小**：常用 3×3、5×5，决定感受野。  
- **步长（Stride）**：控制卷积核滑动步长，影响输出尺寸。  
- **填充（Padding）**：在输入边缘补 0，保持输出尺寸与输入一致（Same Padding）或缩小（Valid Padding）。

---

### 三、池化层：降维和特征增强
[[池化]]
#### 1. 作用
- **降维**：减少特征图尺寸，降低计算复杂度。  
- **增强鲁棒性**：通过聚合局部信息，使特征对平移、旋转等变换更不敏感。

#### 2. 常用池化方式
- **最大池化（Max Pooling）**：取局部区域最大值，保留最显著特征。  
- **平均池化（Average Pooling）**：取局部区域平均值，保留整体信息。  
- **全局池化（Global Pooling）**：将整个特征图压缩为一个值，常用于减少参数。

---

### 四、激活函数与批归一化

#### 1. 激活函数
- **ReLU**：$f(x) = \max(0, x)$，计算高效，缓解梯度消失。  
- **Leaky ReLU**：改进 ReLU，在 $x<0$ 时保留小梯度，避免“神经元死亡”。  
- **Swish**：$f(x) = x \cdot \text{Sigmoid}(x)$，平滑非线性，性能优于 ReLU。

#### 2. 批归一化（Batch Normalization）
- **作用**：加速训练，缓解梯度消失/爆炸，提高模型稳定性。  
- **原理**：对每一批次数据进行归一化，使输入分布更稳定。


通常情况下，CNN的每层卷积之后**会紧跟激活函数**，这是主流网络设计中的常见范式，但并非绝对刚性的规则。 
##### 原因解析： 
1. **引入非线性能力** 卷积操作本质上是线性变换（加权求和），而激活函数（如ReLU、Sigmoid、Tanh等）的作用是为网络注入非线性能力。只有通过非线性激活，CNN才能拟合复杂的非线性关系（如图像中的边缘、纹理、语义等层次化特征），否则无论堆叠多少卷积层，整体仍等价于一个线性模型，无法处理复杂任务。 
2. **主流网络的设计实践** 从经典的LeNet、AlexNet到现代的ResNet、YOLO等，几乎都遵循“卷积层 → 激活函数”的顺序（部分网络会在两者之间或之后加入批归一化BN，即“卷积 → BN → 激活”，如ResNet）。这种设计经过大量实验验证，能有效提升特征提取能力。 
##### 例外情况： 
少数场景下可能不直接紧跟激活函数，例如： 
- **网络最后一层卷积**：若任务是回归（如预测像素值）或生成模型（如GAN的生成器输出），可能省略激活函数，直接输出原始卷积结果。
- **特定架构设计**：某些网络（如ResNet的 shortcut 连接中，部分1×1卷积可能不单独加激活，而是与主分支的激活结果合并后再激活）。 
但总体而言，**卷积层后紧跟激活函数是CNN的常规设计**，其核心目的是通过非线性变换增强网络的特征表达能力。
---

### 五、CNN 的典型架构

| 架构            | 年份   | 关键创新           | 结构特点                        |
| ------------- | ---- | -------------- | --------------------------- |
| **LeNet-5**   | 1998 | 首个成功 CNN       | 卷积 → 池化 → 卷积 → 池化 → 全连接     |
| **AlexNet**   | 2012 | ReLU + Dropout | 8 层，ImageNet 冠军，开启深度学习复兴    |
| **VGG**       | 2014 | 3×3 卷积堆叠       | 16/19 层，全卷积设计               |
| **ResNet**    | 2015 | 残差连接           | 跳层连接，解决梯度消失，可训练 >1000 层     |
| **Inception** | 2014 | 多尺度并行          | Inception 模块，1×1+3×3+5×5 并行 |
[[CNN发展]]

---

### 六、CNN 的优势与应用场景

#### 1. 优势
- **参数效率高**：权值共享大幅减少参数，降低过拟合风险。  
- **局部特征提取**：自动学习图像局部特征（如边缘、纹理），无需手动特征工程。  
- **平移不变性**：对目标在图像中的位置不敏感，适合目标检测与识别。

#### 2. 应用场景
- **图像分类**：ImageNet、人脸识别  
- **目标检测**：[[YOLO]]、Faster R-CNN  
- **语义分割**：U-Net、DeepLab  
- **视频分析**：动作识别、视频理解  
- **其他领域**：语音识别、文本 CNN

---

### 七、PyTorch 代码示例：CIFAR-10 分类 CNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 数据增强 + 归一化
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=100,
                                          shuffle=False, num_workers=2)

# 定义 CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,  3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(256 * 4 * 4, 512)
        self.fc2   = nn.Linear(512, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 训练
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 6):
    model.train()
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 测试
    model.eval(); correct = total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"Epoch {epoch}: Accuracy = {100.*correct/total:.2f}%")
```

---

### 八、总结

CNN 通过卷积、池化和全连接层的组合，实现了高效的图像特征自动提取。其核心优势在于参数效率高、局部特征提取能力强和对平移变换的鲁棒性。从早期的 LeNet 到现代的 ResNet 和 Transformer-CNN 混合架构，CNN 持续演进，成为计算机视觉的基石。深入理解 CNN 原理与实践，是进入深度学习领域的重要一步。
