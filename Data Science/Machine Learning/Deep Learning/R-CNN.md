# Selective Search
2000 regions of  interests
[[Selective Search]]
# Feature Extract

Conv 2000 times

# Bounding Box Regression and Classification

# Fast R-CNN

Extract feature first, then selective search to product 2000 proposals.

# Faster R-CNN

RPN (Region Proposal Network) to product proposals according to feature maps.
在 Faster R-CNN 中，Region Proposal Network (RPN) 是一个关键组件，用于生成候选区域（Region Proposals）。与传统方法（如 Selective Search）不同，RPN 是一个神经网络，能够直接在卷积特征图上生成候选区域。RPN 和检测网络共享卷积特征图，使得候选区域的生成过程高效且紧密结合。以下是 RPN 的训练过程的详细解释。
## RPN (Region Proposal Network)
### 1. RPN 网络结构

RPN 是一个轻量级的全卷积网络，其主要组件包括：

- **共享卷积层**：RPN 和 Faster R-CNN 的检测分支共享卷积层（通常是由主干网络如 ResNet 或 VGG 提供的特征图）。
  
- **3x3 卷积层**：在共享卷积层之后，RPN 使用一个 3x3 的卷积层来进一步提取局部特征。这个层的输出保留了空间位置的信息。

- **两个分支**：
  - **分类分支（cls）**：用于预测每个锚框（Anchor）是否包含前景目标（即该区域是否是一个候选区域）。
  - **边界框回归分支（reg）**：用于预测锚框相对于真实目标的边界框的修正（即预测边界框的四个偏移量）。

### 2. 生成锚框（Anchors）

- **锚框生成**：RPN 在特征图的每个位置生成一组锚框，这些锚框具有不同的尺寸和纵横比。通常，在每个位置生成 9 个锚框（3 个尺寸 × 3 个纵横比）。
  
- **锚框与真实目标的匹配**：
  - 将每个锚框与图像中的所有真实目标（Ground Truth boxes）进行匹配。
  - 如果锚框与某个真实目标的 IoU（Intersection over Union）大于一定阈值（通常是 0.7），则将该锚框标记为正样本（前景）。
  - 如果锚框与所有真实目标的 IoU 小于一定阈值（通常是 0.3），则将该锚框标记为负样本（背景）。
  - 介于这两个阈值之间的锚框不用于训练。

### 3. RPN 的训练目标

RPN 的训练目标包括两个部分：

- **分类损失**：用于区分前景和背景。这部分损失通常使用二元交叉熵损失（Binary Cross-Entropy Loss）。
  
- **边界框回归损失**：用于调整锚框，使其更接近真实目标的边界框。回归损失通常使用平滑的 L1 损失（Smooth L1 Loss）。

总的损失函数为：
$$
 L = \frac{1}{N_{\text{cls}}} \sum L_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum p_i^* L_{\text{reg}}(t_i, t_i^*) 
$$
其中：

- \( $p_i$ \) 是锚框 \( $i$ \) 预测为前景的概率。
- \( $p_i^*$ \) 是锚框 \( $i$ \) 的真实标签（1 表示前景，0 表示背景）。
- \( $t_i$ \) 是锚框 \( $i$ \) 预测的边界框参数。
- \( $t_i^*$ \) 是锚框 \( $i$ \) 对应的真实目标的边界框参数。
- \( $L_{\text{cls}}$ \) 是分类损失（如二元交叉熵）。
- \( $L_{\text{reg}}$ \) 是回归损失（如平滑的 L1 损失）。
- \( $N_{\text{cls}}$ \) 和 \( $N_{\text{reg}}$ \) 分别是分类和回归损失的归一化因子。
- \( $\lambda$ \) 是一个权重因子，用于平衡分类损失和回归损失。

### 4. RPN 的端到端训练

- **端到端训练**：RPN 和 Faster R-CNN 的检测分支可以端到端地共同训练。RPN 生成的候选区域被直接用于后续的检测和分类任务。

- **交替训练**：在实际训练中，通常采用交替训练的策略：首先训练 RPN，然后将 RPN 生成的候选区域用于训练检测器，接着再用训练好的检测器反过来微调 RPN。

### 5. NMS 处理
在 RPN 生成大量候选区域后，通常会应用非极大值抑制（NMS）来减少冗余的候选区域。NMS 会移除那些与高置信度区域有很大重叠的低置信度候选区域，从而保留最有可能包含目标的区域。

### 总结

RPN 是 Faster R-CNN 中的核心组件，它通过学习的方式生成候选区域，并且这些候选区域与检测网络共享特征。这种设计使得候选区域生成过程更加高效，并显著提升了目标检测的速度和精度。RPN 的训练结合了分类和边界框回归任务，能够直接生成高质量的候选区域，为后续的目标检测提供了强有力的支持。

