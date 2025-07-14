U-Net中的skip-connection有以下几个作用：

1. [**空间域信息的保留**：对于分割任务，空间域信息非常重要。网络的encoder部分通过各个pooling层已经把特征图分辨率降得非常小了，这一点不利于精确的分割mask生成。通过skip-connection可以把较浅的卷积层特征引过来，那些特征分辨率较高，且层数浅，会含有比较丰富的low-level信息，更利于生成分割mask](https://zhuanlan.zhihu.com/p/314943727)[1](https://zhuanlan.zhihu.com/p/314943727)。
    
2. [**高分辨率细节信息的保留**：增加了skip connection结构的U-Net，能够使得网络在每一级的上采样过程中，将编码器对应位置的特征图在通道上进行融合。通过底层特征与高层特征的融合，网络能够保留更多高层特征图蕴含的高分辨率细节信息，从而提高了图像分割精度](https://blog.csdn.net/qq_42148951/article/details/106605837)[2](https://blog.csdn.net/qq_42148951/article/details/106605837)[3](https://blog.csdn.net/weixin_43135178/article/details/119976995)。
    
3. [**梯度消失和网络退化问题的缓解**：从Resnet最早引入skip-connection的角度看，这种跳跃连接可以有效的减少梯度消失和网络退化问题，使训练更容易。直观上理解可以认为BP的时候，深层的梯度可以更容易的传回浅层](https://www.zhihu.com/question/358839822)[4](https://www.zhihu.com/question/358839822)。
    

以上就是U-Net中的skip-connection的主要作用。希望对你有所帮助！