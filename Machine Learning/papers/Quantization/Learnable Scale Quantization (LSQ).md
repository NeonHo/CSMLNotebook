Training the quantization scale with the parameters.

Update the scale with backward propagation.


$$
fake\ quant({v}) = round(clip(\frac{v}{s}, -Q_N, Q_P)) \times s=\begin{cases}-Q_N\times s& \frac{v}{s}\le -Q_N\\round(\frac{v}{s})\times s &-Q_N < \frac{v}{s} < Q_P\\ Q_P\times s& \frac{v}{s}\ge Q_P\end{cases}
$$

$$
\frac{\partial \hat{v}}{\partial s}=\begin{cases}-Q_N & \frac{v}{s} \le -Q_N\\round(\frac{v}{s})+\frac{\partial(\frac{v}{s})}{\partial s}\times s & -Q_N < \frac{v}{s} < Q_P\\ Q_P & \frac{v}{s}\ge Q_P\end{cases}
$$


![[Drawing 2023-12-02 17.52.08.excalidraw]]

In the LSQ, the gradients are changed suddenly too in the transition point, so it will be more similar to the real quantization than the ==QIL== and the ==PACT==.

![[微信图片_20231202181911.jpg]]
The author believe that the gradient from the scale and 
$$

$$