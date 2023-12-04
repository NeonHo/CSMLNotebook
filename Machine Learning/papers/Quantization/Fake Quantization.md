All in all, fake quantization is use the quantization error between $w_{quant}$ and $w_{raw}$ as the loss, 
apply gradient descent to update the weight to make it be friendly to minimum the loss value.
As for the no gradient quantization weights, we use STE to estimate the gradient.
# Simple Procedure for Symmetric Quantization

Insert the fake quantization nodes.

## Forward

	input -> float -> int -> float
Simulate the quantization error
## Backward
use STE to estimate the gradient. [[Straight Through Estimator (STE)]]

$$
\begin{array}{lr}
\bar{v}=round(clip(\frac{v}{s}, -Q_N, Q_P))\\
\hat{v}=\bar{v}\times s
\end{array}
$$

The 1st error is a rounding error.
The 2nd error is a clipping error.
The whole error is: $|\hat{v}-v|$.

## After the backward
The weights are updated after the backward.
So we will re-estimate the quantization parameter: $s$.
$$
s=\frac{|v|_{max}}{Q_P}
$$
