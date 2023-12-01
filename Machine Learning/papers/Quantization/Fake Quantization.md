# Simple Procedure

Insert the fake quantization nodes.

## Forward

	input -> float -> int -> float
simulate the quantization error
## Backward
STE

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
The weights is updated after the backward.

