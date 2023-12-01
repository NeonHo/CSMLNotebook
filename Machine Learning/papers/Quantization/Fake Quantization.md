# Simple Procedure

Insert the fake quantization nodes.

## forward

	input -> float -> int -> float
simulate the quantization error
## backward
STE

$$
\begin{array}{lr}
\bar{v}=round(clip(\frac{v}{s}, -Q_N, Q_P))\\
\hat{v}=\bar{v}\times s
\end{array}
$$

The 1st error is rounding error.
The 2nd error is clipping error.
the whole error is: $|\hat{v}-v|$.

