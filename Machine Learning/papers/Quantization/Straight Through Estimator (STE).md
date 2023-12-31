# Forward pass
$$
\begin{array}{lr}
X_{quant}=Round(Clip(X_{raw}, Q_{min}, Q_{max}))\\
\end{array}
$$
# Backward pass
$$
\frac{\partial Loss}{\partial X_{raw}}=\frac{\partial Loss}{\partial X_q}
$$
# Theory
[[Deterministic quantization & Stochastic quantization]]
To regard the expectation of the Stochastic Quantization as the function value for the backward in the Deterministic quantization.
Because the quantized value $X_{quant}$ is vertical, the partial is infinity.
The expectation of the quantized value $X_{quant}$ in stochastic quantization is normal, the partial is a good choice.

# STE for Binarization

[[Deterministic quantization & Stochastic quantization]]

## Deterministic

$$
\frac{\partial x_{binarize}}{\partial x_{raw}}=\frac{\partial clip(x_{raw},-1, 1)}{\partial x_{raw}}=\begin{cases}1&x_{raw}\in(-1,1]\\-1&otherwise\end{cases}
$$

## Stochastic
$$
\frac{\partial \mathbb{E} (x_{binarize})}{\partial x_{raw}}=\frac{\partial clip(x_{raw}, -1, 1)}{\partial x_{raw}}
$$

# General STE
Multi level quantization function cannot only use one STE.
Each Level need an STE.
Regard multi-level quantization functions as multiple binary quantization.
