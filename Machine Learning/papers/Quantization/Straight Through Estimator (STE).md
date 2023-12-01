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
Regard the expectation of the Stochastic Quantization as the function value for the backward in the Deterministic quantization.
Because the quantized value $X_{quant}$ is vertical, the partial is infinity.