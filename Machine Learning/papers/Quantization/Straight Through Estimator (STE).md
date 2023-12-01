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
