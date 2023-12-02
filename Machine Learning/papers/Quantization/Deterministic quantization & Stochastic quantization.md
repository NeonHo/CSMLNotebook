# Binarization
## Deterministic
$$
x_{quant}=\begin{cases}-1&if\quad x_{raw}\leq 0\\
1&otherwise\end{cases}
$$
## Stochastic
$$
x_{quant}=\begin{cases}-1&with\quad prob\quad p=clip\left( \dfrac{1-x_{raw}}{2},0,1\right) \\
1 &with\quad prob\quad p=clip\left( \dfrac{1+x}{2},0,1\right) \end{cases}
$$
