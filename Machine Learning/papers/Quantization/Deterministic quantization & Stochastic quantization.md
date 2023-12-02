# Binarization
## Deterministic
$$
x=\begin{cases}-1&if\quad x\leq 0\\
1&otherwise\end{cases}
$$
## Stochastic
$$
x=\begin{cases}-1&with\quad prob\quad p=clip\left( \dfrac{1-x}{2},0,1\right) \\
1 &with\quad prob\quad p=clip\left( \dfrac{1+x}{2},0,1\right) \end{cases}
$$
