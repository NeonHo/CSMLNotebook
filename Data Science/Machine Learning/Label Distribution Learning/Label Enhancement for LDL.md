# 1. Why we need Label Enhancement?

Most training sets:
1. only contain simple logical labels;
2. difficult to obtain the label distribution directly.
So we need to recover label distributions.
# 2. How to label enhance?
1. use topological information of the feature space.
2. use correlation among the labels.
Label enhancement (LE) is **to recover label distribution from logical labels**.
2 novel LE algorithms are as follows:
## 2.1. Graph Laplacian Label Enhancement (GLLE)

It is verified to be better than several existing LE methods.
### 2.1.1. Description Degree of the Possible Labels

| Sign               | means                                                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| $x$                | an instance                                                                                                           |
| $y$                | one of the possible labels                                                                                            |
| $l_x^y\in \{0,1\}$ | whether $y$ describe $x$, called logical label, reflects the logical relationship between the label and the instance. | 

But the logical label is limited by the exclusive choice, it answers that **which label** can describe the instance, not **the degree** to which $y$ describes $x$.

So we prefer the degree denoted as $d_x^y\in[0,1]$.

We regard it as the description degree of $y$ to $x$.
It will obey the rules:
$$
\sum_y d_x^y=1.0
$$
### 2.1.2. Label Distribution & LDL
**Label Distribution**: A real-valued vector constituted by the description degrees of all the labels.
- This describes the instance more **comprehensively** than logical labels.
- The learning process based on the datasets with Label Distribution called LDL.

## 2.2. 