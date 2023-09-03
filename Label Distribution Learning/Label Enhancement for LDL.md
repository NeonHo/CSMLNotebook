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

| Sign               | means                                                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| $x$                | an instance                                                                                                           |
| $y$                | a possible label                                                                                                      |
| $l_x^y\in \{0,1\}$ | whether $y$ describe $x$, called logical label, reflects the logical relationship between the label and the instance. |

But the logical label is limited by the exclusive choice, it answers that **which label**
## 2.2. 