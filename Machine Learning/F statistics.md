<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# F statistics是什么指标？

F统计量（F-Statistic）是统计学中用于假设检验的一种重要指标，主要用于比较两个样本方差或多个回归模型的拟合优度，其分布遵循F分布。

---

## F统计量的定义与作用

- **定义**：F统计量是两个方差估计值的比值，通常表示为组间方差与组内方差的比率。
- **主要用途**：
    - **方差齐性检验**：检验两组或多组数据的方差是否相等。
    - **方差分析（ANOVA）**：判断多个样本均值是否存在显著差异。
    - **回归模型显著性检验**：检验回归模型中所有自变量对因变量的联合影响是否显著。

---

## F统计量的计算公式（以回归分析为例）

$$
F = \frac{SSR / p}{SSE / (n - p - 1)}
$$

- **SSR（回归平方和）**：模型解释的变异部分。
- **SSE（残差平方和）**：模型未解释的变异部分。
- **p**：自变量个数（自由度1）。
- **n**：样本总数。
![[R-squre#2. **回归平方和（SSR）**]]

---

## 解释与应用

- 计算得到的F值与F分布表中的临界值比较：
    - 若 \$ F \geq F_{临界值} \$，拒绝原假设，说明组间差异显著或模型有效。
    - 若 \$ F < F_{临界值} \$，不能拒绝原假设，说明组间差异不显著或模型无效。
- F统计量与对应的p值紧密相关，p值越小，拒绝原假设的证据越强。
- 在模型比较中，F统计量用于判断更复杂模型是否显著优于简化模型。

---

## 总结

F统计量是通过比较不同组间方差与组内方差的比率，评估组间差异或模型整体显著性的统计量。它是方差分析和回归分析中检验假设的重要工具。

---

（信息来源：[^1][^2][^4][^5]）

<div style="text-align: center">⁂</div>

[^1]: https://blog.csdn.net/IT_ORACLE/article/details/145199730

[^2]: https://blog.csdn.net/xuemanqianshan/article/details/137466845

[^3]: https://imgtec.eetrend.com/blog/2019/100043215.html

[^4]: https://zh-cn.statisticseasily.com/词汇表/f-统计量是什么/

[^5]: https://zh-cn.statisticseasily.com/词汇表/什么是-f-比率-理解-f-统计量/

[^6]: https://www.cnblogs.com/massquantity/p/10486904.html

[^7]: https://indico.ihep.ac.cn/event/17315/attachments/64130/75717/机器学习中的概率与统计.pdf

[^8]: https://www.lianxh.cn/details/1273.html

