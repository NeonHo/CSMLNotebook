<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Apriori算法中有事物和事物平均宽度
[[Apriori算法]]中确实有“事务”和“事务平均宽度”这两个术语。

- **事务（Transaction）**：指数据集中每一条记录或样本，通常表示一次购买或一次事件，包含若干个项（items）。
- **事务平均宽度（Average Transaction Width）**：指所有事务中包含项的平均数量，即平均每个事务包含多少个项。

这两个概念是Apriori算法计算复杂度和性能分析中的重要影响因素。具体来说：

- 事务数越多，算法需要扫描和处理的数据量越大，计算时间也越长。
- 事务平均宽度越大，意味着每条事务中包含的项越多，频繁项集的最大长度可能增加，产生的候选项集和支持度计数的复杂度也会增加，从而提高算法的计算复杂度。

这些内容在多篇技术文章和教学资料中都有详细说明，例如阿里云开发者社区和CSDN博客均提到“事务平均宽度”是影响Apriori算法性能的重要因素[^1][^2][^3]。

---

简要总结：
Apriori算法中，“事务”指的是数据集中的每条记录，“事务平均宽度”指的是每条记录中项的平均数量，这两个术语用于描述数据集特征并影响算法复杂度。

<div style="text-align: center">⁂</div>

[^1]: https://developer.aliyun.com/article/1085024

[^2]: https://blog.csdn.net/qq_40618816/article/details/106699243

[^3]: https://www.cdaglobal.com/tk/question/19818.html

[^4]: https://blog.csdn.net/u010454729/article/details/49078505

[^5]: https://blog.sina.com.cn/s/blog_6002b97001014niv.html

[^6]: https://journal.lnpu.edu.cn/CN/article/downloadArticleFile.do?attachType=PDF\&id=1886

[^7]: https://www.cdaglobal.com/tk/question/14928.html

[^8]: https://www.dqxxkx.cn/CN/10.12082/dqxxkx.2018.170266

