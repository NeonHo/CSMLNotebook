TF-IDF（词频-逆文档频率）
- 对词袋模型的改进，通过“词频（TF）× 逆文档频率（IDF）”加权，降低常见词的权重（如“的”“是”），突出稀有词的重要性。
- 公式：$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$，其中 $TF(t, d)$ 为词 $t$ 在文档 $d$ 中的出现频率，$IDF(t) = \log\left(\frac{\text{总文档数}}{\text{包含词 } t \text{ 的文档数}}\right)$。
- TF-IDF是一种用于信息检索与数据挖掘的常用加权技术。TF意思是词频(Term Frequency),IDF意思是逆文本频率指数(Inverse Document Frequency)。TF-IDF是一种统计方法,用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加,但同时会随着它在语料库中出现的频率成反比下降。

TFC:对文本长度进行归一化处理后的TF-IDF。
ITC:在TFC基础上,对tf的对数值取代tf。