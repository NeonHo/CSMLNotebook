"Masked Language Models"（MLMs）是一类用于自然语言处理（NLP）任务的模型，其特点是在输入文本中随机地遮蔽（mask）一些单词，然后尝试预测这些被遮蔽单词。这种模型的训练有助于学习单词之间的语义关系和上下文信息。

主要的 MLM 模型之一是 BERT（Bidirectional Encoder Representations from Transformers），它在训练时使用了遮蔽单词的策略。BERT 可以同时考虑上下文中的左侧和右侧信息，使得模型对整体文本的理解更为全面。

**MLM 的主要特点和步骤：**

1. **遮蔽输入：** 在训练时，随机选择输入文本中的一些单词并将它们遮蔽，或用特殊的标记（如 \[MASK\]）替换。这样模型在训练时需要预测这些被遮蔽的单词。

2. **双向上下文：** MLM 模型通常设计为能够捕捉左侧和右侧上下文的信息，与传统的从左到右的语言模型不同。

3. **预测遮蔽的单词：** 模型通过上下文信息尝试预测被遮蔽的单词。这个任务促使模型学会理解单词之间的语义关系和上下文语境。

4. **应用领域：** MLM 模型在各种 NLP 任务中都取得了显著的成功，包括文本分类、命名实体识别、问答等。预训练的 MLM 模型通常可以在下游任务上进行微调，从而提高性能。

MLM 模型的成功表明，通过适当的预训练策略，模型可以学到更加通用的语言表示，这对于处理多种 NLP 任务是有益的。BERT 和其后续的模型在 NLP 社区中取得了重大的影响。