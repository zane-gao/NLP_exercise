# 探索：如何更好地同时利用词的上下文信息

NER 的关键不是“每个词自己像不像实体”，而是“它在当前句子和当前文档里承担什么角色”。因此，上下文利用应从单向历史扩展到多粒度、多方向、多层约束。

## 单向 RNN 的不足

单向 RNN 在位置 $t$ 的状态为：

$$
h_t=f(x_t,h_{t-1})
$$

它能利用左侧历史，但不能直接看到右侧词。例如：

```text
苹果 发布 新 机
我 买 了 苹果
```

看到“苹果”本身不足以判断是组织还是水果，右侧词“发布”或左侧词“买了”都很关键。单向模型会把后一侧信息延迟到未来位置，当前 token 的分类仍然不足。

## 方案一：BiRNN/BiLSTM 同时看左右上下文

最直接改进是双向结构：

```text
left context  -> forward RNN  -> h_f[t]
right context -> backward RNN -> h_b[t]
concat(h_f[t], h_b[t]) -> classifier
```

LSTM/GRU 比 vanilla RNN 更适合长句，因为门控机制能缓解梯度消失。对 NER，推荐基线是：

```text
word embedding + char feature
      |
BiLSTM / BiGRU
      |
linear emission scores
      |
CRF
```

## 方案二：CRF 使用标签上下文

上下文不仅来自词，也来自标签序列。BIO 标签具有强结构：

- `I-PER` 前面通常应是 `B-PER` 或 `I-PER`。
- `I-ORG` 不应直接跟在 `B-LOC` 后面。
- 一个实体 span 内部类型应一致。

CRF 用转移矩阵学习这些规律，避免逐词分类产生局部合理但整体非法的标签序列。

## 方案三：字符级与子词级上下文

词级表示处理语义，字符级表示处理形态。中文 NER 中，很多实体边界由字面模式提示：

- 机构：大学、公司、集团、委员会。
- 地点：省、市、区、路。
- 人名：常见姓氏、音译模式。

可用 CharCNN 或 CharBiLSTM 得到字符表示：

$$
z_t=[e_t^{word};e_t^{char}]
$$

这样模型即使遇到未登录词，也能从字符片段猜测实体类型。

## 方案四：文档级上下文

句内上下文仍然不够。很多实体第一次出现时全称明确，后文会用简称或代词：

```text
北京大学 发布 招生 简章。
北大 表示 ...
```

可设计文档级记忆：

```text
for each sentence in document:
    run sentence-level BiLSTM-CRF
    extract high-confidence entity spans
    update entity memory table
    feed memory features into later sentence tagging
```

记忆表可保存实体字符串、类型分布、最近出现位置和别名。后续句子遇到相似字符串时，将记忆特征拼接到 token 表示中。

## 方案五：预训练上下文化表示

更强方案是直接使用 BERT/RoBERTa/MacBERT 等上下文化 encoder：

```text
tokens
  |
pretrained Transformer
  |
contextual vectors
  |
linear layer + CRF
```

Transformer 的自注意力让每个位置直接关注句中所有词，比 RNN 更擅长并行建模长距离依赖。对于中文 NER，常用做法是：

- 使用中文预训练模型。
- 对 subword 只取首 subword 向量或聚合 subword 向量。
- 顶层接 CRF 约束 BIO 序列。

## 推荐的创新方案：多粒度上下文融合 NER

最终方案不把上下文限定为一个来源，而是融合五种线索：

```text
token ids
char ids
document memory
      |
word embedding + char encoder + memory feature
      |
BiLSTM / Transformer encoder
      |
gated fusion
      |
CRF decoding
```

门控融合：

$$
g_t=\sigma(W_g[z_t^{local};z_t^{doc}]+b_g)
$$

$$
z_t=g_t\odot z_t^{local}+(1-g_t)\odot z_t^{doc}
$$

其中 $z_t^{local}$ 表示句内上下文，$z_t^{doc}$ 表示文档级实体记忆。门控的意义是：当当前句子证据足够强时依赖局部上下文；当当前句子模糊时调用文档记忆。

## 验证指标

推荐从三个层次评测：

- token-level accuracy：检查逐词标签是否合理。
- entity-level precision/recall/F1：检查实体 span 是否完整匹配。
- OOV entity F1 与 long-distance alias F1：专门检验字符特征和文档级上下文是否真的发挥作用。

## 本质判断

CBOW 学到的是“词通常和谁一起出现”，而 NER 需要判断“这个词此刻在句子和文档中扮演什么角色”。因此，更好的上下文利用方案必须同时回答三件事：

- 左右词如何改变当前词的实体类型。
- 相邻标签如何限制合法实体边界。
- 文档中已经出现过的实体如何帮助后文消歧。
