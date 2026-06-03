# 第 5 讲：循环神经网络语言模型及序列到序列学习

本文件夹回答两个问题：

- 课内：手写一个基于 RNN 的命名实体识别伪代码。
- 探索：如何更好地同时利用词的上下文信息，并设计相应方案。

命名实体识别可以看成序列标注任务：输入一个词序列，输出同长度的标签序列。

## 任务定义

输入句子：

$$
x_1,x_2,\ldots,x_n
$$

输出 BIO 标签：

$$
y_1,y_2,\ldots,y_n
$$

示例：

```text
高展沛   就读   于   北京大学
B-PER    O      O    B-ORG
```

常见标签包括：

- `B-PER`, `I-PER`：人名实体。
- `B-ORG`, `I-ORG`：组织机构。
- `B-LOC`, `I-LOC`：地点。
- `O`：非实体。

## 基线结构

```text
token ids / char ids
      |
embedding
      |
RNN hidden states
      |
linear classifier
      |
BIO tag distribution
```

单向 RNN 只能利用左侧历史，因此它是最小可解释基线。更强方案应升级为双向 RNN 或 BiLSTM-CRF，以同时利用左上下文、右上下文和标签转移约束。

## 文件说明

- [rnn_ner_pseudocode.md](rnn_ner_pseudocode.md)：基于 RNN 的 NER 训练、解码与评测伪代码。
- [context_design.md](context_design.md)：上下文增强方案，从 BiRNN 到多粒度上下文融合。
- [../verification/lecture05_validation.py](../verification/lecture05_validation.py)：BIO 修复、上下文增强和文档记忆的最小工程验证。

## 最小工程验证

在仓库根目录运行：

```bash
python3 -m unittest tests.test_lecture05_validation -v
```

该验证会检查：

- 非法 `I-X` 标签能否被修复成合法 BIO 序列。
- 双向上下文是否优于只看左侧上下文。
- 字符后缀规则是否能帮助 OOV 实体。
- 文档级记忆是否能识别后文简称实体。

## 核心结论

NER 不只是逐词分类。实体边界和实体类型往往由左右上下文共同决定，标签之间也有合法转移关系。因此，强系统应同时建模词语上下文、字符形态、句内双向依赖、标签依赖和文档级线索。
