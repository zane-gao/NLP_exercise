# 第 2 讲：词表示

本文件夹回答两个问题：

- 课内：构建一个只有一个隐藏层的 MLP 网络，实现朴素 CBOW 或 Skip-Gram，并给出手写伪代码。
- 探索：CBOW/Skip-Gram 存在什么问题，如何解决。

本次选择 CBOW 作为主线，因为它能直接展示“用上下文预测中心词”的词表示学习思想；Skip-Gram 可视作把预测方向反过来，即用中心词预测上下文词。

## 任务定义

给定语料序列：

$$
w_1,w_2,\ldots,w_T
$$

设窗口半径为 $c$。对位置 $t$，CBOW 使用上下文集合：

$$
C_t=\{w_{t-c},\ldots,w_{t-1},w_{t+1},\ldots,w_{t+c}\}
$$

预测中心词 $w_t$。模型目标是最大化：

$$
\sum_t \log p(w_t\mid C_t)
$$

## 单隐藏层 MLP 结构

```text
上下文词 one-hot
      |
共享 embedding 查表
      |
平均池化得到上下文向量
      |
隐藏层：h = tanh(W_h x + b_h)
      |
输出层：softmax(W_o h + b_o)
      |
预测中心词
```

其中 embedding 矩阵既是模型参数，也是最终词表示。隐藏层不是为了堆复杂度，而是让“上下文均值”经过一次非线性变换，从而表达更复杂的词语共现模式。

## 输入输出

- 输入：分词后的语料、词表、窗口大小、embedding 维度、隐藏层维度、学习率和训练轮数。
- 输出：词向量矩阵 $E$，以及可用于预测中心词的 MLP 参数。

## 文件说明

- [cbow_mlp_pseudocode.md](cbow_mlp_pseudocode.md)：朴素 CBOW 的训练伪代码。
- [exploration.md](exploration.md)：CBOW/Skip-Gram 的局限与解决路径。

## 核心结论

CBOW/Skip-Gram 的价值在于把离散词压缩进连续空间，但它只从局部窗口和静态词表中学习。真正的突破方向不是单纯把 MLP 加深，而是让表示具备更强的采样效率、子词泛化、多义区分和上下文化能力。
