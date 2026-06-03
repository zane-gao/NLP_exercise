# 单隐藏层 MLP 版朴素 CBOW 伪代码

## 设计目标

用上下文词预测中心词。算法部分允许调用库函数，例如 `softmax`、`cross_entropy`、`optimizer.step`，但训练数据构造、前向传播和参数更新逻辑需要明确写出。

## 符号

- 语料：$D=[w_1,w_2,\ldots,w_T]$
- 词表大小：$V$
- 窗口半径：$c$
- 词向量维度：$d$
- 隐藏层维度：$m$
- 输入 embedding：$E\in \mathbb{R}^{V\times d}$
- 隐藏层参数：$W_h\in \mathbb{R}^{m\times d}, b_h\in \mathbb{R}^{m}$
- 输出层参数：$W_o\in \mathbb{R}^{V\times m}, b_o\in \mathbb{R}^{V}$

## 伪代码

```text
Algorithm: One-Hidden-Layer-CBOW

Input:
  corpus D = [w_1, ..., w_T]
  vocabulary vocab, window radius c
  embedding dim d, hidden dim m
  learning rate eta, epochs K

Output:
  word embedding matrix E

1. Build vocab:
   word_to_id = {word -> integer id}
   id_to_word = {integer id -> word}

2. Initialize parameters:
   E   ~ Normal(0, 0.01) with shape [V, d]
   W_h ~ XavierInit([m, d])
   b_h = zeros([m])
   W_o ~ XavierInit([V, m])
   b_o = zeros([V])

3. Build training pairs:
   pairs = []
   for t from 1 to T:
       context_ids = []
       for j from t-c to t+c:
           if j != t and 1 <= j <= T:
               context_ids.append(word_to_id[w_j])
       if context_ids is not empty:
           target_id = word_to_id[w_t]
           pairs.append((context_ids, target_id))

4. Train:
   for epoch from 1 to K:
       shuffle(pairs)
       total_loss = 0

       for (context_ids, target_id) in pairs:
           # 前向传播：上下文词共享同一个 embedding 表
           context_vectors = E[context_ids]              # shape: [2c, d]
           x = mean(context_vectors, axis=0)             # shape: [d]

           # 单隐藏层 MLP；非线性使模型不只是线性词频统计
           h_pre = W_h @ x + b_h                         # shape: [m]
           h = tanh(h_pre)                               # shape: [m]

           logits = W_o @ h + b_o                        # shape: [V]
           prob = softmax(logits)                        # shape: [V]
           loss = cross_entropy(prob, target_id)

           # 反向传播：可调用自动微分，也可手写梯度
           gradients = backward(loss, [E, W_h, b_h, W_o, b_o])

           # 参数更新
           for param in [E, W_h, b_h, W_o, b_o]:
               param = param - eta * gradients[param]

           total_loss += loss

       print("epoch =", epoch, "loss =", total_loss / len(pairs))

5. Return E
```

## Mini-batch 版本

朴素版本逐样本更新，便于理解；实际可把多个样本 padding 成 batch。关键是把每个样本的上下文 embedding 做 masked mean：

$$
x_i=\frac{\sum_j M_{ij}E[w_{ij}]}{\sum_j M_{ij}}
$$

其中 $M_{ij}$ 表示第 $i$ 个样本第 $j$ 个上下文位置是否有效。

## Skip-Gram 对照

Skip-Gram 只改变训练样本方向：

```text
for each center word w_t:
    for each context word w_j in window:
        input  = one_hot(w_t)
        target = w_j
        maximize log p(w_j | w_t)
```

CBOW 更像“用多个线索猜一个词”，训练更快；Skip-Gram 更像“用一个词解释周围词”，对低频词通常更友好。

## 复杂度

若使用完整 softmax，每个样本输出层复杂度为 $O(Vm)$。当词表很大时，这一步成为主要瓶颈，因此实际 word2vec 常使用 negative sampling 或 hierarchical softmax。
