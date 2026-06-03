# 基于 RNN 的命名实体识别伪代码

## 设计目标

给定句子 $x_1,\ldots,x_n$，预测每个 token 的 BIO 标签 $y_1,\ldots,y_n$。这里先写最朴素的单向 RNN 版本，再说明如何替换为 BiRNN 或 CRF。

## 符号

- 词表大小：$V$
- 标签集合大小：$L$
- 词向量维度：$d$
- 隐状态维度：$h$
- 词 embedding：$E\in \mathbb{R}^{V\times d}$
- RNN 参数：$W_x\in\mathbb{R}^{h\times d}, W_h\in\mathbb{R}^{h\times h}, b\in\mathbb{R}^{h}$
- 分类器参数：$W_y\in\mathbb{R}^{L\times h}, b_y\in\mathbb{R}^{L}$

## 训练伪代码

```text
Algorithm: RNN-NER-Train

Input:
  labeled sentences S = [(tokens_1, tags_1), ..., (tokens_N, tags_N)]
  word vocabulary, tag vocabulary
  embedding dim d, hidden dim h
  learning rate eta, epochs K

Output:
  trained parameters E, W_x, W_h, b, W_y, b_y

1. Build vocabularies:
   token_to_id = {token -> id}
   tag_to_id   = {BIO tag -> id}

2. Initialize parameters:
   E   ~ Normal(0, 0.01)
   W_x ~ XavierInit([h, d])
   W_h ~ XavierInit([h, h])
   b   = zeros([h])
   W_y ~ XavierInit([L, h])
   b_y = zeros([L])

3. Train:
   for epoch from 1 to K:
       shuffle(S)
       total_loss = 0

       for (tokens, gold_tags) in S:
           n = length(tokens)
           h_prev = zeros([h])
           losses = []

           for t from 1 to n:
               token_id = token_to_id.get(tokens[t], UNK)
               x_t = E[token_id]

               # RNN 用前一时刻状态保存左侧上下文
               h_t = tanh(W_x @ x_t + W_h @ h_prev + b)

               logits_t = W_y @ h_t + b_y
               prob_t = softmax(logits_t)

               tag_id = tag_to_id[gold_tags[t]]
               losses.append(cross_entropy(prob_t, tag_id))

               h_prev = h_t

           loss = sum(losses) / n
           gradients = backward(loss, [E, W_x, W_h, b, W_y, b_y])

           for param in [E, W_x, W_h, b, W_y, b_y]:
               param = param - eta * gradients[param]

           total_loss += loss

       print("epoch =", epoch, "loss =", total_loss / len(S))
```

## 解码伪代码

```text
Algorithm: RNN-NER-Decode

Input:
  tokens = [x_1, ..., x_n]
  trained parameters

Output:
  predicted BIO tags

1. h_prev = zeros([h])
2. predictions = []

3. for t from 1 to n:
       x_t = E[token_to_id.get(tokens[t], UNK)]
       h_t = tanh(W_x @ x_t + W_h @ h_prev + b)
       logits_t = W_y @ h_t + b_y
       tag_t = argmax(softmax(logits_t))
       predictions.append(tag_t)
       h_prev = h_t

4. Repair illegal BIO transitions:
       if tag_t is I-X and previous tag is not B-X or I-X:
           replace tag_t with B-X or O according to validation rule

5. return predictions
```

## 评测伪代码

```text
Algorithm: Entity-Level-Evaluation

Input:
  gold tag sequences, predicted tag sequences

1. Convert BIO tags into entity spans:
       B-PER I-PER -> (start, end, PER)

2. Count:
       TP = predicted spans exactly matching gold spans
       FP = predicted spans not in gold
       FN = gold spans missed by prediction

3. Compute:
       precision = TP / (TP + FP)
       recall    = TP / (TP + FN)
       F1        = 2 * precision * recall / (precision + recall)
```

## 加入字符特征

对中文或形态丰富语言，可以给每个词额外编码字符：

```text
char_vectors = CharCNN(chars_of_token)
x_t = concat(word_embedding, char_vectors)
```

字符特征能缓解 OOV，也能帮助模型识别“有限公司”“大学”“先生”等实体形态线索。

## 升级为 BiRNN

单向 RNN 只有左上下文。BiRNN 增加反向状态：

$$
\overrightarrow{h_t}=RNN_f(x_t,\overrightarrow{h_{t-1}})
$$

$$
\overleftarrow{h_t}=RNN_b(x_t,\overleftarrow{h_{t+1}})
$$

最终分类向量为：

$$
h_t=[\overrightarrow{h_t};\overleftarrow{h_t}]
$$

这样每个位置都同时看到左侧和右侧上下文。

## 升级为 CRF 解码

逐词 softmax 会独立预测标签，可能出现非法序列，例如 `O I-ORG`。CRF 引入标签转移分数：

$$
score(x,y)=\sum_t s_t[y_t]+\sum_t A[y_{t-1},y_t]
$$

训练最大化整句标签序列概率，解码用 Viterbi 找全局最优标签序列。这样能显式约束 BIO 合法性。
