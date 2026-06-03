"""第 2 讲词表示的最小工程验证。

这些函数不是为了替代完整训练框架，而是把报告中的验证点落成可重复运行的
小实验：loss 是否下降、近邻是否形成语义簇、负采样是否减少输出计算量、
subword 是否能处理 OOV。
"""

import math
import random
from collections import Counter


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cosine(a, b):
    denom = math.sqrt(_dot(a, a)) * math.sqrt(_dot(b, b))
    if denom == 0:
        return 0.0
    return _dot(a, b) / denom


def _softmax(logits):
    max_logit = max(logits)
    exp_values = [math.exp(x - max_logit) for x in logits]
    total = sum(exp_values)
    return [x / total for x in exp_values]


def _matvec(matrix, vector):
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def _add_bias(vector, bias):
    return [x + b for x, b in zip(vector, bias)]


def _build_pairs(tokens, word_to_id, window):
    pairs = []
    for t, token in enumerate(tokens):
        context = []
        for j in range(t - window, t + window + 1):
            if 0 <= j < len(tokens) and j != t:
                context.append(word_to_id[tokens[j]])
        if context:
            pairs.append((context, word_to_id[token]))
    return pairs


def _average_loss(pairs, params):
    total_loss = 0.0
    for context, target in pairs:
        _, _, _, probs = _forward(context, params)
        total_loss += -math.log(max(probs[target], 1e-12))
    return total_loss / len(pairs)


def _forward(context, params):
    embeddings, hidden_w, hidden_b, output_w, output_b = params
    dim = len(embeddings[0])
    x = [0.0] * dim
    for idx in context:
        for k in range(dim):
            x[k] += embeddings[idx][k]
    x = [value / len(context) for value in x]

    hidden_pre = _add_bias(_matvec(hidden_w, x), hidden_b)
    hidden = [math.tanh(value) for value in hidden_pre]
    logits = _add_bias(_matvec(output_w, hidden), output_b)
    return x, hidden_pre, hidden, _softmax(logits)


def _train_cbow(pairs, vocab_size, seed, epochs):
    rng = random.Random(seed)
    dim = 10
    hidden_dim = 8
    learning_rate = 0.045

    # 小随机初始化足够暴露“训练是否真的在动”的趋势。
    embeddings = [[rng.uniform(-0.08, 0.08) for _ in range(dim)] for _ in range(vocab_size)]
    hidden_w = [[rng.uniform(-0.08, 0.08) for _ in range(dim)] for _ in range(hidden_dim)]
    hidden_b = [0.0 for _ in range(hidden_dim)]
    output_w = [[rng.uniform(-0.08, 0.08) for _ in range(hidden_dim)] for _ in range(vocab_size)]
    output_b = [0.0 for _ in range(vocab_size)]
    params = (embeddings, hidden_w, hidden_b, output_w, output_b)

    initial_loss = _average_loss(pairs, params)
    for _ in range(epochs):
        rng.shuffle(pairs)
        for context, target in pairs:
            x, _, hidden, probs = _forward(context, params)

            d_logits = probs[:]
            d_logits[target] -= 1.0

            old_output_w = [row[:] for row in output_w]
            for out_id in range(vocab_size):
                for j in range(hidden_dim):
                    output_w[out_id][j] -= learning_rate * d_logits[out_id] * hidden[j]
                output_b[out_id] -= learning_rate * d_logits[out_id]

            d_hidden = [0.0 for _ in range(hidden_dim)]
            for out_id in range(vocab_size):
                for j in range(hidden_dim):
                    d_hidden[j] += old_output_w[out_id][j] * d_logits[out_id]

            d_hidden_pre = [d_hidden[j] * (1.0 - hidden[j] * hidden[j]) for j in range(hidden_dim)]
            old_hidden_w = [row[:] for row in hidden_w]
            for j in range(hidden_dim):
                for k in range(dim):
                    hidden_w[j][k] -= learning_rate * d_hidden_pre[j] * x[k]
                hidden_b[j] -= learning_rate * d_hidden_pre[j]

            d_x = [0.0 for _ in range(dim)]
            for j in range(hidden_dim):
                for k in range(dim):
                    d_x[k] += old_hidden_w[j][k] * d_hidden_pre[j]

            # 上下文词共享平均池化梯度，验证 embedding 确实参与更新。
            for idx in context:
                for k in range(dim):
                    embeddings[idx][k] -= learning_rate * d_x[k] / len(context)

    final_loss = _average_loss(pairs, params)
    return params, initial_loss, final_loss


def _nearest_neighbors(words, embeddings, top_k=3):
    neighbors = {}
    for word, idx in words.items():
        scored = []
        for other, other_idx in words.items():
            if other == word:
                continue
            scored.append((other, _cosine(embeddings[idx], embeddings[other_idx])))
        scored.sort(key=lambda item: item[1], reverse=True)
        neighbors[word] = [word for word, _ in scored[:top_k]]
    return neighbors


def run_cbow_toy_validation(seed=7, epochs=80):
    """验证 toy CBOW 训练能降低 loss，并让相似上下文词靠近。"""
    royal = "royal king queen crown palace throne".split()
    pet = "pet cat dog kitten puppy animal home".split()
    tech = "apple phone device launch company market".split()
    corpus = []
    for _ in range(20):
        corpus.extend("royal king crown palace throne".split())
        corpus.extend("royal queen crown palace throne".split())
        corpus.extend("king queen royal crown".split())
        corpus.extend("pet cat dog animal home".split())
        corpus.extend("pet dog cat animal home".split())
        corpus.extend("cat kitten pet animal".split())
        corpus.extend("dog puppy pet animal".split())
        corpus.extend("apple company launch phone device".split())
        corpus.extend("apple phone device market".split())

    vocab = sorted(set(royal + pet + tech))
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    pairs = _build_pairs(corpus, word_to_id, window=2)
    params, initial_loss, final_loss = _train_cbow(pairs, len(vocab), seed, epochs)
    neighbors = _nearest_neighbors(word_to_id, params[0])
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "nearest_neighbors": neighbors,
        "pages_expected": "engineering",
    }


def run_negative_sampling_cost_validation(vocab_size=5000, positive_pairs=120, negatives_per_pair=5):
    """用输出打分次数验证负采样比完整 softmax 更可扩展。"""
    full_softmax_score_count = vocab_size * positive_pairs
    negative_sampling_score_count = positive_pairs * (1 + negatives_per_pair)
    return {
        "full_softmax_score_count": full_softmax_score_count,
        "negative_sampling_score_count": negative_sampling_score_count,
        "saving_ratio": negative_sampling_score_count / full_softmax_score_count,
    }


def _char_ngrams(word, min_n=3, max_n=4):
    ngrams = []
    for n in range(min_n, max_n + 1):
        if len(word) < n:
            continue
        for start in range(0, len(word) - n + 1):
            ngrams.append(word[start : start + n])
    return ngrams


def _sparse_vector(word):
    counts = Counter(_char_ngrams(word))
    return dict(counts)


def _sparse_cosine(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(key, 0.0) * b.get(key, 0.0) for key in keys)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def run_subword_oov_validation():
    """验证纯词表查不到 OOV，而 subword 可以组合出可比较向量。"""
    known_vectors = {
        "alpha": _sparse_vector("alpha"),
        "tech": _sparse_vector("tech"),
        "banana": _sparse_vector("banana"),
    }
    oov_word = "alphatech"
    plain_lookup = known_vectors.get(oov_word)
    subword_vector = _sparse_vector(oov_word)
    return {
        "plain_lookup": plain_lookup,
        "subword_vector": subword_vector,
        "subword_similarity_to_related": _sparse_cosine(subword_vector, known_vectors["tech"]),
        "subword_similarity_to_unrelated": _sparse_cosine(subword_vector, known_vectors["banana"]),
    }
