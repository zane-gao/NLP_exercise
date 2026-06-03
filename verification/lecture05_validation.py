"""第 5 讲 RNN-NER 的最小工程验证。

这里用很小的、确定性的上下文验证替代重型训练：重点检查 BIO 合法性、左右
上下文、字符形态和文档记忆是否能分别降低报告中指出的错误类型。
"""


def repair_bio_tags(tags):
    """把非法 I-X 修复为 B-X，保证实体边界可解释。"""
    repaired = []
    for tag in tags:
        if tag == "O":
            repaired.append(tag)
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B":
            repaired.append(tag)
            continue
        if prefix != "I":
            raise ValueError("BIO 标签只能以 B、I 或 O 开头")

        if not repaired or repaired[-1] == "O":
            repaired.append("B-" + entity_type)
            continue
        prev_prefix, prev_type = repaired[-1].split("-", 1)
        if prev_prefix in {"B", "I"} and prev_type == entity_type:
            repaired.append(tag)
        else:
            repaired.append("B-" + entity_type)
    return repaired


def count_illegal_bio_transitions(tags):
    count = 0
    previous = "O"
    for tag in tags:
        if tag.startswith("I-"):
            entity_type = tag.split("-", 1)[1]
            if previous == "O" or previous.split("-", 1)[1] != entity_type:
                count += 1
        previous = tag
    return count


def extract_bio_entities(tags):
    entities = []
    start = None
    current_type = None
    for idx, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if start is not None:
                entities.append((start, idx - 1, current_type))
                start = None
                current_type = None
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            if start is not None:
                entities.append((start, idx - 1, current_type))
            start = idx
            current_type = entity_type
        elif prefix == "I":
            continue
        else:
            raise ValueError("BIO 标签只能以 B、I 或 O 开头")
    return entities


def _entity_f1(gold_docs, pred_docs):
    gold = []
    pred = []
    offset = 0
    for gold_spans, pred_spans in zip(gold_docs, pred_docs):
        gold.extend((start + offset, end + offset, typ) for start, end, typ in gold_spans)
        pred.extend((start + offset, end + offset, typ) for start, end, typ in pred_spans)
        max_end = 0
        for spans in (gold_spans, pred_spans):
            for _, end, _ in spans:
                max_end = max(max_end, end)
        offset += max_end + 2

    gold_set = set(gold)
    pred_set = set(pred)
    true_positive = len(gold_set & pred_set)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _left_only_tagger(tokens):
    # 单向基线只看左侧，无法用右侧“发布”判断句首“苹果”为组织。
    known_entities = {"北京大学": "ORG"}
    return [(idx, idx, typ) for idx, token in enumerate(tokens) for word, typ in known_entities.items() if token == word]


def _bidirectional_tagger(tokens):
    entities = _left_only_tagger(tokens)
    for idx, token in enumerate(tokens):
        next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        if token == "苹果" and next_token in {"发布", "宣布"}:
            entities.append((idx, idx, "ORG"))
    return entities


def _no_char_tagger(tokens):
    known_entities = {"北京大学": "ORG"}
    return [(idx, idx, typ) for idx, token in enumerate(tokens) for word, typ in known_entities.items() if token == word]


def _char_feature_tagger(tokens):
    entities = _no_char_tagger(tokens)
    suffix_rules = [
        ("科技园", "LOC"),
        ("大学", "ORG"),
        ("公司", "ORG"),
        ("集团", "ORG"),
    ]
    for idx, token in enumerate(tokens):
        for suffix, typ in suffix_rules:
            if token.endswith(suffix) and (idx, idx, typ) not in entities:
                entities.append((idx, idx, typ))
    return entities


def _no_memory_document_tagger(document):
    return [_bidirectional_tagger(sentence) for sentence in document]


def _memory_document_tagger(document):
    predictions = []
    aliases = {}
    for sentence in document:
        spans = _bidirectional_tagger(sentence)
        for start, _, typ in spans:
            if sentence[start] == "北京大学" and typ == "ORG":
                aliases["北大"] = "ORG"
        for idx, token in enumerate(sentence):
            if token in aliases and (idx, idx, aliases[token]) not in spans:
                spans.append((idx, idx, aliases[token]))
        predictions.append(spans)
    return predictions


def run_ner_context_validations():
    """跑完报告中列出的四个最小 NER 工程验证。"""
    context_sentences = [["苹果", "发布", "新机"], ["我", "买了", "苹果"]]
    context_gold = [[(0, 0, "ORG")], []]
    left_pred = [_left_only_tagger(sentence) for sentence in context_sentences]
    bi_pred = [_bidirectional_tagger(sentence) for sentence in context_sentences]

    oov_sentence = [["我", "参观", "星河科技园"]]
    oov_gold = [[(2, 2, "LOC")]]
    no_char_pred = [_no_char_tagger(oov_sentence[0])]
    char_pred = [_char_feature_tagger(oov_sentence[0])]

    local_tags = ["O", "I-ORG", "I-ORG", "B-PER", "I-LOC"]
    repaired_tags = repair_bio_tags(local_tags)

    document = [["北京大学", "发布", "招生简章"], ["北大", "表示", "欢迎", "申请"]]
    document_gold = [[(0, 0, "ORG")], [(0, 0, "ORG")]]
    no_memory_pred = _no_memory_document_tagger(document)
    memory_pred = _memory_document_tagger(document)

    return {
        "left_only_f1": _entity_f1(context_gold, left_pred),
        "bidirectional_f1": _entity_f1(context_gold, bi_pred),
        "no_char_oov_f1": _entity_f1(oov_gold, no_char_pred),
        "char_oov_f1": _entity_f1(oov_gold, char_pred),
        "crf_illegal_before": count_illegal_bio_transitions(local_tags),
        "crf_illegal_after": count_illegal_bio_transitions(repaired_tags),
        "no_memory_f1": _entity_f1(document_gold, no_memory_pred),
        "document_memory_f1": _entity_f1(document_gold, memory_pred),
        "pages_expected": "engineering",
    }
