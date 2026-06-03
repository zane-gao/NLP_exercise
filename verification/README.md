# 最小工程验证

本目录把两份报告中写到的“最小工程验证”落成可运行 Python 代码。它不是完整训练框架，而是用于证明每个关键判断都能被一个小实验检查。

## 一键运行

在仓库根目录执行：

```bash
python3 -m verification.run_all
```

输出 JSON 中 `passed: true` 表示两讲验证均通过。

## 测试入口

```bash
python3 -m unittest discover -s tests -v
```

## 第 2 讲覆盖点

- toy CBOW 训练：验证 loss 下降。
- 近邻聚类：验证 `king/queen`、`cat/dog` 这类相似上下文词靠近。
- 负采样成本：用输出打分次数比较完整 softmax 与 negative sampling。
- subword OOV：验证纯词表查不到 OOV 时，字符 n-gram 仍可组合出可比较向量。

## 第 5 讲覆盖点

- BIO 修复：验证非法 `I-X` 转移被修复。
- 左右上下文：比较只看左侧与可看右侧的实体 F1。
- 字符特征：验证后缀信息能帮助未登录实体。
- 文档记忆：验证前文全称能帮助后文简称识别。
