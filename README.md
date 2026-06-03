# NLP Exercise

作者：高展沛

本仓库完成自然语言处理课程中第 2 讲“词表示”和第 5 讲“循环神经网络语言模型及序列到序列学习”的课堂伪代码与探索任务。整体组织思路是：先从局部窗口里的静态词表示出发，再推进到能显式利用左右上下文和文档上下文的序列标注模型。

## 目录

```text
.
├── lecture02_word_representation/
│   ├── README.md
│   ├── cbow_mlp_pseudocode.md
│   ├── exploration.md
│   ├── report.tex
│   └── report.pdf
├── lecture05_rnn_ner/
│   ├── README.md
│   ├── rnn_ner_pseudocode.md
│   ├── context_design.md
│   ├── report.tex
│   └── report.pdf
├── verification/
│   ├── README.md
│   ├── lecture02_validation.py
│   ├── lecture05_validation.py
│   └── run_all.py
└── tests/
    ├── test_lecture02_validation.py
    ├── test_lecture05_validation.py
    └── test_run_all_validation.py
```

## 最短阅读路径

1. 阅读 [lecture02_word_representation/README.md](lecture02_word_representation/README.md)，理解 CBOW/Skip-Gram 的输入输出和训练闭环。
2. 阅读 [lecture02_word_representation/cbow_mlp_pseudocode.md](lecture02_word_representation/cbow_mlp_pseudocode.md)，查看单隐藏层 MLP 版朴素 CBOW 伪代码。
3. 阅读 [lecture02_word_representation/exploration.md](lecture02_word_representation/exploration.md)，理解静态词向量的问题和解决谱系。
4. 阅读 [lecture05_rnn_ner/README.md](lecture05_rnn_ner/README.md)，理解 RNN NER 的任务设置。
5. 阅读 [lecture05_rnn_ner/rnn_ner_pseudocode.md](lecture05_rnn_ner/rnn_ner_pseudocode.md)，查看基于 RNN 的命名实体识别伪代码。
6. 阅读 [lecture05_rnn_ner/context_design.md](lecture05_rnn_ner/context_design.md)，查看上下文增强方案。
7. 阅读 [lecture02_word_representation/report.pdf](lecture02_word_representation/report.pdf)，查看第 2 讲不超过两页的最终报告。
8. 阅读 [lecture05_rnn_ner/report.pdf](lecture05_rnn_ner/report.pdf)，查看第 5 讲不超过两页的最终报告。
9. 运行 [verification/README.md](verification/README.md) 中的最小工程验证，检查报告中的验证点是否可执行。

## 最小工程验证

一键运行全部验证：

```bash
python3 -m verification.run_all
```

运行测试套件：

```bash
python3 -m unittest discover -s tests -v
```

验证覆盖：

- 第 2 讲：toy CBOW loss 下降、近邻语义聚类、完整 softmax 与负采样输出打分次数对比、subword OOV。
- 第 5 讲：非法 BIO 修复、单向上下文与双向上下文对比、字符特征 OOV、文档记忆简称识别。

## 报告编译

第 2 讲报告：

```bash
cd lecture02_word_representation
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
```

第 5 讲报告：

```bash
cd lecture05_rnn_ner
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
```

清理辅助文件：

```bash
cd lecture02_word_representation
latexmk -c
cd ../lecture05_rnn_ner
latexmk -c
```

## 验证记录

- LaTeX 报告使用 `xelatex` 编译。
- 每个作业独立成文：第 2 讲 `Pages = 2`，第 5 讲 `Pages = 2`。
- 已补充最小 Python 工程验证；验证代码只用于复现实验检查点，不替代完整训练框架。

## 提交说明

本地首次初始化为 `main` 分支，远端地址为：

```bash
git@github.com:zane-gao/NLP_exercise.git
```

若后续继续补充实验代码，建议保持当前分层：算法伪代码、探索文档、报告源文件分开维护，避免把课堂解释、实现脚本和报告排版混在同一文件中。
