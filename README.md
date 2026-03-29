# 有趣的算法题 🎯

精选算法题目的中文深度解析，每道题配有动画演示和完整推导。

## 题目列表

| # | 题目 | 类型 | 难度 |
|---|------|------|------|
| 001 | [马与4×4棋盘](./algorithms/horse-4x4-chessboard/) | 马尔可夫链、模拟 | ⭐⭐⭐ |
| 002 | [LLM Beam Search](./algorithms/llm-beam-search/) | 搜索算法、解码策略 | ⭐⭐ |
| 003 | [手写多头注意力 MHA/GQA/MLA](./algorithms/multi-head-attention/) | 深度学习、注意力机制 | ⭐⭐ |
| 004 | [手写 LayerNorm & RMSNorm](./algorithms/rmsnorm-layernorm/) | 归一化、深度学习基础 | ⭐ |
| 005 | [Adam / AdamW / Muon 优化器](./algorithms/adam-adamw-muon/) | 优化器、训练策略 | ⭐⭐ |
| 006 | [手写 SwiGLU 激活函数](./algorithms/swiglu-activation/) | 激活函数、FFN 架构 | ⭐ |
| 007 | [手写数值稳定的 Softmax](./algorithms/stable-softmax/) | 数值稳定性、Softmax | ⭐ |
| 008 | [手写 Cross Entropy Loss](./algorithms/cross-entropy-loss/) | 损失函数、信息论 | ⭐ |
| 009 | [PPO / GRPO / DPO 对比](./algorithms/ppo-grpo-dpo/) | RLHF、强化学习 | ⭐⭐⭐ |
| 010 | [手写 MSE Loss](./algorithms/mse-loss/) | 损失函数、数值稳定性 | ⭐ |
| 011 | [LoRA 低秩适配](./algorithms/lora/) | 参数高效微调、低秩分解 | ⭐⭐ |
| 012 | [BPE Tokenizer](./algorithms/bpe-tokenizer/) | 分词算法、BPE vs BBPE | ⭐⭐ |
| 013 | [Top-p & Top-k 采样](./algorithms/top-p-top-k/) | 解码策略、采样算法 | ⭐ |
| 014 | [K-Means 聚类](./algorithms/k-means/) | 无监督学习、聚类算法 | ⭐ |
| 015 | [快速排序](./algorithms/quick-sort/) | 排序、分治、Partition | ⭐⭐ |
| 016 | [多元线性回归](./algorithms/linear-regression/) | 梯度下降、Forward & Backward | ⭐ |

## 目录结构

```
interesting-algos/
└── algorithms/
    └── <题目名>/
        ├── README.md       # 中文题解（含动画）
        ├── solution.py     # Python 解法
        └── animation.html  # 交互式动画
```

## 如何使用

每道题的 `README.md` 包含：
- 题目描述与直觉分析
- 数学建模与推导过程
- 代码实现（附注释）
- 交互式动画（打开 `animation.html`）

打开动画：

```bash
open algorithms/<题目>/animation.html
```

## Vibe Coding 添加新题

在这个 repo 里直接告诉 Claude：

> 增加题目：XXX问题，类型：YYY，描述：ZZZ

Claude 会自动生成 README、代码和动画。参考 [`.claude/skills/add-algorithm.md`](./.claude/skills/add-algorithm.md)。
