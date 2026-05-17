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
| 016 | [MLP 多元回归（两层感知机）](./algorithms/linear-regression/) | 手写 Autograd、反向传播、链式法则 | ⭐⭐⭐ |
| 017 | [VLM DataLoader](./algorithms/vlm-dataloader/) | 多进程、生产者-消费者队列 | ⭐⭐ |
| 018 | [VLM 训练流水线](./algorithms/vlm-training-pipeline/) | ViT 全量+LLM Micro-Batch+Recompute | ⭐⭐⭐ |
| 019 | [寻找两个有序数组的中位数](./algorithms/median-of-two-sorted-arrays/) | 二分查找、数组 | ⭐⭐⭐ |
| 020 | [洛伦兹吸引子](./algorithms/lorenz-attractor/) | 动力系统、混沌理论、数值模拟 | ⭐⭐⭐ |
| 021 | [优惠券收集问题](./algorithms/coupon-collector/) | 概率论、期望、调和数 | ⭐⭐⭐ |
| 022 | [MHA 手写 Forward & Backward（困难版）](./algorithms/multi-head-attention-with-backward/) | 反向传播、注意力机制、Autograd | ⭐⭐⭐⭐ |
| 023 | [访问所有节点的最短路径](./algorithms/shortest-path-visiting-all-nodes/) | 图论、状态压缩、BFS | ⭐⭐⭐⭐ |
| 024 | [LRU Cache（线程安全进阶版）](./algorithms/lru-cache/) | 数据结构、哈希表、双向链表、并发 | ⭐⭐⭐ |
| 025 | [矩阵中的最长递增路径](./algorithms/longest-increasing-path-in-matrix/) | 动态规划、记忆化搜索、DFS | ⭐⭐⭐⭐ |
| 026 | [MoE Transformer Layer Forward & Backward](./algorithms/moe-transformer/) | 混合专家、手写反向传播、负载均衡 | ⭐⭐⭐⭐⭐ |
| 027 | [生日问题](./algorithms/birthday-problem/) | 概率论、组合数学、反直觉 | ⭐⭐ |
| 028 | [等待模式问题 — 抛硬币与收集问题](./algorithms/waiting-for-patterns/) | 概率论、马尔可夫链、期望值 | ⭐⭐⭐ |
| 029 | [坏芯片检测问题](./algorithms/chip-testing-problem/) | 组合数学、信息论、决策树 | ⭐⭐⭐ |
| 030 | [用 Rand7 构造 Rand10](./algorithms/rand7-to-rand10/) | 拒绝采样、均匀分布、概率论 | ⭐⭐⭐ |
| 031 | [最大矩形](./algorithms/maximal-rectangle/) | 单调栈、动态规划、柱状图 | ⭐⭐⭐⭐ |
| 032 | [投机解码工程实现](./algorithms/speculative-decoding/) | LLM推理、拒绝采样、工程架构 | ⭐⭐⭐⭐ |
| 033 | [投机解码进阶: Exact vs Fallback](./algorithms/speculative-decoding-advanced/) | LLM推理优化、工程权衡、分布偏差 | ⭐⭐⭐⭐⭐ |
| 034 | [DPO 训练 MHA Transformer](./algorithms/dpo-training-mha/) | DPO、偏好优化、梯度推导、收敛性 | ⭐⭐⭐⭐⭐ |
| 035 | [GRPO 训练 MHA Transformer](./algorithms/grpo-training-mha/) | GRPO、Group Sampling、PPO-Clip、KL约束、梯度推导 | ⭐⭐⭐⭐⭐ |
| 036 | [单词接龙 (Word Ladder)](./algorithms/word-ladder/) | BFS、双向 BFS、图论最短路径 | ⭐⭐⭐ |

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
