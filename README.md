# 有趣的算法题 🎯

精选算法题目的中文深度解析，每道题配有动画演示和完整推导。

## 题目列表

| # | 题目 | 类型 | 难度 |
|---|------|------|------|
| 001 | [马与4×4棋盘](./algorithms/horse-4x4-chessboard/) | 马尔可夫链、模拟 | ⭐⭐⭐ |

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
