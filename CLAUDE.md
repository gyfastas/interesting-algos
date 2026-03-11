# CLAUDE.md — 算法题解析 Repo

这是一个用于讲解有趣算法题的 repo，面向中文读者，每道题配有动画演示。

## 目录规范

每道题放在 `algorithms/<kebab-case-名称>/` 下，包含：
- `README.md` — 中文完整题解（见下方模板）
- `solution.py` — Python 解法，有详细注释
- `animation.html` — 自包含的交互动画（纯 HTML/CSS/JS，无依赖）

## README 模板结构

```markdown
# 题目名称

## 问题描述
## 直觉分析
## 数学建模
## 求解过程（含公式推导）
## 代码实现
## 动画演示
## 答案与总结
```

## 动画设计规范

- 使用纯 HTML + CSS + JS，零外部依赖，可直接 `open` 打开
- 动画要有：棋盘/状态可视化、马尔可夫链/概率流、Monte Carlo 模拟按钮
- 配色：深色背景 (#1a1a2e)，高亮用 (#e94560 或 #0f3460)
- 要有暂停/播放控制

## 添加题目流程

1. 在 `algorithms/` 下创建新目录
2. 写 README.md（中文，含 LaTeX 公式用 `$...$` 包裹，注意 animation.html 是独立文件）
3. 写 solution.py（含测试用例和 Monte Carlo 验证）
4. 写 animation.html（自包含，可视化核心数学过程）
5. 更新根 README.md 的题目列表

## 数学公式

README 中的公式用标准 Markdown 数学格式：
- 行内：`$E_B = 8$`
- 块级：`$$E_B = 1 + \frac{2}{3}E_B + \frac{1}{3}E_I$$`

## 代码规范

- Python 3.10+，不引入外部包（numpy 可以用于验证）
- 核心解法要有纯 Python 实现（高斯消元或迭代）
- Monte Carlo 模拟作为验证
