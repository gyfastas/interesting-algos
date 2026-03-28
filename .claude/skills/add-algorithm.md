---
name: add-algorithm
description: 添加新算法题到 repo，包含题解、代码和交互动画
user_invocable: true
---

# 添加新算法题

用户通过 `/add-algorithm <题目描述>` 调用。参数格式灵活，可以是一句话描述，也可以包含题目名称、类型、详细描述。

## Step 1: 解析用户输入

从用户输入中提取：
- **题目名称** — 简短中文名（如"马与4×4棋盘"）
- **目录名** — 转为 kebab-case 英文（如 `horse-4x4-chessboard`）
- **类型标签** — 如：马尔可夫链、动态规划、图论、贪心、搜索算法等
- **难度** — 根据题目复杂度判断 ⭐~⭐⭐⭐⭐

如果信息不完整，用 AskUserQuestion 补充。

## Step 2: 先求解，验证答案

在写任何文件之前，**必须先用 Python 验证答案正确**：

1. 写一个临时脚本或直接 Bash 运行，用暴力/模拟方法求解
2. 用数学推导得到解析解
3. 用 Monte Carlo 模拟交叉验证
4. 确认数字一致后再进入下一步

这一步是最重要的——错误的答案会让整篇题解失去意义。

## Step 3: 创建文件

按顺序创建以下文件：

### 3a. `algorithms/<dir>/solution.py`

```python
# 包含：
# - 数学解法（线性方程组 / 递推 / 高斯消元）
# - Monte Carlo 模拟验证
# - 清晰的中文注释
# - 打印中间状态和最终答案
# - if __name__ == "__main__" 入口
```

规范：
- Python 3.10+，不引入外部包（numpy 仅用于验证）
- 核心解法必须有纯 Python 实现

### 3b. `algorithms/<dir>/README.md`

严格按照以下模板结构：

```markdown
# 题目名称

## 问题描述
（清晰描述问题，让没有背景的读者也能理解）

## 直觉分析
（用类比或简单例子建立直觉）

## 数学建模
（定义状态、变量、转移关系）

## 求解过程
（完整推导，含 LaTeX 公式）

## 代码实现
（嵌入 solution.py 的核心代码，加解释）

## 动画演示
（说明动画展示了什么，链接到 animation.html）
> 打开 `animation.html` 查看交互动画

## 答案与总结
（最终答案 + 关键 insight）
```

公式格式：
- 行内：`$E_B = 8$`
- 块级：`$$E_B = 1 + \frac{2}{3}E_B + \frac{1}{3}E_I$$`

### 3c. `algorithms/<dir>/animation.html`

自包含 HTML 文件，零外部依赖，`open` 即可运行。

必须包含：
- **状态可视化** — 棋盘、图、树、或其他核心数据结构
- **算法动画** — 逐步演示核心算法过程
- **Monte Carlo 模拟** — 实时模拟按钮，显示收敛过程
- **交互控制** — 播放/暂停/速度调节/重置

设计规范：
- 深色背景 `#1a1a2e`
- 高亮色 `#e94560`（主）、`#0f3460`（辅）
- 响应式布局，桌面端优先
- 动画流畅，用 requestAnimationFrame

## Step 4: 运行验证

```bash
python3 algorithms/<dir>/solution.py
```

确认输出正确。

## Step 5: 更新根 README.md

在题目列表表格中添加新行，编号递增：

```markdown
| 00N | [题目名称](./algorithms/<dir>/) | 类型标签 | ⭐⭐ |
```

## Step 6: 用浏览器打开动画预览

```bash
open algorithms/<dir>/animation.html
```

## 注意事项

- **先验证再写文档** — 题目必须有明确的数学答案
- **动画要直观** — 体现核心思想，不只是花哨
- **README 要循序渐进** — 让没学过相关知识的人也能看懂
- **每一步完成后标记 Task 进度** — 用 TaskCreate/TaskUpdate 跟踪
