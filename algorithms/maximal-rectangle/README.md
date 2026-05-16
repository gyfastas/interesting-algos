# 最大矩形 (Maximal Rectangle)

> LeetCode 85 — [Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

## 问题描述

给定一个只包含 `'0'` 和 `'1'` 的二维矩阵，找出只包含 `'1'` 的最大矩形，并返回其面积。

**示例**：

```
输入:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出: 6
解释: 第2-3行、第2-4列构成的 2×3 矩形，面积 = 6
```

## 直觉分析

如果矩阵只有一行，问题退化为：找连续 `1` 的最长子数组 → 线性扫描即可。

如果有多行，关键观察是：**以第 $i$ 行为底边的最大矩形**，等价于以 `heights` 数组为高度的柱状图中的最大矩形。

其中 `heights[j]` = 第 $j$ 列向上连续 `1` 的个数。

**转化思路**：
- 第 1 行：`[1,0,1,0,0]` → 柱状图高度
- 第 2 行：`[2,0,2,1,1]` → 第 2 列断裂（`0` 上方即使之前有高度也清零）
- 第 3 行：`[3,1,3,2,2]` → 继续向上累积
- 对每行的柱状图求最大矩形 → 取全局最大

## 数学建模

### 子问题：柱状图中最大的矩形

给定 $n$ 个柱子的高度 $h_0, h_1, \ldots, h_{n-1}$，求最大矩形面积。

**关键问题**：对于柱子 $i$，以它为高的最大矩形能有多宽？

答案是：向左右扩展，直到遇到比 $h_i$ 矮的柱子为止。

$$\text{宽度} = \text{右边第一个更矮的索引} - \text{左边第一个更矮的索引} - 1$$

**单调栈**：维护一个高度**单调递增**的栈。当遇到更矮的柱子时，弹出栈顶计算面积。

### 单调栈算法

```
遍历每个位置 i（末尾补 0）:
  while 栈非空 且 当前高度 < 栈顶高度:
    弹出栈顶索引 top
    高度 = heights[top]
    宽度 = i - 新栈顶索引 - 1   (栈空时宽度 = i)
    更新最大面积
  将 i 入栈
```

**为什么正确？**
- 栈中索引对应的高度单调递增
- 弹出栈顶时，新栈顶就是左边第一个比它矮的柱子
- 当前位置 $i$ 就是右边第一个比它矮的柱子
- 因此宽度 = $i - \text{新栈顶} - 1$ 恰好是以该高度能扩展的最大宽度

### 时间复杂度分析

- 每个索引入栈一次、出栈一次
- 总操作 $O(n)$

### 应用到原问题

逐行维护 `heights` 数组：

$$\text{heights}[j] = \begin{cases} \text{heights}[j] + 1 & \text{if } \text{matrix}[i][j] = \text{'1'} \\ 0 & \text{if } \text{matrix}[i][j] = \text{'0'} \end{cases}$$

对每行调用柱状图算法，时间 $O(m \cdot n)$。

## 代码实现

### 柱状图最大矩形（LeetCode 84）

```python
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    n = len(heights)
    for i in range(n + 1):
        h = heights[i] if i < n else 0  # 末尾补 0
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area
```

### 最大矩形（LeetCode 85）

```python
def maximal_rectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0
    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        max_area = max(max_area, largest_rectangle_area(heights))
    return max_area
```

### DP 解法（另一种思路）

逐行维护每个位置的 `height`、`left`、`right`：

- `height[j]`：向上连续 `1` 的个数
- `left[j]`：以当前行为底，第 $j$ 列能向左延伸的最左边界
- `right[j]`：以当前行为底，第 $j$ 列能向右延伸的最右边界

当前行以 $(i,j)$ 为底边的最大矩形面积：

$$\text{area} = \text{height}[j] \times (\text{right}[j] - \text{left}[j])$$

## 复杂度分析

| 方法 | 时间 | 空间 |
|------|------|------|
| 暴力枚举 | $O(m^2 \cdot n^2)$ | $O(m \cdot n)$ |
| 单调栈 | $O(m \cdot n)$ | $O(n)$ |
| DP | $O(m \cdot n)$ | $O(n)$ |

## 动画演示

> 打开 `animation.html` 查看交互动画

动画包含：
- **矩阵可视化**：`0` 和 `1` 的网格，最大矩形高亮显示
- **逐行 heights**：实时计算每行的柱状图高度
- **单调栈过程**：柱子弹入/弹出动画，面积计算高亮
- **DP 对比**：同时展示 DP 的 left/right/height 变化
- **交互控制**：选择不同矩阵、单步执行/自动播放、速度调节

## 答案与总结

**核心代码**：

```python
def maximal_rectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    heights = [0] * len(matrix[0])
    max_area = 0
    for row in matrix:
        for j, val in enumerate(row):
            heights[j] = heights[j] + 1 if val == '1' else 0
        max_area = max(max_area, largest_rectangle_area(heights))
    return max_area
```

### 核心 Insight

1. **降维**：二维问题 → 一维问题（逐行转化为柱状图）。
2. **单调栈**：$O(n)$ 求柱状图最大矩形，关键是找到每个柱子左右第一个更矮的柱子。
3. **断裂处理**：遇到 `0` 时 `heights[j]` 清零，相当于柱子高度归零，自动切断上方连续区域。
4. **与 LeetCode 84 的关系**：本题是 84 题的直接扩展，每一行都是独立的柱状图子问题。
