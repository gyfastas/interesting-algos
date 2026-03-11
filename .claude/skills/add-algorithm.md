# Skill: 添加新算法题

当用户说"增加题目"或"添加算法"时，按以下步骤执行：

## 输入信息收集

需要用户提供：
- 题目名称（会转为 kebab-case 作为目录名）
- 类型（如：马尔可夫链、动态规划、图论、贪心等）
- 题目描述

## 执行步骤

### 1. 先求解问题

在写任何文件之前，用 Python 验证答案：
- 暴力/模拟验证
- 数学推导
- Monte Carlo 验证

确保数字正确再写 README。

### 2. 创建目录

```bash
mkdir -p algorithms/<kebab-case-名称>
```

### 3. 写 solution.py

包含：
- 数学解法（线性方程组或递推）
- Monte Carlo 模拟验证
- 清晰的注释（中文）
- 打印所有中间状态和最终答案

### 4. 写 README.md

按模板结构：完整数学推导，LaTeX 公式，步骤图示（用 ASCII 或表格）

### 5. 写 animation.html

自包含 HTML，包含：
- 状态/棋盘可视化
- 核心算法动画
- Monte Carlo 实时模拟
- 交互控制（播放/暂停/速度调节）

### 6. 更新根 README.md

在题目列表中添加新行。

## 动画模板

```html
<!DOCTYPE html>
<html>
<head>
  <style>/* 深色主题 */</style>
</head>
<body>
  <!-- 可视化区域 -->
  <!-- 控制面板 -->
  <script>
    // 状态机 + 动画循环
    // Monte Carlo 模拟
  </script>
</body>
</html>
```

## 注意事项

- 题目必须有明确的数学答案，先验证再写文档
- 动画要直观体现核心思想，而不只是花哨
- README 要让没学过相关知识的人也能看懂（循序渐进）
