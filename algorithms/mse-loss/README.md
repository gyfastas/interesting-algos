# 手写 MSE 损失函数

## 问题描述

**编写一个函数，输入实际值数组 `y_true` 和预测值数组 `y_pred`，返回均方误差。注意实现效率。**

MSE 是回归任务最基础的损失函数。公式很简单，但面试考的是：你能不能写出**高效、数值稳定**的版本。

## 数学定义

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

梯度（对 $\hat{y}_i$）：

$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2(\hat{y}_i - y_i)}{n}$$

梯度和误差成正比——误差越大，梯度越大，修正越猛。这也是 MSE 对异常值敏感的根本原因。

## 代码实现

### 最高效版本（一行）

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
```

效率分析：
- **时间** $O(n)$：单次遍历，`zip` + 生成器表达式不创建中间列表
- **空间** $O(1)$：生成器是惰性求值，不像 list comprehension 会分配 $O(n)$ 内存
- **除法只做一次**：在 `sum` 之后才除以 $n$，不是每个元素都除

### 对比：空间效率

```python
# ❌ O(n) 额外空间 — 创建了中间列表
sum([(t - p) ** 2 for t, p in zip(y_true, y_pred)]) / n

# ✅ O(1) 额外空间 — 生成器表达式
sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
```

方括号和圆括号的区别：`[...]` 是 list comprehension（立即生成完整列表），`(...)` 是 generator expression（惰性逐个产生）。

### PyTorch 版本

```python
def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return (y_true - y_pred).pow(2).mean()
```

一步向量化，等价于 `F.mse_loss(y_pred, y_true)`。

## 效率陷阱

### 1. 不要逐元素除以 n

```python
# ❌ 慢：n 次除法
total = 0
for i in range(n):
    total += (y_true[i] - y_pred[i]) ** 2 / n

# ✅ 快：1 次除法
total = 0
for i in range(n):
    total += (y_true[i] - y_pred[i]) ** 2
return total / n
```

除法比乘法慢约 3-5 倍。$n$ 次除法 vs 1 次除法，差别不可忽略。

### 2. 不要创建中间数组

```python
# ❌ 慢 + 浪费内存：创建 diff 数组、diff² 数组
diff = [t - p for t, p in zip(y_true, y_pred)]
sq = [d ** 2 for d in diff]
return sum(sq) / len(sq)

# ✅ 一次遍历搞定
return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
```

### 3. `zip` 比索引快

```python
# ❌ 慢：索引访问有边界检查开销
for i in range(n):
    total += (y_true[i] - y_pred[i]) ** 2

# ✅ 快：zip 迭代器直接访问
for t, p in zip(y_true, y_pred):
    total += (t - p) ** 2
```

Python 的 `list[i]` 每次都要做类型检查和边界检查，`zip` 迭代器绕过了这些开销。

## 数值稳定性

### 大数组求和的精度问题

当 $n$ 很大时，朴素求和会丢失精度（大数吃小数）：

```
sum = 1e16 + 0.001 + 0.001 + ... (100万个 0.001)
期望: 1e16 + 1000 = 10000000000001000
实际 float64: 10000000000001000.0  (可能 OK)
实际 float32: 10000000000000000.0  (1000 被吃了)
```

两��解决方案：

### Welford 在线算法

不累加总和，而是增量更新均值：

```python
def mse_stable(y_true, y_pred):
    mean_se = 0.0
    for i in range(len(y_true)):
        diff_sq = (y_true[i] - y_pred[i]) ** 2
        mean_se += (diff_sq - mean_se) / (i + 1)
    return mean_se
```

每步只做一次小数的加减，不会出现"大数 + 小数"的精度丢失。

### Kahan 补偿求和

用一个额外变量追踪丢失的低位精度：

```python
def mse_kahan(y_true, y_pred):
    total, comp = 0.0, 0.0
    for t, p in zip(y_true, y_pred):
        y = (t - p) ** 2 - comp
        new_total = total + y
        comp = (new_total - total) - y  # 补偿丢失的位
        total = new_total
    return total / len(y_true)
```

精度接近 float64，只多一次加减运算。Python 的 `math.fsum()` 内部就用类似的补偿算法。

## MSE vs MAE vs Huber

| | MSE | MAE | Huber |
|---|---|---|---|
| **公式** | $(y-\hat{y})^2$ | $\|y-\hat{y}\|$ | 分段：小误差用 MSE，大误差用 MAE |
| **梯度** | $2(y-\hat{y})$ | $\text{sign}(y-\hat{y})$ | 分段 |
| **异常值** | 极敏感（平方放大） | 鲁棒 | 鲁棒 |
| **零点梯度** | 0（平滑） | ±1（不可导） | 0（平滑） |
| **用途** | 通用回归 | 鲁棒回归 | 两者折中 |

**异常值示例**：

```
y_true = [1, 2, 3, 4, 100]   ← 100 是异常值
y_pred = [1.1, 2.1, 3.1, 4.1, 5.0]

MSE   = 1805.08  ← (100-5)²=9025 主导了整个 loss
MAE   =   19.08  ← |100-5|=95 影响线性
Huber =   47.58  ← 大误差被截断到线性
```

MSE 的 loss 几乎完全被那一个异常值控制了。

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化 MSE/MAE/Huber 的损失曲线和异常值影响。

## 答案与总结

| 要点 | 结论 |
|------|------|
| 最高效写法 | `sum(generator) / n`，单次遍历，O(1) 空间，1 次除法 |
| 效率陷阱 | 别逐元素除 n、别建中间列表、用 zip 不用索引 |
| 数值稳定 | 大数组用 Welford 或 Kahan 补偿求和 |
| 梯度 | $2(\hat{y} - y) / n$，和误差成正比 → 对异常值敏感 |
| vs MAE | MSE 平滑可导但怕异常值，MAE 鲁棒但零点不可导 |

**一句话总结**：MSE 本身是一行公式，面试考的是你对 `O(1)` 空间、单次除法、数值稳定的理解。
