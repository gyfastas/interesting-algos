# 手写数值稳定的 Softmax

## 问题描述

面试经典：**手写一个 softmax 函数，要求数值稳定。**

看似简单——就是 $\frac{e^{x_i}}{\sum e^{x_j}}$——但直接算 `exp(1000)` 会得到 `inf`。面试考的就是你知不知道"减最大值"这个技巧，以及为什么它数学上等价。

## 直觉分析

### Softmax 做了什么？

把一个任意实数向量变成**概率分布**（所有值非负，加和为 1）：

```
logits: [2.0,  1.0,  0.1]
           ↓ softmax
probs:  [0.659, 0.242, 0.099]    sum = 1.0
```

大的值得到大的概率，小的值得到小的概率，而且是"指数级放大差异"。

### 问题在哪？

`exp()` 增长极快：

```
exp(10)   = 22026
exp(100)  = 2.69e+43
exp(709)  = 8.22e+307    ← float64 能表示的极限
exp(710)  = OverflowError!
```

float32 更惨，`exp(89)` 就溢出了。

在实际模型中，attention logits 或分类 logits 的值域完全可能超过 100。**Naive softmax 在生产环境中一定会出问题。**

### 解法：减去最大值

在算 exp 之前，把所有元素减去向量中的最大值 $M = \max(x_i)$：

$$\text{softmax}(x_i) = \frac{e^{x_i - M}}{\sum_j e^{x_j - M}}$$

减完之后：
- 最大的元素变成 0，$e^0 = 1$
- 其他元素全是负数，$e^{负数} < 1$
- **永远不会溢出**

## 数学证明：减最大值不改变结果

$$\frac{e^{x_i - M}}{\sum_j e^{x_j - M}} = \frac{e^{x_i} \cdot e^{-M}}{\sum_j e^{x_j} \cdot e^{-M}} = \frac{e^{x_i} \cdot \cancel{e^{-M}}}{\cancel{e^{-M}} \cdot \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

$e^{-M}$ 在分子分母中同时出现，完全约掉。所以**减去任何常数 $c$ 都不改变 softmax 的结果**，选 $c = \max(x_i)$ 是因为它让所有 exp 参数 $\leq 0$，最大化数值安全性。

## 代码实现

### Naive 版本（会溢出）

```python
def softmax_naive(x):
    exps = [math.exp(xi) for xi in x]
    total = sum(exps)
    return [e / total for e in exps]
```

### 数值稳定版本

```python
def softmax_stable(x):
    m = max(x)
    exps = [math.exp(xi - m) for xi in x]
    total = sum(exps)
    return [e / total for e in exps]
```

只多了一行 `m = max(x)` 和一处 `- m`，但解决了所有溢出问题。

### PyTorch 版本

```python
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    exps = torch.exp(x - m)
    return exps / exps.sum(dim=dim, keepdim=True)
```

PyTorch 内置的 `F.softmax` 和 `torch.softmax` 内部就是这么实现的。

## 进阶：三个遍历的问题

标准的稳定 softmax 需要**三次遍历**数据：

```
Pass 1: m = max(x)                        ← 遍历一次求最大值
Pass 2: d = Σ exp(x_i - m)                ← 遍历一次求分母
Pass 3: out_i = exp(x_i - m) / d          ← 遍历一次求结果
```

对于 GPU 来说，每次遍历意味着一次 global memory 读取。Attention 的 softmax 作用在 $(S \times S)$ 的矩阵上，三次遍历就是三次读取 $O(S^2)$ 的数据。当 $S$ 很大时，这成为**显存带宽瓶颈**。

## 进阶：Online Softmax

Online Softmax 把 Pass 1 和 Pass 2 合并成一次遍历：**边扫描边维护 running max 和 running sum**。

核心思想：当遇到一个新的最大值时，用一个修正因子把之前的 sum 调整过来。

```python
def softmax_online(x):
    m = float('-inf')   # running max
    d = 0.0             # running sum of exp

    for xi in x:
        if xi > m:
            d = d * math.exp(m - xi) + 1.0   # 修正旧 sum + 加入新元素
            m = xi
        else:
            d += math.exp(xi - m)

    return [math.exp(xi - m) / d for xi in x]
```

**修正因子 `d * exp(m_old - m_new)` 的直觉**：

之前的 $d = \sum_{j < i} e^{x_j - m_{old}}$。现在 max 变了，需要把每项变成 $e^{x_j - m_{new}}$：

$$e^{x_j - m_{new}} = e^{x_j - m_{old}} \cdot e^{m_{old} - m_{new}}$$

所以整个 sum 乘以 $e^{m_{old} - m_{new}}$ 就行了。

### Online Softmax 为什么重要？

这是 **FlashAttention** 的核心技巧。FlashAttention 在计算 attention 时不能把完整的 $S \times S$ attention 矩阵放在 HBM 里，而是分块 (tiling) 计算。每个块只看到一部分 key，所以无法提前知道全局 max——Online Softmax 解决了这个问题。

```
FlashAttention 的分块计算:
  Block 1: keys[0:B]   → 局部 softmax → 输出 O₁, 局部 max m₁, 局部 sum d₁
  Block 2: keys[B:2B]  → 局部 softmax → 输出 O₂, 局部 max m₂, 局部 sum d₂
  ...
  合并时: 用 Online Softmax 的修正逻辑，把 O₁ 按新的全局 max 修正
```

## 进阶：Log-Softmax

交叉熵损失 $\text{CE} = -\log(\text{softmax}(x_i))$ 需要 log-softmax。直接算 `log(softmax(x))` 有问题：softmax 可能下溢到 0，`log(0) = -inf`。

用 log-sum-exp 技巧直接算：

$$\log \text{softmax}(x_i) = x_i - \log \sum_j e^{x_j} = x_i - M - \log \sum_j e^{x_j - M}$$

```python
def log_softmax_stable(x):
    m = max(x)
    lse = m + math.log(sum(math.exp(xi - m) for xi in x))
    return [xi - lse for xi in x]
```

全程没有算过 softmax 本身，避开了下溢。PyTorch 的 `F.log_softmax` 和 `nn.CrossEntropyLoss`（内部调 log_softmax）都用这个技巧。

## Softmax 的温度

顺带一提：

$$\text{softmax}(x_i / T)$$

- $T \to 0$：趋近 argmax（one-hot）
- $T = 1$：标准 softmax
- $T \to \infty$：趋近均匀分布

LLM 推理时调 temperature 就是这个 $T$。实现时减 max 的技巧同样适用：$\max(x_i / T) = \max(x_i) / T$。

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化溢出问题、-max 修复、以及 temperature 对分布的影响。

## 答案与总结

| 要点 | 结论 |
|------|------|
| Naive softmax 的问题 | `exp(x)` 当 x > 709 (float64) 或 x > 88 (float32) 溢出 |
| -max 技巧 | 先减最大值，保证 exp 参数 ≤ 0，数学上完全等价 |
| 为什么等价 | $e^{-M}$ 在分子分母约掉 |
| Online Softmax | 单次遍历同时维护 max 和 sum，是 FlashAttention 的核心 |
| Log-Softmax | 用 log-sum-exp 避免 softmax 下溢后取 log 的问题 |

**一句话总结**：手写 softmax 只需在 `exp` 前减去 `max(x)`——一行代码的差距，但区分了"能用"和"会崩"。
