# 手写 Cross Entropy Loss

## 问题描述

面试经典：**手写一个 cross entropy loss，要求数值稳定。**

几乎所有分类任务（图像分类、LLM next-token prediction）的损失函数都是 CE loss。和上一题 softmax 配套——实际上 PyTorch 的 `nn.CrossEntropyLoss` 接受的是 **raw logits**，内部一步完成 log-softmax + NLL，从不单独算 softmax。

## 直觉分析

### Cross Entropy 在做什么？

模型输出 logits（未归一化的分数），CE loss 衡量"模型的预测分布 $q$ 和真实分布 $p$ 之间的差距"：

```
logits:     [2.0,  1.0,  0.1]     ← 模型觉得第0类最可能
target:     0                      ← 真实标签就是第0类
softmax:    [0.659, 0.242, 0.099]  ← 转成概率
CE loss:    -log(0.659) = 0.417    ← 取正确类的概率，求负对数
```

**CE loss 的本质**：正确类的概率越高，loss 越小。

- 完美预测 $p_{target} = 1.0$ → $-\log(1) = 0$
- 均匀猜测 $p_{target} = 1/C$ → $-\log(1/C) = \log(C)$
- 完全错误 $p_{target} \to 0$ → $-\log(0) \to \infty$

### 为什么不能先 softmax 再 log？

两个数值问题：

**1. Softmax 上溢**：`exp(1000)` = inf（上一题的问题）

**2. Log 下溢**：softmax 的输出可能极小（如 $10^{-40}$），在 float32 下直接变成 0，`log(0)` = -inf

解决方案：**不算 softmax 本身，用 log-sum-exp 直接算 log-softmax**。

## 数学推导

### 定义

给定 logits $z \in \mathbb{R}^C$ 和正确类别 $y$：

$$\text{CE}(z, y) = -\log \text{softmax}(z)_y = -\log \frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}}$$

展开 log：

$$= -z_y + \log \sum_{j=1}^{C} e^{z_j}$$

### 数值稳定版本

用 log-sum-exp 技巧（和上一题的 -max 一脉相承）：

$$\log \sum_j e^{z_j} = M + \log \sum_j e^{z_j - M}, \quad M = \max_j z_j$$

所以：

$$\text{CE}(z, y) = -z_y + M + \log \sum_j e^{z_j - M}$$

全程：
- $e^{z_j - M} \leq 1$，不会上溢
- 直接算 log，不需要中间的 softmax 概率值，不会下溢

### 梯度

CE loss 对 logits 的梯度非常简洁：

$$\frac{\partial \text{CE}}{\partial z_i} = \text{softmax}(z)_i - \mathbb{1}[i = y] = p_i - y_i$$

即：**softmax 输出减去 one-hot 标签**。对于正确类，梯度是 $p_y - 1$（负值，推动 logit 增大）；对于其他类，梯度是 $p_i$（正值，推动 logit 减小）。

## 代码实现

### 纯 Python 数值稳定版

```python
def cross_entropy_stable(logits, target):
    m = max(logits)
    log_sum_exp = m + math.log(sum(math.exp(x - m) for x in logits))
    return -logits[target] + log_sum_exp
```

三行代码。核心就是 $-z_y + \text{LogSumExp}(z)$。

### PyTorch 版

```python
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: (B, C), targets: (B,)"""
    # log-sum-exp
    m = logits.max(dim=-1, keepdim=True).values
    lse = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))

    # 取 target 对应的 logit
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    return (-target_logits + lse).mean()
```

这就是 `F.cross_entropy(logits, targets)` 的内部实现。

### 注意：PyTorch 的 API 设计

```python
# ✅ 正确：传 raw logits，内部做 log-softmax
loss = F.cross_entropy(logits, targets)

# ❌ 错误：先 softmax 再传入（数值不稳定 + 语义不对）
loss = F.cross_entropy(F.softmax(logits, dim=-1), targets)

# ❌ 错误：nn.NLLLoss 需要的是 log-probabilities，不是 logits
loss = F.nll_loss(logits, targets)
```

`nn.CrossEntropyLoss` = `log_softmax` + `nll_loss`，一步到位。

## 扩展：Label Smoothing

### 问题

标准 CE 的目标是 one-hot 分布——让模型 100% 确信正确类。这会导致：
- 模型过度自信（logits 差距越来越大）
- 泛化能力下降

### Label Smoothing 的做法

把 one-hot 目标"软化"：

$$y_i^{smooth} = \begin{cases} 1 - \alpha + \alpha/C & \text{if } i = y \\ \alpha/C & \text{otherwise} \end{cases}$$

$\alpha = 0.1$ 时，正确类目标从 1.0 变成 0.9 + 0.1/C，其他类从 0 变成 0.1/C。

```python
def cross_entropy_label_smoothing(logits, targets, smoothing=0.1):
    C = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)      # 数值稳定的 log-softmax

    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # 正确类
    smooth_loss = -log_probs.mean(dim=-1)                         # 均匀分布的 CE

    return ((1 - smoothing) * nll + smoothing * smooth_loss).mean()
```

## 扩展：Focal Loss

### 问题

在类别极度不平衡的任务中（如目标检测，99% 是背景），大量简单样本（背景）主导梯度，模型学不好困难样本（小目标）。

### Focal Loss 的做法

给每个样本一个权重 $(1 - p_t)^\gamma$：

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

- 简单样本 $p_t \to 1$：权重 $(1-p_t)^\gamma \to 0$，loss 被压低
- 困难样本 $p_t \to 0$：权重 $(1-p_t)^\gamma \to 1$，loss 保持原样

$\gamma = 0$ 时退化为标准 CE，$\gamma = 2$ 是常用值。

## CE Loss 与 Perplexity

LLM 训练中最常看的指标是 **perplexity**：

$$\text{PPL} = e^{\text{CE}}$$

| CE Loss | PPL | 含义 |
|---------|-----|------|
| $\log(V)$ | $V$ | 随机猜（vocab 大小） |
| 4.0 | 54.6 | 很差 |
| 2.0 | 7.4 | 不错 |
| 1.5 | 4.5 | 好 |
| 1.0 | 2.7 | 非常好 |

典型 LLM vocab_size=128000，随机猜的 CE = $\ln(128000) \approx 11.76$。

## 完整计算流程

```
logits z: (B, C)          ← 模型输出，未归一化
target y: (B,)            ← 正确类别索引
     │
     ▼
M = max(z, dim=-1)        ← 数值稳定
     │
     ▼
LSE = M + log(Σ exp(z-M)) ← log-sum-exp
     │
     ▼
CE = -z[y] + LSE          ← 负目标 logit + LogSumExp
     │
     ▼
loss = mean(CE)            ← batch 平均
```

全程**不算 softmax 概率值**，直接在 log 域完成。

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化 CE loss 的计算过程和 -log(p) 曲线。

## 答案与总结

| 要点 | 结论 |
|------|------|
| CE loss 公式 | $-z_y + \log\sum e^{z_j}$，不需要显式算 softmax |
| 数值稳定 | 用 log-sum-exp 减最大值，上一题 softmax 的 -max 技巧直接复用 |
| 梯度 | $\nabla_{z} \text{CE} = \text{softmax}(z) - \text{one\_hot}(y)$，极其简洁 |
| PyTorch | `F.cross_entropy` 接受 raw logits，内部一步到位 |
| Label Smoothing | 软化目标分布，防止过度自信 |
| Focal Loss | $(1-p_t)^\gamma$ 加权，关注困难样本 |

**一句话总结**：手写 CE loss 就是 `-logits[target] + LogSumExp(logits)`——千万别先 softmax 再 log，直接在 log 域算完。
