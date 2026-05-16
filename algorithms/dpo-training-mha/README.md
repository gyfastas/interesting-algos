# DPO 训练 One-Layer MHA Transformer

## 问题描述

在 [Multi-Head Attention with Backward](../multi-head-attention/) 的基础上，**用 DPO (Direct Preference Optimization) Loss 训练一个单层 MHA Transformer，并验证收敛性**。

核心考点：**DPO Loss 的梯度推导** —— 不是调包，而是手写从 logits 到 MHA 参数的完整反向传播。

---

## 直觉分析

### 什么是 DPO？

传统 RLHF  pipeline：
```
SFT → 训练 Reward Model → PPO 优化
```

DPO 的洞察：**不需要显式训练 Reward Model**，直接用偏好数据优化策略模型：

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \cdot \left(\log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)$$

其中：
- $y_w$：偏好回答（win）
- $y_l$：非偏好回答（lose）
- $\pi$：策略模型（被训练）
- $\pi_{ref}$：参考模型（冻结）
- $\beta$：温度参数，控制偏离参考模型的程度

### 为什么 DPO 的梯度推导是考点？

面试中光会写 MHA 的 forward 不够，还要能推导出 DPO Loss 对 logits 的梯度，并理解它如何通过 MHA 的 backward 回传到所有参数。

---

## 数学建模

### 1. DPO Loss 的形式

令：

$$h = \beta \cdot (\log P_w - \log P_l - \log P_{ref,w} + \log P_{ref,l})$$

$$\mathcal{L} = -\log \sigma(h) = \log(1 + e^{-h})$$

### 2. 梯度推导（核心考点）

**Step 1：对 h 的导数**

$$\frac{d\mathcal{L}}{dh} = -(1 - \sigma(h)) = -\alpha$$

其中 $\alpha = 1 - \sigma(h) = \frac{1}{1+e^h}$

**Step 2：log-softmax 的导数**

对于 $\log P = \text{log-softmax}(\text{logits})[\text{target}]$：

$$\frac{d(\log P)}{d(\text{logits})} = \mathbf{1}_{\text{target}} - \text{softmax}(\text{logits})$$

**Step 3：链式法则**

对 $y_w$ 的 logits：

$$\frac{d\mathcal{L}}{d(\text{logits}_w)} = \frac{d\mathcal{L}}{dh} \cdot \frac{dh}{d(\log P_w)} \cdot \frac{d(\log P_w)}{d(\text{logits}_w)}$$

$$= (-\alpha) \cdot \beta \cdot (\mathbf{1}_w - \text{softmax}_w)$$

$$= \alpha\beta \cdot (\text{softmax}_w - \mathbf{1}_w)$$

对 $y_l$ 的 logits：

$$\frac{d\mathcal{L}}{d(\text{logits}_l)} = \alpha\beta \cdot (\mathbf{1}_l - \text{softmax}_l)$$

$$= -\alpha\beta \cdot (\text{softmax}_l - \mathbf{1}_l)$$

### 3. 关键观察

| | $y_w$（偏好） | $y_l$（非偏好）|
|---|---|---|
| 梯度方向 | 标准 CE backward | **负的** 标准 CE backward |
| 效果 | **提升** target 概率 | **降低** target 概率 |
| 缩放系数 | $\alpha \cdot \beta$ | $\alpha \cdot \beta$ |

$\alpha = 1 - \sigma(h)$ 的物理意义：
- 模型越确信 $y_w$ 优于 $y_l$（$h$ 越大），$\alpha \to 0$，更新越小
- 模型越不确定（$h \approx 0$），$\alpha \approx 0.5$，更新最大

---

## 代码实现

### DPO Loss 核心

```python
class DPOLoss:
    def forward(self, logits_w, logits_l, target_w, target_l,
                logPref_w, logPref_l, response_mask):
        # log P = log_softmax(logits)[target]
        logPw = sum(log_softmax(logits_w)[target_w] * mask)
        logPl = sum(log_softmax(logits_l)[target_l] * mask)

        h = beta * (logPw - logPl - logPref_w + logPref_l)
        sigma_h = 1 / (1 + exp(-h))
        alpha = 1 - sigma_h
        return -log(sigma_h)

    def backward(self):
        scale = alpha * beta
        # y_w: 提升 target 概率
        dlogits_w = scale * (softmax_w - one_hot_w)
        # y_l: 降低 target 概率
        dlogits_l = scale * (one_hot_l - softmax_l)
        return dlogits_w, dlogits_l
```

### 梯度回传路径

```
DPO Loss
    │
    ├── dlogits_w ──→ proj.backward ──→ ffn.backward ──→ mha.backward ──→ dEmbed
    │                                                                          │
    ├── dlogits_l ──→ proj.backward ──→ ffn.backward ──→ mha.backward ──→ dEmbed
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
                                        合并梯度 + 更新 token_embed / pos_embed / mha / ffn / proj
```

注意：**两次 forward（$y_w$ 和 $y_l$）共享同一套参数**，梯度需要累加后统一更新。

---

## 训练与收敛

### 任务设计

固定偏好数据集：
- prompt = $[a]$
- $y_w = [a+1]$（偏好）
- $y_l = [a+3]$（非偏好）

模型需要通过 causal attention 学习：给定 prompt $a$，生成 $a+1$ 比 $a+3$ 更好。

### 训练结果

```
epoch     0: DPO_loss=0.692850, pref_acc=0.4688, alpha=0.5000
epoch   100: DPO_loss=0.464933, pref_acc=0.8594, alpha=0.5534
epoch   200: DPO_loss=0.019595, pref_acc=1.0000, alpha=0.0261
epoch   499: DPO_loss=0.001156, pref_acc=1.0000, alpha=0.0013
```

- **Loss**：从 0.69（随机，$\sigma(h)=0.5$）下降到 0.001（几乎完全确信）
- **偏好准确率**：从 ~47%（随机）提升到 **100%**
- **$\alpha$**：从 0.5 降到 0.001，说明模型越来越确信偏好判断

### 不同 β 的对比

| beta | Final Loss | Pref Acc | 说明 |
|------|-----------|----------|------|
| 0.1 | 0.545 | 68.75% | 温度太高，收敛慢 |
| 0.3 | 0.004 | 100% | **推荐值**，收敛快且稳定 |
| 0.5 | 0.002 | 100% | 温度偏低，偏离参考模型更多 |

---

## 动画演示

> 打开 `animation.html` 查看交互动画：
> - DPO Loss 曲线随训练下降
> - 偏好对准确率从随机提升到 100%
> - α（确信度）从 0.5 下降到接近 0
> - 模型在 response 位置的概率分布可视化

---

## 答案与总结

| 要点 | 结论 |
|------|------|
| DPO 核心思想 | 跳过 Reward Model，直接用偏好数据优化策略 |
| 梯度推导关键 | $d\mathcal{L}/d(\text{logits}_w) = \alpha\beta(\text{softmax}_w - \mathbf{1}_w)$ |
| $y_w$ vs $y_l$ 梯度 | 方向相反：一个提升 target 概率，一个降低 |
| $\alpha = 1-\sigma(h)$ | 自适应学习率：越确信更新越小 |
| 工程注意 | 两次 forward 共享参数，梯度需累加后统一更新 |

**一句话总结**：DPO 的梯度 = 带自适应缩放系数 $\alpha\beta$ 的对比学习 —— 把偏好回答往上拉，非偏好回答往下压，模型越确信时更新越温和。
