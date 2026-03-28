# Adam、AdamW、Muon 优化器原理与对比

## 问题描述

面试/讨论常见问题：**Adam 和 AdamW 有什么区别？Muon 是什么？为什么新的优化器能比 Adam 更好？**

这三个优化器代表了深度学习优化的三个阶段：
- **Adam (2014)** — 自适应学习率的奠基之作，至今仍是最广泛使用的优化器
- **AdamW (2017)** — 修复了 Adam 中权重衰减的 bug，成为现代大模型的标配
- **Muon (2024)** — 新思路：用矩阵正交化替代逐元素自适应，在同等算力下训练更好

## 从 SGD 到 Adam：为什么需要自适应学习率

### 朴素 SGD 的问题

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

所有参数共享同一个学习率 $\eta$。问题在于：
- 稀疏特征（如 NLP 中低频词的 embedding）梯度很小且稀少，需要更大的学习率
- 高频特征梯度大且频繁，需要更小的学习率
- 不同参数的梯度量级可能差好几个数量级

**核心需求**：给每个参数一个独立的、自动调整的学习率。

### SGD with Momentum

先加入动量，用历史梯度的指数移动平均来平滑更新方向：

$$m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot m_t$$

动量解决了震荡问题，但学习率仍然是全局共享的。

## Adam：自适应学习率

### 核心思想

Adam = **Adaptive Moment Estimation**，同时维护梯度的一阶矩（均值）和二阶矩（未中心化方差）：

- **一阶矩 $m_t$**：梯度方向的指数移动平均 → 决定"往哪���"
- **二阶矩 $v_t$**：梯度平方的指数移动平均 → 决定"走多快"

### 更新公式

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

偏差校正（因为 $m_0 = v_0 = 0$，前几步估计偏小）：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

参数更新：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 直觉理解

$\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}$ 做了什么？

- 如果某个参数的梯度一直很大（$v_t$ 大），更新步长会被**缩小** → 防止震荡
- 如果某个参数的梯度一直很小（$v_t$ 小），更新步长会被**放大** → 加速收敛
- 相当于每个参数有了自己的**有效学习率** $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$

### 默认超参数

| 超参数 | 默认值 | 含义 |
|--------|--------|------|
| $\eta$ | 0.001 | 全局学习率 |
| $\beta_1$ | 0.9 | 一阶矩衰减率（动量） |
| $\beta_2$ | 0.999 | 二阶矩衰减率（自适应） |
| $\epsilon$ | 1e-8 | 数值稳定性 |

## AdamW：解耦的权重衰减

### Adam 的权重衰减有什么问题？

训练时通常会加 L2 正则化来防止过拟合：

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2$$

对 $\theta$ 求导后，梯度变成：

$$g_t^{\text{reg}} = g_t + \lambda \cdot \theta_t$$

这在 SGD 里完全等价于权重衰减：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t - \eta \lambda \cdot \theta_t = (1 - \eta\lambda) \cdot \theta_t - \eta \cdot g_t$$

**但在 Adam 里不等价！** 因为 L2 正则项的梯度 $\lambda \theta_t$ 也会进入 $m_t$ 和 $v_t$ 的计算，被���适应学习率缩放：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t(\text{含正则梯度})}{\sqrt{\hat{v}_t(\text{含正则梯度})} + \epsilon}$$

问题：
1. **权重衰减强度被自适应机制扭曲**——梯度大的参数，权重衰减反而被缩小了
2. **L2 正则的效果不可预测**——衰减强度随训练动态变化

### AdamW 的修复

Loshchilov & Hutter (2017) 的方案很简单：**把权重衰减从梯度计算中拆出来，直接作用在参数上**：

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad \text{(纯梯度，不含正则项)}$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
$$\theta_{t+1} = (1 - \eta\lambda) \cdot \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

关键区别只有一行��权重衰减 $(1 - \eta\lambda) \cdot \theta_t$ 独立于自适应学习率，**每个参数受到等比例的衰减**。

### Adam vs AdamW 一图流

```
Adam (L2 正则):
  g' = g + λθ          ← 正则项混入梯度
  m, v = EMA(g')        ← 被 Adam 的自适应机制处理
  θ -= η · m̂/√v̂        ← 权重衰减被扭曲

AdamW (解耦权重衰减):
  m, v = EMA(g)         ← 纯梯度
  θ *= (1 - ηλ)         ← 权重衰减独立作用
  θ -= η · m̂/√v̂        ← Adam 更新不受干扰
```

### 实际影响

- 在小模型上差异不大，但在**大模型 + 长训练**中，AdamW 一致优于 Adam + L2
- 所有现代大模型（GPT-3/4, Llama, DeepSeek）都用 AdamW
- AdamW 的权重衰减率 $\lambda$ 更容易调参，因为它的效果是确定性的

## Muon：矩阵正交化优化器

### Adam/AdamW 的根本局限

Adam 的自适应是**逐元素 (element-wise)** 的——每个参数 $\theta_i$ 有独立的 $v_i$，但参数之间的**相关性**完全被忽略了。

对于一个权重矩阵 $W \in \mathbb{R}^{m \times n}$：
- Adam 把它展平成 $mn$ 个独立标量来处理
- 完全丢失了矩阵的结构信息

### Muon 的核心思想

Muon (Momentum + Orthogonalization) 来自 Keller Jordan 等人 (2024)，核心观察：

> 最优的参数更新方向应该是梯度矩阵经过**正交化处理**后的结果。

具体来说，对权重矩阵 $W$ 的梯度 $G$，Muon 不直接用 $G$ 更新，而是先对 $G$ 做一次 Newton-Schulz 迭代，将其**正交化**为最近的正交矩阵。

### 更新公式

**Step 1: 动量**

和 SGD with Momentum 一样：

$$M_t = \beta \cdot M_{t-1} + G_t$$

其中 $G_t$ 是梯度矩阵，$M_t$ 是动量矩阵。注意这里**不是** Adam 那样的逐元素自适应。

**Step 2: 正交化 (Newton-Schulz 迭代)**

对动量矩阵 $M_t$ 做正交化——找到最接近 $M_t$ 的正交矩阵 $U_t$，即求解：

$$U_t = \arg\min_{U: U^TU = I} \|U - M_t\|_F$$

这等价于 $M_t$ 的极分解 (polar decomposition) 中的正交因子。实际计算用 Newton-Schulz 迭代（不需要 SVD，只用矩阵乘法，GPU 友好）：

$$X_0 = \frac{M_t}{\|M_t\|_F}$$
$$X_{k+1} = aX_k + bX_kX_k^TX_k + cX_k(X_k^TX_k)^2$$

其中 $a, b, c$ 是预设系数（通常迭代 5 次就够了）。

**Step 3: 更新参数**

$$W_{t+1} = W_t - \eta \cdot U_t$$

### 为什么正交化有效？

**1. 统一了更新尺度**

Adam 通过逐元素除以 $\sqrt{v}$ 来归一化更新幅度。Muon 通过正交化直接让更新矩阵的**所有奇异值都等于 1**——这是一种更强、更结构化的归一化。

**2. 保留了方向信息**

正交化只改变了梯度的"尺度"，保留了梯度的"方向"（最大程度保持和原始梯度的 Frobenius 内积）。

**3. 消除了参数间的冗余**

正交矩阵的列/行是正交的，这意味着不同维度的更新不会互相干扰。Adam 的逐元素处理无法做到这一点。

**4. 不需要二阶矩**

Muon 不维护 $v_t$（二阶矩），**状态量是 Adam 的一半**。对于大模型，这意味着显存节省约 $\frac{1}{3}$（Adam 需要 $m + v$ 两个 buffer，Muon 只需要 $M$ 一个）。

### Muon 的限制

Muon 只适用于**矩阵形状的参数**（$\geq$ 2D 的线性层权重）。对于 1D 参数（bias, LayerNorm/RMSNorm 的 γ, embedding），仍然需要用 Adam/AdamW。

实际使用中是**混合策略**：
- 线性层权重 → Muon
- 其他参数 → AdamW

## 三者对比

| | Adam | AdamW | Muon |
|---|---|---|---|
| **年份** | 2014 | 2017 | 2024 |
| **自适应粒度** | 逐元素 | 逐元素 | 逐矩阵（正交化） |
| **状态量 / 参数** | 2× (m + v) | 2× (m + v) | 1× (M) |
| **权重衰减** | 耦合在梯度中 | 解耦，直接作用 | 解耦 |
| **核心操作** | $m/\sqrt{v}$ | $m/\sqrt{v}$ + 解耦WD | Newton-Schulz 正交化 |
| **适用范围** | 所有参数 | 所有参数 | 仅矩阵参数（需搭配 AdamW） |
| **代表模型** | 早期 Transformer | GPT-3/4, Llama, DeepSeek | 小规模验证中 |

### 从信息利用角度理解

```
SGD:       只用当前梯度 g_t
Momentum:  用梯度的一阶矩 E[g]           → "往哪走"
Adam:      用一阶矩 + 二阶矩 E[g²]       → "往哪走" + "每个维度走多快"
Muon:      用一阶矩 + 矩阵结构信息        → "往哪走" + "维度间关系怎样"
```

Adam 回答的是"每个参数独立地该走多快"，Muon 回答的是"这个权重矩阵整体该怎么更新"。

### 一个直觉类比

想象你在调一台混音台（每个参数是一个旋钮）：

- **SGD**：把所有旋钮朝梯度方向拧同样的角度
- **Adam**：看每个旋钮历史上的调整幅度，给每个旋钮不同的灵敏度
- **Muon**：先看整个混音台的"音场"，找到一个让所有通道正交（互不干扰）的调整方案，然后整体调整

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化三种优化器在 2D 损失面上的优化轨迹对比。

## 答案与总结

| 要点 | 结论 |
|------|------|
| Adam vs SGD | 自适应学习率，每个参数有独立的有效学习率 |
| AdamW vs Adam | 权重衰减从梯度中解耦出���，效果更稳定可控 |
| Muon vs Adam | 用矩阵正交化替代逐元素自适应，更节省内存，捕获参数间结构 |
| 实践建议 | 当前主流用 AdamW，Muon 是有前景的新方向 |

**一句话总结**：Adam 给每个参数一把自己的尺子，AdamW 修好了 Adam 的正则化 bug，Muon 则说"别一个个量了，我直接看整个矩阵的结构"。
