# MHA 手写 Forward & Backward（困难版）

> 深度学习基础 · 反向传播 · 注意力机制 · ⭐⭐⭐⭐

## 问题描述

在「手写多头注意力」的基础版中，我们只实现了前向传播。现在进入**困难版**：用纯 NumPy 手写完整的 **Forward + Backward**，包括：

1. **Softmax 的梯度推导**（Jacobian 矩阵）
2. **Sequence Cross-Entropy 与 Softmax 合并后的优美简化**
3. **Multi-Head Attention 的完整反向传播链**
4. **训练收敛验证**（Copy Task）

网络结构：

$$
\text{Input} \xrightarrow{\text{Embed}} X \xrightarrow{\text{MHA}} \xrightarrow{+X} \xrightarrow{\text{FFN}} \xrightarrow{+X} \xrightarrow{\text{Projection}} \text{Logits} \xrightarrow{\text{Softmax+CE}} L
$$

## 直觉分析

### 为什么要手写 Backward？

PyTorch 的 `loss.backward()` 一键搞定，但隐藏了所有数学细节。理解这些细节是成为深度学习工程师的必经之路：

- **Softmax 不是 element-wise 的**：它的 Jacobian 是一个矩阵，每个输出对**所有输入**都有依赖
- **CE + Softmax 有神仙简化**：分开看很复杂，合起来却美得惊人——梯度就是 `softmax(x) - one_hot`
- **MHA 的梯度会"分叉"**：dL/dOutput 要同时回传给 Q、K、V 三个分支，再汇聚到 X

### Copy Task：注意力的"Hello World"

我们选择**复制序列任务**来验证模型：输入 `[3, 5, 1, 2]`，模型需要输出 `[3, 5, 1, 2]`。

这个任务强制模型学习"每个位置关注输入中对应位置的 token"——这正是注意力机制最擅长的事情。

## 数学建模

### 网络结构

| 模块 | 运算 | 参数 |
|------|------|------|
| Embedding | `X = TokenEmbed[tokens] + PosEmbed[pos]` | $W_{token} \in \mathbb{R}^{V \times D}$ |
| MHA | `Q, K, V = XW_q, XW_k, XW_v` → `softmax(QK^T/√d_k)V` → `concat @ W_o` | $W_q, W_k, W_v, W_o \in \mathbb{R}^{D \times D}$ |
| FFN | `Linear → ReLU → Linear` | $W_1 \in \mathbb{R}^{D \times d_{ff}}, W_2 \in \mathbb{R}^{d_{ff} \times D}$ |
| Projection | `Logits = X @ W_{proj}` | $W_{proj} \in \mathbb{R}^{D \times V}$ |
| Loss | `CE(softmax(logits), targets)` | 无参数 |

## 求解过程

### 1. Softmax 的梯度推导

**Forward：**

$$
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i}}{Z}
$$

**Jacobian 矩阵：**

$$
\frac{\partial y_i}{\partial x_j} = \frac{\partial}{\partial x_j}\left(\frac{e^{x_i}}{Z}\right)
$$

分两种情况：

**当 $i = j$ 时：**

$$
\frac{\partial y_i}{\partial x_i} = \frac{e^{x_i} \cdot Z - e^{x_i} \cdot e^{x_i}}{Z^2} = y_i(1 - y_i)
$$

**当 $i \neq j$ 时：**

$$
\frac{\partial y_i}{\partial x_j} = \frac{0 \cdot Z - e^{x_i} \cdot e^{x_j}}{Z^2} = -y_i y_j
$$

**统一写成矩阵形式：**

$$
\frac{\partial y}{\partial x} = \text{diag}(y) - y \cdot y^T
$$

**Backward（向量形式）：**

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = y \odot \left(\frac{\partial L}{\partial y} - \sum_j \frac{\partial L}{\partial y_j} \cdot y_j \right)
$$

> **代码实现**：`dx = y * (dy - sum(dy * y, axis=-1, keepdims=True))`

### 2. CrossEntropy + Softmax 的合并简化

**分开看时：**
- Softmax: `p = softmax(logits)`
- CE Loss: `L = -log(p_t)`，其中 $t$ 是正确类别

**合并后的梯度（深度学习最优美的公式之一）：**

$$
\boxed{\frac{\partial L}{\partial \text{logits}} = \frac{p - \text{one\_hot}(t)}{N}}
$$

**证明：**

对正确类别 $t$：

$$
\frac{\partial L}{\partial x_t} = -\frac{1}{p_t} \cdot \frac{\partial p_t}{\partial x_t} = -\frac{1}{p_t} \cdot p_t(1 - p_t) = p_t - 1
$$

对其他类别 $i \neq t$：

$$
\frac{\partial L}{\partial x_i} = -\frac{1}{p_t} \cdot \frac{\partial p_t}{\partial x_i} = -\frac{1}{p_t} \cdot (-p_t p_i) = p_i
$$

综合：$\dfrac{\partial L}{\partial x} = p - \text{one\_hot}(t)$，再除以 batch 大小 $N$ 取平均。

### 3. Multi-Head Attention 的反向传播

**Forward 回顾：**

$$
\begin{aligned}
Q &= XW_q, \quad K = XW_k, \quad V = XW_v \\
\text{scores} &= \frac{Q_h K_h^T}{\sqrt{d_k}} \\
\text{attn} &= \text{softmax}(\text{scores}) \\
\text{out} &= \text{concat}(\text{attn} \cdot V_h) W_o
\end{aligned}
$$

**Backward 链（从 dOutput 回传）：**

```
dOutput
  ↓
[Output Projection]  →  dW_o, db_o, d(concat)
  ↓
[Reshape]  →  d(out_h)
  ↓
[Attention]  →  dV_h = attn^T @ d(out_h)
             →  dAttn = d(out_h) @ V_h^T
             →  dScores = softmax_backward(dAttn, attn)
             →  dQ_h = dScores @ K_h / √d_k
             →  dK_h = dScores^T @ Q_h / √d_k
  ↓
[Reshape back]  →  dQ, dK, dV
  ↓
[Q,K,V Projections]  →  dW_q, dW_k, dW_v
                     →  dX = dQ·W_q^T + dK·W_k^T + dV·W_v^T
```

> **关键难点**：dX 有三路来源（Q、K、V），必须全部累加。

### 4. 残差连接的梯度

残差连接 `output = X + f(X)` 的 backward 非常简单：

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \text{output}} + \frac{\partial L}{\partial \text{output}} \cdot \frac{\partial f}{\partial X}
$$

即：**梯度直接相加**。

## 代码实现

### Softmax + Backward

```python
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_backward(dy, y):
    inner = np.sum(dy * y, axis=-1, keepdims=True)
    return y * (dy - inner)
```

### CrossEntropy（合并简化版）

```python
class SequenceCrossEntropyLoss:
    def forward(self, logits, targets):
        self.probs = softmax(logits, axis=-1)
        correct = self.probs[np.arange(N), targets]
        return -np.mean(np.log(correct + 1e-10))

    def backward(self):
        dx = self.probs.copy()
        dx[np.arange(N), targets] -= 1.0
        return dx / N   # ← 深度学习最优美的公式
```

### MHA Backward 核心片段

```python
def backward(self, dOutput):
    # Output projection
    dConcat = dOutput @ W_o.T
    dW_o = concat.T @ dOutput

    # Attention
    dV_h = attn.T @ dOut_h
    dAttn = dOut_h @ V_h.T
    dScores = softmax_backward(dAttn, attn)

    # Q, K
    dQ_h = dScores @ K_h / sqrt(d_k)
    dK_h = dScores.T @ Q_h / sqrt(d_k)

    # Projections back to X
    dX = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T
    return dX
```

## 动画演示

动画展示 MHA 训练的全流程：

- **Attention 热力图**：实时显示 attention 矩阵的分布，观察模型如何学会"对角线关注"（Copy Task 的特征）
- **Loss 曲线**：训练 loss 的下降轨迹
- **梯度验证面板**：一键运行 Softmax、CE、MHA 三个模块的数值梯度验证，用进度条展示解析梯度与数值梯度的匹配程度
- **预测对比**：输入序列 vs 模型预测的实时对比，训练初期全是错的，后期逐渐对齐
- **参数控制**：调节 head 数量（1~8）、学习率、序列长度

> 打开 `animation.html` 查看交互动画

## 答案与总结

**核心 Insight**：反向传播的精髓是**链式法则的模块化实现**。每个模块只需要知道：
1. 前向时自己做了什么
2. 如何根据上游梯度 `dL/dy` 计算本地梯度 `dL/dW` 和下游梯度 `dL/dx`

**三个最优美的梯度公式：**

| 模块 | 公式 | 优雅之处 |
|------|------|---------|
| **Softmax** | $dx = y \odot (dy - \sum(dy \cdot y))$ | Jacobian 的秩-1 更新形式 |
| **CE + Softmax** | $d\text{logits} = (p - \text{one\_hot}) / N$ | 两个复杂运算合并后极简 |
| **MHA Output Proj** | $dW_o = \text{concat}^T \cdot d\text{out}$ | 标准线性层 backward |

**收敛验证**：
- Copy Task（seq=8, vocab=8）在 200 个 epoch 内达到 **100% 准确率**
- Softmax backward 数值验证误差 $< 10^{-10}$
- CE+Softmax 合并梯度验证误差 $< 10^{-9}$
- MHA backward 数值验证误差 $< 10^{-9}$

**复杂度**：
- Forward: $O(B \cdot S^2 \cdot D)$，主要来自 attention scores 的矩阵乘法
- Backward: 与 Forward 同阶（反向传播的优美性质）
- 空间：$O(B \cdot S^2 \cdot H)$，存储 attention 矩阵
