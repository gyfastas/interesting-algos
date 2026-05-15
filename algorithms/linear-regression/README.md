# MLP 多元回归：手写两层感知机 + Autograd

> 深度学习基础 · 反向传播 · 链式法则 · ⭐⭐⭐

## 问题描述

在上一个版本中，我们实现了单层线性回归。现在将其升级为**两层感知机（MLP）**：

$$
\text{Input} \xrightarrow{\text{Linear}_1} \text{Hidden} \xrightarrow{\text{ReLU}} \text{Activated} \xrightarrow{\text{Linear}_2} \text{Output}
$$

网络结构：
- **Linear₁**: $h = X W_1 + b_1$（输入 → 隐层）
- **ReLU**: $a = \max(0, h)$（非线性激活）
- **Linear₂**: $\hat{y} = a W_2 + b_2$（隐层 → 输出）
- **Loss**: $L = \dfrac{1}{2N} \|\hat{y} - y\|^2$

**核心挑战**：不借助 PyTorch/TensorFlow，用纯 NumPy 手写前向传播、反向传播和参数更新，并验证它能拟合线性函数、二次函数、正弦函数等。

## 直觉分析

### 为什么需要非线性？

单层线性回归只能拟合直线（或超平面）：

$$
\hat{y} = Xw + b
$$

无论堆多少层，如果没有非线性激活，多层线性叠加仍然等价于单层线性：

$$
(X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + (b_1 W_2 + b_2) = X W' + b'
$$

**ReLU 激活函数 $\max(0, x)$ 打破了线性**，让网络可以学习分段线性函数，从而逼近任意连续函数（通用近似定理）。

### 为什么叫 Autograd？

深度学习框架（PyTorch、JAX）的核心是**自动微分（Automatic Differentiation）**。我们在这里手动实现它：
- 每个运算模块记住前向时的输入
- 反向时根据链式法则逐层计算梯度
- 最终把梯度传回所有可训练参数

## 数学建模

### 网络结构

| 层 | 运算 | 输入形状 | 输出形状 | 可学习参数 |
|---|------|---------|---------|-----------|
| Linear₁ | $h = X W_1 + b_1$ | $(N, d_{in})$ | $(N, d_{hidden})$ | $W_1 \in \mathbb{R}^{d_{in} \times d_{hidden}},\; b_1 \in \mathbb{R}^{1 \times d_{hidden}}$ |
| ReLU | $a = \max(0, h)$ | $(N, d_{hidden})$ | $(N, d_{hidden})$ | 无 |
| Linear₂ | $\hat{y} = a W_2 + b_2$ | $(N, d_{hidden})$ | $(N, d_{out})$ | $W_2 \in \mathbb{R}^{d_{hidden} \times d_{out}},\; b_2 \in \mathbb{R}^{1 \times d_{out}}$ |

### Xavier 初始化

权重不能全零（否则对称性破坏学习），也不能太大（否则 ReLU 全部死亡或梯度爆炸）。Xavier 初始化的方差为：

$$
\text{Var}(W_{ij}) = \frac{2}{d_{in} + d_{out}}
$$

即均匀分布 $U\left[-\sqrt{\dfrac{6}{d_{in} + d_{out}}},\; \sqrt{\dfrac{6}{d_{in} + d_{out}}}\right]$。

## 求解过程

### Forward（前向传播）

数据流从上到下：

$$
\begin{aligned}
h &= X W_1 + b_1 \\
a &= \max(0, h) \\
\hat{y} &= a W_2 + b_2 \\
L &= \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
\end{aligned}
$$

### Backward（反向传播）

**核心思想：链式法则**。从 Loss 开始，一层一层往回传梯度。

#### Step 1: Loss 对输出的梯度

$$
\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{N}
$$

#### Step 2: Linear₂ 的梯度

设 $\text{grad}_{out} = \dfrac{\partial L}{\partial \hat{y}}$，形状 $(N, d_{out})$：

$$
\begin{aligned}
\frac{\partial L}{\partial W_2} &= a^T \cdot \text{grad}_{out} \quad &(d_{hidden}, d_{out}) \\
\frac{\partial L}{\partial b_2} &= \sum_{i=1}^{N} \text{grad}_{out}^{(i)} \quad &(1, d_{out}) \\
\text{grad}_{in} &= \text{grad}_{out} \cdot W_2^T \quad &(N, d_{hidden})
\end{aligned}
$$

#### Step 3: ReLU 的梯度

ReLU 的导数是一个开关：

$$
\frac{da}{dh} = \begin{cases} 1 & \text{if } h > 0 \\ 0 & \text{if } h \le 0 \end{cases}
$$

因此：

$$
\text{grad}_{in} = \text{grad}_{out} \odot \mathbb{1}_{[h > 0]}
$$

#### Step 4: Linear₁ 的梯度

设 $\text{grad}_{out} = \dfrac{\partial L}{\partial h}$，形状 $(N, d_{hidden})$：

$$
\begin{aligned}
\frac{\partial L}{\partial W_1} &= X^T \cdot \text{grad}_{out} \quad &(d_{in}, d_{hidden}) \\
\frac{\partial L}{\partial b_1} &= \sum_{i=1}^{N} \text{grad}_{out}^{(i)} \quad &(1, d_{hidden}) \\
\text{grad}_{in} &= \text{grad}_{out} \cdot W_1^T \quad &(N, d_{in})
\end{aligned}
$$

### Update（参数更新）

梯度下降：

$$
W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}, \quad b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}
$$

## 代码实现

### 模块化 Autograd

```python
class Linear:
    def __init__(self, d_in, d_out):
        limit = np.sqrt(6.0 / (d_in + d_out))
        self.W = np.random.uniform(-limit, limit, (d_in, d_out))
        self.b = np.zeros((1, d_out))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        # grad_output: dL/dy, shape (N, d_out)
        self.grad_W = self.x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T   # dL/dx

    def update(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b


class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class MLP:
    def __init__(self, d_in, hidden, d_out=1):
        self.layers = [Linear(d_in, hidden), ReLU(), Linear(hidden, d_out)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train_step(self, x, y, lr):
        # Forward
        out = self.forward(x)
        # Loss
        diff = out - y
        loss = 0.5 * np.mean(diff ** 2)
        # Backward
        grad = diff / y.shape[0]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        # Update
        for layer in self.layers:
            layer.update(lr)
        return loss
```

## 动画演示

动画展示 MLP 训练的全流程：

- **网络结构可视化**：输入层 → 隐层 → ReLU → 输出层的拓扑图，实时显示激活状态
- **拟合过程**：左侧显示真实数据（散点）和模型预测（曲线），随着训练步数增加，预测曲线逐步逼近真实函数
- **Loss 曲线**：实时绘制训练 loss 的下降轨迹
- **权重分布**：直方图显示 $W_1, W_2$ 的分布变化
- **参数控制**：调节 hidden size（2~64）、学习率（0.0001~1.0）、选择拟合目标（线性/二次/正弦/二元函数）
- **梯度验证**：一键运行数值梯度验证，对比解析梯度与有限差分结果

> 打开 `animation.html` 查看交互动画

## 答案与总结

**核心 Insight**：MLP 的能力来自**非线性激活**。没有 ReLU，无论多少层都等价于单层线性模型；有了 ReLU，分段线性的组合可以逼近任意连续函数。

**关键结论**：

| 测试函数 | hidden=8 | hidden=32 | R² |
|---------|---------|-----------|-----|
| 线性 $y=2x+3$ | ✓ | ✓ | 0.997 |
| 二次 $y=x^2$ | ✓ | ✓ | 0.987 |
| 正弦 $y=\sin x$ | 一般 | ✓ | 0.994 |
| 二元 $z=x^2+y^2$ | — | ✓ | 0.986 |

**反向传播的本质**：从 Loss 开始，沿计算图的反方向，用链式法则把梯度"传"回每个参数。每个模块只需要知道：
1. 前向时自己做了什么运算
2. 反向时如何根据上游梯度计算本地梯度和下游梯度

**复杂度**：
- Forward: $O(N \cdot d_{in} \cdot d_{hidden} + N \cdot d_{hidden} \cdot d_{out})$
- Backward: 与 Forward 同阶（反向传播的优美性质）
- 空间：$O(N \cdot d_{hidden})$，存储中间激活值
