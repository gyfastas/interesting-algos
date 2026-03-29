# 多元线性回归：手写 Forward & Backward

## 问题描述

**手写多元线性回归的前向传播和反向传播，用梯度下降训练。展示不同学习率对拟合效果的影响。**

线性回归是机器学习最基础的模型，也是理解神经网络反向传播的起点。所有深度学习的训练本质上都是：forward → loss → backward → update。

## 直觉分析

### 模型

$$\hat{y} = Xw + b$$

- $X$：输入矩阵 $(N, D)$，N 个样本，每个 D 维特征
- $w$：权重向量 $(D, 1)$
- $b$：偏置标量
- $\hat{y}$：预测值 $(N, 1)$

### 训练流程

```
重复:
  1. Forward:  ŷ = Xw + b
  2. Loss:     L = (1/2N) Σ(ŷ_i - y_i)²     ← MSE
  3. Backward: ∂L/∂w = ?, ∂L/∂b = ?           ← 求梯度
  4. Update:   w = w - lr * ∂L/∂w              ← 梯度下降
```

## 数学推导

### Forward

$$\hat{y} = Xw + b$$

矩阵形式：$(N,1) = (N,D) \cdot (D,1) + (1)$，广播加法。

### Loss (MSE)

$$L = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 = \frac{1}{2N} \|\hat{y} - y\|^2$$

用 $\frac{1}{2}$ 是为了求导时消掉指数 2，简化公式。

### Backward

设 $e = \hat{y} - y$（残差向量，$(N,1)$）

**对 $w$ 求梯度：**

$$\frac{\partial L}{\partial w} = \frac{1}{N} X^T e = \frac{1}{N} X^T (\hat{y} - y)$$

推导：$L = \frac{1}{2N} e^T e$，$e = Xw + b - y$

$$\frac{\partial L}{\partial w} = \frac{1}{N} \frac{\partial e^T e}{\partial w} \cdot \frac{1}{2} \cdot 2 = \frac{1}{N} X^T e$$

**对 $b$ 求梯度：**

$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} e_i = \frac{1}{N} \mathbf{1}^T e$$

直觉：$b$ 对每个样本的残差贡献相同，所以梯度就是残差的平均值。

### Update (梯度下降)

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

$\eta$ 就是学习率 (learning rate)。

## 核心代码

```python
class LinearRegression:
    def __init__(self, d_in):
        self.w = np.zeros((d_in, 1))
        self.b = 0.0

    def forward(self, X):
        self.X = X
        self.y_hat = X @ self.w + self.b
        return self.y_hat

    def loss(self, y):
        self.residual = self.y_hat - y
        return 0.5 * np.mean(self.residual ** 2)

    def backward(self):
        N = self.X.shape[0]
        self.grad_w = (1/N) * self.X.T @ self.residual
        self.grad_b = (1/N) * np.sum(self.residual)

    def update(self, lr):
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b
```

## 学习率的影响

| 学习率 | 效果 |
|--------|------|
| **太小** (0.0001) | 收敛很慢，需要几千步 |
| **合适** (0.01) | 稳定收敛，几十到几百步 |
| **太大** (1.0) | 震荡，loss 来回跳 |
| **过大** (10.0) | 发散，loss 爆炸到 NaN |

### 为什么会震荡/发散？

梯度下降的更新 $w \leftarrow w - \eta \nabla L$ 本质是在 loss 曲面上走一步。

- $\eta$ 太大 → 一步走太远 → 跳过了最低点 → 到另一侧更高的地方 → 来回震荡
- ���果 $\eta > \frac{2}{\lambda_{\max}}$（$\lambda_{\max}$ 是 Hessian 最大特征值），梯度下降会发散

### 解析解 (Normal Equation)

线性回归有闭合解，不需要迭代：

$$w^* = (X^T X)^{-1} X^T y$$

但梯度下降更通用（适用于所有可微模型），且大数据时更高效（不需要矩阵求逆）。

## 动画演示

> 打开 `animation.html` 查看交互动画，对比不同学习率的拟合过程和 loss 曲线。

## 答案与总结

| 要点 | 结论 |
|------|------|
| Forward | $\hat{y} = Xw + b$ |
| Loss | $L = \frac{1}{2N} \|\hat{y} - y\|^2$ |
| Backward | $\nabla_w L = \frac{1}{N} X^T(\hat{y}-y)$，$\nabla_b L = \frac{1}{N} \sum(\hat{y}-y)$ |
| Update | $w \leftarrow w - \eta \nabla_w L$ |
| 学习率 | 太小慢收敛，太大震荡/发散，合适最快 |
| 解析解 | $w^* = (X^TX)^{-1}X^Ty$，但梯度下降更通用 |

**一句话总结**：线性回归的 backward 就是 $X^T \cdot \text{残差}$——理解了这个，神经网络的反向传播就是它的链式法则推广。
