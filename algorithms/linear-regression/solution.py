"""
多元线性回归 — 手写 Forward & Backward

包含: 完整梯度推导 + 梯度下降训练 + 解析解对比 + 不同学习率实验
"""

import numpy as np


# ============================================================
# 1. 线性回归模型
# ============================================================
class LinearRegression:
    """
    多元线性回归: ŷ = Xw + b

    Forward:  ŷ = X @ w + b
    Loss:     L = (1/2N) * ||ŷ - y||²
    Backward: ∂L/∂w = (1/N) * X^T @ (ŷ - y)
              ∂L/∂b = (1/N) * sum(ŷ - y)
    Update:   w -= lr * ∂L/∂w
              b -= lr * ∂L/∂b
    """

    def __init__(self, d_in: int):
        # 零初始化 (线性回归用零初始化没问题，不像神经网络)
        self.w = np.zeros((d_in, 1))
        self.b = 0.0

        # 缓存，backward 需要用
        self.X = None
        self.y_hat = None
        self.residual = None

        # 梯度
        self.grad_w = None
        self.grad_b = None

    # ---------- Forward ----------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播: ŷ = Xw + b

        X: (N, D)
        w: (D, 1)
        b: scalar
        ŷ: (N, 1)
        """
        self.X = X
        self.y_hat = X @ self.w + self.b  # (N, D) @ (D, 1) + scalar = (N, 1)
        return self.y_hat

    # ---------- Loss ----------

    def loss(self, y: np.ndarray) -> float:
        """
        MSE Loss: L = (1/2N) * ||ŷ - y||²

        用 1/2 是为了求导消掉 2，简化梯度公式。
        """
        self.residual = self.y_hat - y     # (N, 1) 残差
        N = y.shape[0]
        return 0.5 * np.mean(self.residual ** 2)

    # ---------- Backward ----------

    def backward(self) -> None:
        """
        反向传播: 计算 ∂L/∂w 和 ∂L/∂b

        推导:
          L = (1/2N) * e^T @ e,  其中 e = ŷ - y = Xw + b - y

          ∂L/∂w = (1/N) * X^T @ e
            因为 ∂(e^Te)/∂w = 2 * X^T @ e，乘上 1/(2N) 的系数 → (1/N) * X^T @ e

          ∂L/∂b = (1/N) * sum(e)
            因为 b 对每个样本的贡献相同，∂e_i/∂b = 1
        """
        N = self.X.shape[0]
        self.grad_w = (1 / N) * (self.X.T @ self.residual)   # (D, N) @ (N, 1) = (D, 1)
        self.grad_b = (1 / N) * np.sum(self.residual)        # scalar

    # ---------- Update ----------

    def update(self, lr: float) -> None:
        """梯度下降: w -= lr * ∂L/∂w, b -= lr * ∂L/∂b"""
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b

    # ---------- 完整训练步 ----------

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """一步完整训练: forward → loss → backward → update"""
        self.forward(X)
        l = self.loss(y)
        self.backward()
        self.update(lr)
        return l


# ============================================================
# 2. 解析解 (Normal Equation)
# ============================================================
def normal_equation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    闭合解: w* = (X^T X)^{-1} X^T y

    增广矩阵版本 (把 b 合并到 w 里):
    X_aug = [X | 1]
    w_aug = (X_aug^T X_aug)^{-1} X_aug^T y
    """
    N = X.shape[0]
    X_aug = np.hstack([X, np.ones((N, 1))])   # (N, D+1)
    # w_aug = (X_aug^T @ X_aug)^{-1} @ X_aug^T @ y
    w_aug = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
    w = w_aug[:-1]      # (D, 1)
    b = w_aug[-1, 0]    # scalar
    return w, b


# ============================================================
# 3. 手动验证梯度 (数值梯度 vs 解析梯度)
# ============================================================
def check_gradient(model: LinearRegression, X: np.ndarray, y: np.ndarray, eps: float = 1e-5):
    """
    用有限差分验证解析梯度:
      数值梯度 ≈ (L(w + ε) - L(w - ε)) / (2ε)
    """
    model.forward(X)
    model.loss(y)
    model.backward()

    # 检查 grad_w
    numerical_grad_w = np.zeros_like(model.w)
    for i in range(model.w.shape[0]):
        old = model.w[i, 0]

        model.w[i, 0] = old + eps
        model.forward(X)
        loss_plus = model.loss(y)

        model.w[i, 0] = old - eps
        model.forward(X)
        loss_minus = model.loss(y)

        numerical_grad_w[i, 0] = (loss_plus - loss_minus) / (2 * eps)
        model.w[i, 0] = old

    # 恢复
    model.forward(X)
    model.loss(y)

    diff = np.max(np.abs(model.grad_w - numerical_grad_w))
    return diff, model.grad_w.flatten(), numerical_grad_w.flatten()


# ============================================================
# 演示
# ============================================================
def demo():
    np.random.seed(42)

    # 生成数据: y = 3*x1 + (-2)*x2 + 5 + noise
    N, D = 100, 2
    X = np.random.randn(N, D)
    true_w = np.array([[3.0], [-2.0]])
    true_b = 5.0
    y = X @ true_w + true_b + np.random.randn(N, 1) * 0.5

    print("=" * 60)
    print("多元线性回归 — Forward & Backward 演示")
    print("=" * 60)
    print(f"数据: N={N}, D={D}")
    print(f"真实参数: w={true_w.flatten()}, b={true_b}")

    # 1. 梯度验证
    print(f"\n--- 梯度验证 ---")
    model = LinearRegression(D)
    model.w = np.random.randn(D, 1) * 0.1
    diff, analytical, numerical = check_gradient(model, X, y)
    print(f"解析梯度: {analytical}")
    print(f"数值梯度: {numerical}")
    print(f"最大差异: {diff:.2e} {'✓ 正确' if diff < 1e-6 else '✗ 有误'}")

    # 2. 不同学习率对比
    print(f"\n--- 不同学习率训练对比 ---")
    lrs = [0.0001, 0.001, 0.01, 0.1, 1.0]

    for lr in lrs:
        model = LinearRegression(D)
        losses = []
        for epoch in range(500):
            l = model.train_step(X, y, lr)
            losses.append(l)
            if np.isnan(l) or l > 1e10:
                break

        final_loss = losses[-1]
        status = "发散!" if np.isnan(final_loss) or final_loss > 100 else f"loss={final_loss:.4f}"
        w_err = np.linalg.norm(model.w - true_w)
        print(f"  lr={lr:<8} → {status:>20}  |w-w*|={w_err:.4f}  "
              f"w=[{model.w[0,0]:.2f}, {model.w[1,0]:.2f}]  b={model.b:.2f}")

    # 3. 解析解
    print(f"\n--- 解析解 (Normal Equation) ---")
    w_star, b_star = normal_equation(X, y)
    print(f"w* = {w_star.flatten()}")
    print(f"b* = {b_star:.4f}")
    y_hat = X @ w_star + b_star
    loss_star = 0.5 * np.mean((y_hat - y) ** 2)
    print(f"最优 loss = {loss_star:.4f}")

    # 4. 梯度下降 vs 解析解
    print(f"\n--- 梯度下降 (lr=0.01, 1000步) vs 解析解 ---")
    model = LinearRegression(D)
    for _ in range(1000):
        model.train_step(X, y, lr=0.01)

    print(f"梯度下降: w=[{model.w[0,0]:.4f}, {model.w[1,0]:.4f}], b={model.b:.4f}, loss={model.train_step(X, y, 0):.4f}")
    print(f"解析解:   w=[{w_star[0,0]:.4f}, {w_star[1,0]:.4f}], b={b_star:.4f}, loss={loss_star:.4f}")
    print(f"差距: |w_gd - w*| = {np.linalg.norm(model.w - w_star):.6f}")


if __name__ == "__main__":
    demo()
