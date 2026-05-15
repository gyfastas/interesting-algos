"""
MLP 多元回归 — 手写 Autograd（两层 Linear + ReLU）
=====================================================

用纯 NumPy 实现一个两层感知机：
  Input → Linear1 → ReLU → Linear2 → Output

包含:
  - 每个运算模块的前向 + 反向传播
  - 链式法则手动组装
  - 数值梯度验证
  - 四个非线性函数的拟合测试
"""

import numpy as np


# =============================================================================
# 基础运算模块（带 Autograd）
# =============================================================================

class Linear:
    """
    全连接层: y = x @ W + b

    参数:
        d_in:  输入维度
        d_out: 输出维度
    """
    def __init__(self, d_in: int, d_out: int):
        # Xavier 初始化: 方差 = 2 / (d_in + d_out)
        limit = np.sqrt(6.0 / (d_in + d_out))
        self.W = np.random.uniform(-limit, limit, (d_in, d_out))
        self.b = np.zeros((1, d_out))

        # 缓存前向输入，反向需要
        self.x = None

        # 梯度
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, d_in) → y: (N, d_out)"""
        self.x = x
        return x @ self.W + self.b          # 广播加法

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output: dL/dy, shape (N, d_out)

        根据链式法则:
          dL/dx  = grad_output @ W^T     → (N, d_out) @ (d_out, d_in) = (N, d_in)
          dL/dW  = x^T @ grad_output     → (d_in, N) @ (N, d_out) = (d_in, d_out)
          dL/db  = sum(grad_output, axis=0) → (1, d_out)
        """
        self.grad_W = self.x.T @ grad_output       # (d_in, d_out)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)  # (1, d_out)
        grad_input = grad_output @ self.W.T        # (N, d_in)
        return grad_input

    def update(self, lr: float) -> None:
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b


class ReLU:
    """ReLU: y = max(0, x)"""
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0)           # 保存正数位置
        return x * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        dy/dx = 1 if x > 0 else 0
        dL/dx = dL/dy * dy/dx = grad_output * mask
        """
        return grad_output * self.mask

    def update(self, lr: float) -> None:
        pass  # ReLU 没有可学习参数


class MSELoss:
    """MSE Loss: L = 0.5 * mean((y_hat - y)^2)"""
    def __init__(self):
        self.y_hat = None
        self.y = None

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        self.y_hat = y_hat
        self.y = y
        self.diff = y_hat - y
        return 0.5 * np.mean(self.diff ** 2)

    def backward(self) -> np.ndarray:
        """
        dL/dy_hat = (y_hat - y) / N
        """
        N = self.y_hat.shape[0]
        return self.diff / N


# =============================================================================
# MLP 模型组装
# =============================================================================

class MLP:
    """
    两层感知机: Input → Linear(d_in, hidden) → ReLU → Linear(hidden, d_out) → Output
    """
    def __init__(self, d_in: int, hidden: int, d_out: int = 1):
        self.layers = [
            Linear(d_in, hidden),
            ReLU(),
            Linear(hidden, d_out),
        ]
        self.loss_fn = MSELoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, y: np.ndarray) -> float:
        return self.loss_fn.forward(self.layers[-1].forward_cache if hasattr(self.layers[-1], 'forward_cache') else self.layers[-2].forward(self.layers[1].forward(self.layers[0].forward(self.layers[0].x))), y)

    # 修正: 重新写一个完整的 backward
    def full_backward(self, y: np.ndarray) -> None:
        """从 loss 开始，逐层反向传播。"""
        # 先 forward 一遍，确保所有层都有缓存
        # 但假设 forward 已经调用过了
        grad = self.loss_fn.backward()   # dL/dy_hat
        # 反向遍历层
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        """完整的一步训练。"""
        # Forward
        out = self.forward(x)
        # Loss
        l = self.loss_fn.forward(out, y)
        # Backward
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        # Update
        for layer in self.layers:
            layer.update(lr)
        return l


# =============================================================================
# 数值梯度验证
# =============================================================================

def numerical_gradient(model: MLP, x: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """用有限差分验证 Linear 层 W 的梯度。"""
    # 只检查第一个 Linear 层的 W[0,0]
    W = model.layers[0].W
    orig = W[0, 0]

    W[0, 0] = orig + eps
    loss_plus = model.loss_fn.forward(model.forward(x), y)

    W[0, 0] = orig - eps
    loss_minus = model.loss_fn.forward(model.forward(x), y)

    W[0, 0] = orig
    num_grad = (loss_plus - loss_minus) / (2 * eps)

    # 重新 forward 恢复缓存
    model.forward(x)
    model.loss_fn.forward(model.layers[-1].x @ model.layers[-1].W + model.layers[-1].b, y)
    # 解析梯度
    model.train_step(x, y, lr=0)  # 执行 backward 但不 update
    ana_grad = model.layers[0].grad_W[0, 0]

    return abs(num_grad - ana_grad)


# =============================================================================
# 训练与测试
# =============================================================================

def train_model(model, x, y, lr=0.01, epochs=5000, print_every=500):
    """训练并返回 loss 历史。"""
    losses = []
    for epoch in range(epochs):
        l = model.train_step(x, y, lr)
        losses.append(l)
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>5}: loss = {l:.6f}")
        if np.isnan(l) or l > 1e6:
            print(f"  ⚠ 发散于 epoch {epoch}, loss = {l}")
            break
    return losses


def test_fit(name, true_fn, x, y, d_in=1, hidden=32, lr=0.01, epochs=5000):
    """
    测试 MLP 拟合一个函数。

    参数:
        name:     测试名称
        true_fn:  真实函数（用于打印和对比）
        x:        输入数据 (N, d_in)
        y:        目标数据 (N, 1)
    """
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    print(f"数据: N={x.shape[0]}, 输入维度={d_in}, hidden={hidden}")

    model = MLP(d_in, hidden, d_out=1)

    # 梯度验证
    model.forward(x)
    diff = numerical_gradient(model, x, y)
    print(f"梯度验证: |num_grad - ana_grad| = {diff:.2e} {'✓' if diff < 1e-4 else '✗'}")

    # 训练前 loss
    model = MLP(d_in, hidden, d_out=1)  # 重新初始化
    init_loss = model.loss_fn.forward(model.forward(x), y)
    print(f"初始 loss: {init_loss:.4f}")

    # 训练
    losses = train_model(model, x, y, lr=lr, epochs=epochs)

    # 最终评估
    final_pred = model.forward(x)
    final_loss = model.loss_fn.forward(final_pred, y)
    mae = np.mean(np.abs(final_pred - y))
    r2 = 1 - np.sum((final_pred - y)**2) / np.sum((y - np.mean(y))**2)

    print(f"最终 loss: {final_loss:.6f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²:  {r2:.4f}")
    return losses, model


# =============================================================================
# 主程序
# =============================================================================

def demo():
    np.random.seed(42)

    print("=" * 70)
    print("MLP 多元回归 — 手写 Autograd（两层 Linear + ReLU）")
    print("=" * 70)

    # ---------- 测试 1: 线性函数 y = 2x + 3 ----------
    N = 200
    x1 = np.linspace(-3, 3, N).reshape(-1, 1)
    y1 = 2 * x1 + 3 + np.random.randn(N, 1) * 0.2
    losses1, model1 = test_fit("线性函数 y = 2x + 3 + noise", "y = 2x + 3", x1, y1, d_in=1, hidden=8, lr=0.05, epochs=3000)

    # ---------- 测试 2: 二次函数 y = x^2 ----------
    x2 = np.linspace(-2, 2, N).reshape(-1, 1)
    y2 = x2 ** 2 + np.random.randn(N, 1) * 0.1
    losses2, model2 = test_fit("二次函数 y = x² + noise", "y = x²", x2, y2, d_in=1, hidden=16, lr=0.02, epochs=5000)

    # ---------- 测试 3: 正弦函数 y = sin(x) ----------
    x3 = np.linspace(-np.pi, np.pi, N).reshape(-1, 1)
    y3 = np.sin(x3) + np.random.randn(N, 1) * 0.05
    losses3, model3 = test_fit("正弦函数 y = sin(x) + noise", "y = sin(x)", x3, y3, d_in=1, hidden=32, lr=0.02, epochs=8000)

    # ---------- 测试 4: 二元二次 z = x² + y² ----------
    N4 = 500
    x4_1 = np.random.uniform(-2, 2, N4)
    x4_2 = np.random.uniform(-2, 2, N4)
    x4 = np.column_stack([x4_1, x4_2])
    y4 = (x4_1 ** 2 + x4_2 ** 2).reshape(-1, 1) + np.random.randn(N4, 1) * 0.2
    losses4, model4 = test_fit("二元函数 z = x² + y² + noise", "z = x² + y²", x4, y4, d_in=2, hidden=32, lr=0.02, epochs=8000)

    # ---------- 对比实验: hidden size 的影响 ----------
    print(f"\n{'='*60}")
    print("对比实验: hidden size 对拟合能力的影响")
    print(f"{'='*60}")
    print("目标: y = sin(x), 训练 5000 步, lr=0.02")
    print(f"{'hidden':>8} | {'最终 loss':>12} | {'R²':>8}")
    print("-" * 40)
    for h in [2, 4, 8, 16, 32, 64]:
        m = MLP(1, h, 1)
        for _ in range(5000):
            m.train_step(x3, y3, lr=0.02)
        pred = m.forward(x3)
        fl = 0.5 * np.mean((pred - y3) ** 2)
        r2 = 1 - np.sum((pred - y3)**2) / np.sum((y3 - np.mean(y3))**2)
        print(f"{h:>8} | {fl:>12.6f} | {r2:>8.4f}")

    # ---------- 对比实验: 学习率的影响 ----------
    print(f"\n{'='*60}")
    print("对比实验: 学习率对收敛的影响")
    print(f"{'='*60}")
    print("目标: y = x², hidden=16, 训练 3000 步")
    print(f"{'lr':>10} | {'最终 loss':>12} | {'状态':>10}")
    print("-" * 40)
    for lr in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        m = MLP(1, 16, 1)
        loss_history = []
        for _ in range(3000):
            l = m.train_step(x2, y2, lr=lr)
            loss_history.append(l)
            if np.isnan(l) or l > 1e6:
                break
        final_l = loss_history[-1]
        status = "发散!" if np.isnan(final_l) or final_l > 100 else "收敛"
        print(f"{lr:>10.4f} | {final_l:>12.6f} | {status:>10}")


if __name__ == "__main__":
    demo()
