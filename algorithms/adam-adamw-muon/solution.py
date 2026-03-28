"""
Adam / AdamW / Muon 优化器 — 更新公式的纯 Python 演示

不是完整实现，重点展示一步更新的计算过程和三者的差异。
"""

import math


# ============================================================
# 1. Adam — 一步更新
# ============================================================
def adam_step(params, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam 单步更新（逐元素）。

    params, grads, m, v: 都是等长的 list[float]
    t: 当前步数（从 1 开始）
    """
    new_params = []
    for i in range(len(params)):
        # 更新一阶矩（动量）和二阶矩（梯度平方的 EMA）
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2

        # 偏差校正
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)

        # 更新参数
        new_params.append(params[i] - lr * m_hat / (math.sqrt(v_hat) + eps))

    return new_params


# ============================================================
# 2. AdamW — 一步更新（解耦权重衰减）
# ============================================================
def adamw_step(params, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
    """
    AdamW 单步更新。
    唯一区别：权重衰减直接作用在参数上，不进入 m/v。
    """
    new_params = []
    for i in range(len(params)):
        # 一阶矩和二阶矩只看纯梯度（不含正则项！）
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2

        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)

        # 先做权重衰减（解耦！），再做 Adam 更新
        decayed = params[i] * (1 - lr * wd)
        new_params.append(decayed - lr * m_hat / (math.sqrt(v_hat) + eps))

    return new_params


# ============================================================
# 3. Muon — 一步更新（矩阵正交化）
# ============================================================
def mat_mul(A, B):
    """矩阵乘法 (m×k) @ (k×n) -> (m×n)。"""
    m, k = len(A), len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


def mat_transpose(A):
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def mat_add(A, B, alpha=1.0):
    return [[A[i][j] + alpha * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def frobenius_norm(A):
    return math.sqrt(sum(A[i][j] ** 2 for i in range(len(A)) for j in range(len(A[0]))))


def newton_schulz_orthogonalize(M, steps=5):
    """
    Newton-Schulz 迭代：将矩阵 M 正交化为最近的正交矩阵。

    这是 Muon 的核心操作。不需要 SVD，只用矩阵乘法。
    """
    # 归一化
    norm = frobenius_norm(M)
    if norm < 1e-12:
        return M
    X = mat_scale(M, 1.0 / norm)

    # 预设系数（来自 Muon 论文）
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(steps):
        Xt = mat_transpose(X)
        XtX = mat_mul(Xt, X)          # X^T X
        XtX2 = mat_mul(XtX, XtX)      # (X^T X)^2

        # X_{k+1} = aX + b·X·(X^TX) + c·X·(X^TX)^2
        term1 = mat_scale(X, a)
        term2 = mat_scale(mat_mul(X, XtX), b)
        term3 = mat_scale(mat_mul(X, XtX2), c)

        X = mat_add(mat_add(term1, term2), term3)

    return X


def muon_step(W, G, M, lr=0.02, beta=0.95):
    """
    Muon 单步更新（矩阵级别）。

    W: 权重矩阵 (m × n)
    G: 梯度矩阵 (m × n)
    M: 动量矩阵 (m × n)
    """
    m, n = len(W), len(W[0])

    # Step 1: 更新动量（和 SGD momentum 一样）
    for i in range(m):
        for j in range(n):
            M[i][j] = beta * M[i][j] + G[i][j]

    # Step 2: 正交化动量矩阵
    U = newton_schulz_orthogonalize(M)

    # Step 3: 用正交化后的方向更新参数
    W_new = [[W[i][j] - lr * U[i][j] for j in range(n)] for i in range(m)]
    return W_new


# ============================================================
# 对比演示
# ============================================================
def demo():
    print("=" * 60)
    print("Adam vs AdamW vs Muon — 单步更新演示")
    print("=" * 60)

    # --- 标量参数演示 (Adam vs AdamW) ---
    params = [2.0, -1.0, 0.5, 3.0]
    grads = [0.1, -0.5, 0.3, 0.01]
    D = len(params)

    m_adam = [0.0] * D
    v_adam = [0.0] * D
    m_adamw = [0.0] * D
    v_adamw = [0.0] * D

    print(f"\n初始参数:  {params}")
    print(f"梯度:      {grads}")

    adam_result = adam_step(params[:], grads, m_adam, v_adam, t=1)
    adamw_result = adamw_step(params[:], grads, m_adamw, v_adamw, t=1, wd=0.01)

    print(f"\nAdam  更新后: [{', '.join(f'{x:.6f}' for x in adam_result)}]")
    print(f"AdamW 更新后: [{', '.join(f'{x:.6f}' for x in adamw_result)}]")

    # 差异来自权重衰减
    diff = [abs(a - b) for a, b in zip(adam_result, adamw_result)]
    print(f"差异 (|Adam-AdamW|): [{', '.join(f'{x:.6f}' for x in diff)}]")
    print("→ 差异来自权重衰减的解耦：AdamW 对大参数衰减更多")

    # --- 矩阵参数演示 (Muon) ---
    print(f"\n{'='*60}")
    print("Muon 矩阵正交化演示 (3×3 权重矩阵)")
    print("=" * 60)

    W = [[1.0, 0.2, -0.3],
         [0.5, -1.0, 0.1],
         [-0.2, 0.3, 0.8]]
    G = [[0.1, -0.2, 0.05],
         [-0.3, 0.15, 0.1],
         [0.02, -0.1, 0.2]]
    M = [[0.0] * 3 for _ in range(3)]

    print("梯度矩阵 G:")
    for row in G:
        print(f"  [{', '.join(f'{x:7.3f}' for x in row)}]")

    # 正交化后的更新方向
    for i in range(3):
        for j in range(3):
            M[i][j] = G[i][j]  # 第一步动量 = 梯度
    U = newton_schulz_orthogonalize(M)

    print("\n正交化后 U (Newton-Schulz):")
    for row in U:
        print(f"  [{', '.join(f'{x:7.3f}' for x in row)}]")

    # 验证正交性: U^T U ≈ I
    UtU = mat_mul(mat_transpose(U), U)
    print("\nU^T·U (应接近单位矩阵):")
    for row in UtU:
        print(f"  [{', '.join(f'{x:7.3f}' for x in row)}]")

    print("\n关键区别:")
    print("  Adam:  每个参数独立缩放，g_i / √v_i")
    print("  Muon:  整个矩阵正交化，所有奇异值→1")
    print("  效果:  Muon 的更新方向各维度正交，不会互相干扰")


if __name__ == "__main__":
    demo()
