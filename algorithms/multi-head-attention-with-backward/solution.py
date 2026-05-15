"""
Multi-Head Attention 手写 Forward & Backward（困难版）
=========================================================

用纯 NumPy 实现单头/多头注意力 + Softmax + CrossEntropy 的完整反向传播，
包括：
  1. Softmax 的 Jacobian 梯度推导
  2. Sequence Cross-Entropy 与 Softmax 合并后的优美简化
  3. Multi-Head Attention 的完整反向传播链
  4. 数值梯度验证 + 训练收敛测试

网络结构（简化 1-layer Transformer）:
  Input Embedding → Multi-Head Attention → Add&Norm →
  FeedForward (Linear→ReLU→Linear) → Add&Norm →
  Linear Projection → CrossEntropyLoss
"""

import numpy as np


# =============================================================================
# 1. Softmax（带稳定实现 + 完整反向传播）
# =============================================================================

def softmax(x, axis=-1):
    """
    数值稳定的 Softmax。
    x: (..., D)，沿 axis 做 softmax
    返回 y: (..., D)，每个位置上的概率分布
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(dy, y):
    """
    Softmax 的反向传播。

    数学推导：
      y_i = exp(x_i) / Z,  Z = sum(exp(x_j))
      dy_i/dx_j = y_i * (δ_ij - y_j)

    矩阵形式：
      dx = dy * diag(y) - y * (sum(dy * y, axis=-1))
      或者等价地：dx = y * (dy - sum(dy * y, axis=-1, keepdims=True))

    参数:
      dy: dL/dy, shape 与 y 相同
      y:  softmax 的输出概率
    返回:
      dx: dL/dx
    """
    # 对于每个样本/位置，计算 inner = sum_j(dy_j * y_j)
    inner = np.sum(dy * y, axis=-1, keepdims=True)  # (..., 1)
    return y * (dy - inner)


# =============================================================================
# 2. Sequence Cross-Entropy Loss（与 Softmax 合并的简化梯度）
# =============================================================================

class SequenceCrossEntropyLoss:
    """
    Sequence Cross-Entropy Loss: 对每个时间步做分类。

    Forward:
      L = -(1/N) * sum_i(log(p_i[correct_i]))
      其中 p = softmax(logits), N = batch * seq

    Backward（Softmax + CE 合并简化）：
      若令 u = logits（softmax 前的值），则
      dL/du = (p - one_hot_target) / N

      这是深度学习中最优美的梯度公式之一：
      Softmax 输出与 one-hot 目标的差，就是 logits 的梯度。

    证明概要：
      L = -log(p_t)      (t = correct class)
      p_t = exp(u_t) / Z
      dL/du_t = -1/p_t * dp_t/du_t = -1/p_t * p_t(1-p_t) = p_t - 1
      dL/du_i = -1/p_t * dp_t/du_i = -1/p_t * (-p_t * p_i) = p_i   (i≠t)
      综合：dL/du = p - one_hot(t)
    """

    def forward(self, logits, targets):
        """
        logits: (N, V)   N = batch*seq, V = vocab_size
        targets: (N,)    每个位置的正确类别索引
        """
        self.logits = logits
        self.targets = targets
        self.N = logits.shape[0]

        self.probs = softmax(logits, axis=-1)
        # 避免 log(0)
        correct_probs = self.probs[np.arange(self.N), targets]
        self.loss = -np.mean(np.log(correct_probs + 1e-10))
        return self.loss

    def backward(self):
        """
        返回 dL/d(logits) = (probs - one_hot) / N
        """
        dx = self.probs.copy()
        dx[np.arange(self.N), self.targets] -= 1.0
        dx /= self.N
        return dx


# =============================================================================
# 3. Multi-Head Attention（完整 Forward + Backward）
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Self-Attention。

    Forward:
      Q = X @ W_q + b_q
      K = X @ W_k + b_k
      V = X @ W_v + b_v
      scores = Q_h @ K_h^T / sqrt(d_k)
      attn = softmax(scores)
      out_h = attn @ V_h
      output = concat(out_h) @ W_o + b_o

    Backward（从 dOutput 回传）：
      1. dL/dW_o = concat(out_h)^T @ dOutput
         dL/db_o = sum(dOutput)
         dL/d(concat_out_h) = dOutput @ W_o^T

      2. Reshape dL/d(out_h): (B, S, D) -> (B, H, S, d_k)
         dL/dV_h = attn^T @ dL/d(out_h)
         dL/dAttn = dL/d(out_h) @ V_h^T

      3. dL/dScores = softmax_backward(dL/dAttn, attn)
         dL/dQ_h = dL/dScores @ K_h / sqrt(d_k)
         dL/dK_h = dL/dScores^T @ Q_h / sqrt(d_k)

      4. Reshape back: (B, H, S, d_k) -> (B, S, D)
         dL/dW_q = X^T @ dL/dQ
         ... (W_k, W_v 同理)
    """

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Xavier 初始化
        lim = np.sqrt(6.0 / (d_model + d_model))
        self.W_q = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_k = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_v = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_o = np.random.uniform(-lim, lim, (d_model, d_model))

        self.b_q = np.zeros((1, d_model))
        self.b_k = np.zeros((1, d_model))
        self.b_v = np.zeros((1, d_model))
        self.b_o = np.zeros((1, d_model))

        # 梯度缓存
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_b_v = np.zeros_like(self.b_v)
        self.grad_b_o = np.zeros_like(self.b_o)

    def forward(self, X):
        """
        X: (batch, seq, d_model)
        返回: (batch, seq, d_model)
        """
        self.X = X
        B, S, D = X.shape
        H, d_k = self.n_heads, self.d_k

        # Step 1: Linear projections
        self.Q = X @ self.W_q + self.b_q   # (B, S, D)
        self.K = X @ self.W_k + self.b_k
        self.V = X @ self.W_v + self.b_v

        # Step 2: Reshape for multi-head: (B, S, D) -> (B, H, S, d_k)
        self.Q_h = self.Q.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
        self.K_h = self.K.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
        self.V_h = self.V.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)

        # Step 3: Attention scores
        self.scores = self.Q_h @ self.K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (B, H, S, S)
        self.attn = softmax(self.scores, axis=-1)  # (B, H, S, S)

        # Step 4: Apply attention to values
        self.out_h = self.attn @ self.V_h  # (B, H, S, d_k)

        # Step 5: Concatenate heads and output projection
        self.concat = self.out_h.transpose(0, 2, 1, 3).reshape(B, S, D)  # (B, S, D)
        output = self.concat @ self.W_o + self.b_o

        return output

    def backward(self, dOutput):
        """
        dOutput: (B, S, D)
        返回 dX: (B, S, D)
        """
        B, S, D = dOutput.shape
        H, d_k = self.n_heads, self.d_k

        # --- Output projection backward ---
        self.grad_W_o = self.concat.transpose(1, 2, 0).reshape(D, -1) @ dOutput.reshape(-1, D)
        # 更清晰的写法:
        concat_2d = self.concat.reshape(-1, D)          # (B*S, D)
        dOut_2d = dOutput.reshape(-1, D)                # (B*S, D)
        self.grad_W_o = concat_2d.T @ dOut_2d           # (D, D)
        self.grad_b_o = np.sum(dOut_2d, axis=0, keepdims=True)

        dConcat = dOut_2d @ self.W_o.T                  # (B*S, D)
        dConcat = dConcat.reshape(B, S, D)

        # --- Reshape dConcat -> dOut_h: (B, S, D) -> (B, H, S, d_k) ---
        dOut_h = dConcat.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)

        # --- Attention backward ---
        # dL/dV_h = attn^T @ dOut_h
        dV_h = self.attn.transpose(0, 1, 3, 2) @ dOut_h  # (B, H, S, d_k)

        # dL/dAttn = dOut_h @ V_h^T
        dAttn = dOut_h @ self.V_h.transpose(0, 1, 3, 2)  # (B, H, S, S)

        # dL/dScores = softmax_backward(dAttn, attn)
        dScores = softmax_backward(dAttn, self.attn)     # (B, H, S, S)

        # dL/dQ_h = dScores @ K_h / sqrt(d_k)
        dQ_h = dScores @ self.K_h / np.sqrt(d_k)         # (B, H, S, d_k)

        # dL/dK_h = dScores^T @ Q_h / sqrt(d_k)
        dK_h = dScores.transpose(0, 1, 3, 2) @ self.Q_h / np.sqrt(d_k)  # (B, H, S, d_k)

        # --- Reshape back: (B, H, S, d_k) -> (B, S, D) ---
        dQ = dQ_h.transpose(0, 2, 1, 3).reshape(B, S, D)
        dK = dK_h.transpose(0, 2, 1, 3).reshape(B, S, D)
        dV = dV_h.transpose(0, 2, 1, 3).reshape(B, S, D)

        # --- Q, K, V projection backward ---
        X_2d = self.X.reshape(-1, D)
        dQ_2d = dQ.reshape(-1, D)
        dK_2d = dK.reshape(-1, D)
        dV_2d = dV.reshape(-1, D)

        self.grad_W_q = X_2d.T @ dQ_2d
        self.grad_W_k = X_2d.T @ dK_2d
        self.grad_W_v = X_2d.T @ dV_2d

        self.grad_b_q = np.sum(dQ_2d, axis=0, keepdims=True)
        self.grad_b_k = np.sum(dK_2d, axis=0, keepdims=True)
        self.grad_b_v = np.sum(dV_2d, axis=0, keepdims=True)

        # dL/dX from Q, K, V
        dX = dQ_2d @ self.W_q.T + dK_2d @ self.W_k.T + dV_2d @ self.W_v.T
        dX = dX.reshape(B, S, D)

        return dX

    def update(self, lr):
        self.W_q -= lr * self.grad_W_q; self.b_q -= lr * self.grad_b_q
        self.W_k -= lr * self.grad_W_k; self.b_k -= lr * self.grad_b_k
        self.W_v -= lr * self.grad_W_v; self.b_v -= lr * self.grad_b_v
        self.W_o -= lr * self.grad_W_o; self.b_o -= lr * self.grad_b_o


# =============================================================================
# 4. 简化 Transformer Layer（MHA + FFN + Residual）
# =============================================================================

class LinearLayer:
    """全连接层，带 Xavier 初始化。"""
    def __init__(self, d_in, d_out):
        lim = np.sqrt(6.0 / (d_in + d_out))
        self.W = np.random.uniform(-lim, lim, (d_in, d_out))
        self.b = np.zeros((1, d_out))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        dy_2d = dy.reshape(-1, dy.shape[-1])
        self.grad_W = x_2d.T @ dy_2d
        self.grad_b = np.sum(dy_2d, axis=0, keepdims=True)
        dx = dy_2d @ self.W.T
        return dx.reshape(self.x.shape)

    def update(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b


class ReLULayer:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask

    def update(self, lr): pass


class SimpleTransformerLayer:
    """
    简化版 Transformer Encoder Layer：
      X → MHA → Add&Norm → FFN → Add&Norm
    
    为简化教学，此处省略 LayerNorm（其 backward 较复杂，
    不影响理解 MHA + CrossEntropy 的核心梯度流）。
    """

    def __init__(self, d_model, n_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn1 = LinearLayer(d_model, d_ff)
        self.relu = ReLULayer()
        self.ffn2 = LinearLayer(d_ff, d_model)

    def forward(self, X):
        # MHA + Residual
        self.X_in = X
        mha_out = self.mha.forward(X)
        self.after_mha = X + mha_out  # 残差连接

        # FFN + Residual
        ffn1 = self.ffn1.forward(self.after_mha)
        ffn_relu = self.relu.forward(ffn1)
        ffn2 = self.ffn2.forward(ffn_relu)
        output = self.after_mha + ffn2  # 残差连接

        return output

    def backward(self, dOutput):
        # --- FFN backward ---
        dAfterMha = dOutput.copy()
        dFfn2 = dOutput.copy()
        dFfn_relu = self.ffn2.backward(dFfn2)
        dFfn1 = self.relu.backward(dFfn_relu)
        dAfterMha += self.ffn1.backward(dFfn1)

        # --- MHA backward ---
        dX = dAfterMha.copy()
        dMha = dAfterMha.copy()
        dX += self.mha.backward(dMha)

        return dX

    def update(self, lr):
        self.mha.update(lr)
        self.ffn1.update(lr)
        self.ffn2.update(lr)


# =============================================================================
# 5. 完整模型 + 训练
# =============================================================================

class TinyTransformer:
    """
    微型 Transformer 语言模型：
      Embedding → TransformerLayer → Linear Projection → CrossEntropy
    """

    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=64, max_len=16):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 词嵌入 + 位置编码
        lim = np.sqrt(6.0 / (vocab_size + d_model))
        self.token_embed = np.random.uniform(-lim, lim, (vocab_size, d_model))
        self.pos_embed = np.random.uniform(-lim, lim, (max_len, d_model))

        self.transformer = SimpleTransformerLayer(d_model, n_heads, d_ff)
        self.projection = LinearLayer(d_model, vocab_size)
        self.loss_fn = SequenceCrossEntropyLoss()

    def embed(self, tokens):
        """tokens: (batch, seq) int"""
        B, S = tokens.shape
        tok_emb = self.token_embed[tokens]            # (B, S, D)
        pos_emb = self.pos_embed[np.arange(S)]         # (S, D)
        return tok_emb + pos_emb[np.newaxis, :, :]     # (B, S, D)

    def forward(self, tokens):
        """返回 logits: (batch*seq, vocab_size)"""
        self.tokens = tokens
        self.embedded = self.embed(tokens)
        self.transformed = self.transformer.forward(self.embedded)
        B, S, D = self.transformed.shape
        logits = self.projection.forward(self.transformed)  # (B, S, V)
        return logits.reshape(B * S, self.vocab_size)

    def backward(self, logits, targets, lr):
        """完整训练一步。"""
        # Forward loss
        loss = self.loss_fn.forward(logits, targets)

        # Backward: CE -> Projection -> Transformer -> Embedding
        dLogits = self.loss_fn.backward()                   # (B*S, V)
        B, S = self.tokens.shape
        dLogits = dLogits.reshape(B, S, self.vocab_size)

        dTransformed = self.projection.backward(dLogits)    # (B, S, D)
        dEmbedded = self.transformer.backward(dTransformed) # (B, S, D)

        # Embedding 梯度（只更新被用到的 token）
        np.add.at(self.token_embed, self.tokens, -lr * dEmbedded)
        # 位置编码梯度
        dPos = np.sum(dEmbedded, axis=0)
        self.pos_embed[:S] -= lr * dPos

        # Update layers
        self.transformer.update(lr)
        self.projection.update(lr)

        return loss


# =============================================================================
# 6. 数值梯度验证
# =============================================================================

def numerical_grad_check_softmax():
    """验证 Softmax 的 backward。"""
    x = np.random.randn(3, 5)
    y = softmax(x, axis=-1)
    dy = np.random.randn(3, 5)
    dx_ana = softmax_backward(dy, y)

    # 数值梯度
    eps = 1e-5
    dx_num = np.zeros_like(x)
    for i in range(3):
        for j in range(5):
            x[i, j] += eps
            y_plus = softmax(x, axis=-1)
            loss_plus = np.sum(y_plus * dy)
            x[i, j] -= 2 * eps
            y_minus = softmax(x, axis=-1)
            loss_minus = np.sum(y_minus * dy)
            x[i, j] += eps
            dx_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

    diff = np.max(np.abs(dx_ana - dx_num))
    return diff


def numerical_grad_check_mha():
    """验证 MHA 的 backward（检查 W_q 的第一个元素）。"""
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    X = np.random.randn(2, 5, 16)

    out = mha.forward(X)
    dOut = np.random.randn(2, 5, 16)
    mha.backward(dOut)
    ana_grad = mha.grad_W_q[0, 0]

    eps = 1e-5
    mha2 = MultiHeadAttention(d_model=16, n_heads=4)
    # 复制权重
    for attr in ['W_q', 'W_k', 'W_v', 'W_o', 'b_q', 'b_k', 'b_v', 'b_o']:
        setattr(mha2, attr, getattr(mha, attr).copy())

    mha2.W_q[0, 0] += eps
    out_plus = mha2.forward(X)
    loss_plus = np.sum(out_plus * dOut)

    mha2.W_q[0, 0] -= 2 * eps
    out_minus = mha2.forward(X)
    loss_minus = np.sum(out_minus * dOut)

    num_grad = (loss_plus - loss_minus) / (2 * eps)
    return abs(ana_grad - num_grad)


# =============================================================================
# 7. 训练测试：Copy Task（复制序列）
# =============================================================================

def generate_copy_task(batch, seq_len, vocab_size):
    """
    Copy Task: 输入序列，输出相同序列。
    这是测试注意力机制最基本的任务。
    """
    tokens = np.random.randint(0, vocab_size, size=(batch, seq_len))
    return tokens, tokens.copy()


def train_copy_task(vocab_size=8, d_model=32, n_heads=4, d_ff=64,
                    seq_len=8, batch=16, epochs=2000, lr=0.05):
    """训练 Copy Task。"""
    print(f"\n{'='*70}")
    print("训练任务: Copy Task (复制输入序列)")
    print(f"{'='*70}")
    print(f"配置: vocab={vocab_size}, d_model={d_model}, heads={n_heads}, "
          f"seq_len={seq_len}, batch={batch}")

    model = TinyTransformer(vocab_size, d_model, n_heads, d_ff, max_len=seq_len)

    losses = []
    for epoch in range(epochs):
        x, y = generate_copy_task(batch, seq_len, vocab_size)
        logits = model.forward(x)
        targets = y.reshape(-1)
        loss = model.backward(logits, targets, lr)
        losses.append(loss)

        if epoch % 200 == 0 or epoch == epochs - 1:
            # 计算准确率
            preds = np.argmax(logits, axis=-1)
            acc = np.mean(preds == targets)
            print(f"  epoch {epoch:>5}: loss={loss:.6f}, acc={acc:.4f}")

    # 最终测试
    x_test, y_test = generate_copy_task(4, seq_len, vocab_size)
    logits_test = model.forward(x_test)
    preds_test = np.argmax(logits_test, axis=-1).reshape(4, seq_len)
    print(f"\n最终测试 (batch=4):")
    for i in range(4):
        match = np.array_equal(preds_test[i], y_test[i])
        print(f"  输入: {x_test[i]}  预测: {preds_test[i]}  目标: {y_test[i]}  {'✓' if match else '✗'}")

    return losses


# =============================================================================
# 8. 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("Multi-Head Attention 手写 Forward & Backward（困难版）")
    print("=" * 70)

    # 1. Softmax 梯度验证
    print("\n[1] Softmax Backward 梯度验证")
    diff = numerical_grad_check_softmax()
    print(f"  解析梯度 vs 数值梯度最大差异: {diff:.2e}")
    print(f"  {'✓ 通过' if diff < 1e-4 else '✗ 失败'}")

    # 2. CrossEntropy + Softmax 合并简化验证
    print("\n[2] CrossEntropy + Softmax 合并梯度验证")
    np.random.seed(42)
    logits = np.random.randn(10, 8)
    targets = np.random.randint(0, 8, size=10)
    loss_fn = SequenceCrossEntropyLoss()
    loss = loss_fn.forward(logits, targets)
    dx_ana = loss_fn.backward()

    eps = 1e-5
    dx_num = np.zeros_like(logits)
    for i in range(10):
        for j in range(8):
            logits[i, j] += eps
            loss_plus = loss_fn.forward(logits, targets)
            logits[i, j] -= 2 * eps
            loss_minus = loss_fn.forward(logits, targets)
            logits[i, j] += eps
            dx_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

    diff = np.max(np.abs(dx_ana - dx_num))
    print(f"  解析梯度 vs 数值梯度最大差异: {diff:.2e}")
    print(f"  {'✓ 通过' if diff < 1e-4 else '✗ 失败'}")
    print(f"  注意: dL/dlogits = (softmax(logits) - one_hot) / N，这是深度学习中最优美的公式之一")

    # 3. MHA 梯度验证
    print("\n[3] Multi-Head Attention Backward 梯度验证")
    diff = numerical_grad_check_mha()
    print(f"  W_q[0,0] 解析梯度 vs 数值梯度差异: {diff:.2e}")
    print(f"  {'✓ 通过' if diff < 1e-4 else '✗ 失败'}")

    # 4. Copy Task 训练
    losses = train_copy_task(vocab_size=8, d_model=32, n_heads=4, d_ff=64,
                             seq_len=8, batch=16, epochs=2000, lr=0.05)

    # 5. 不同配置对比
    print(f"\n{'='*70}")
    print("[5] 不同 head 数量对 Copy Task 的影响")
    print(f"{'='*70}")
    print(f"{'Heads':>8} | {'Final Loss':>12} | {'Final Acc':>10}")
    print("-" * 40)
    for heads in [1, 2, 4, 8]:
        model = TinyTransformer(vocab_size=8, d_model=32, n_heads=heads, d_ff=64, max_len=8)
        for _ in range(1000):
            x, y = generate_copy_task(16, 8, 8)
            logits = model.forward(x)
            model.backward(logits, y.reshape(-1), 0.05)
        x_test, y_test = generate_copy_task(16, 8, 8)
        logits_test = model.forward(x_test)
        preds = np.argmax(logits_test, axis=-1)
        acc = np.mean(preds == y_test.reshape(-1))
        loss_fn = SequenceCrossEntropyLoss()
        final_loss = loss_fn.forward(logits_test, y_test.reshape(-1))
        print(f"{heads:>8} | {final_loss:>12.6f} | {acc:>10.4f}")


if __name__ == "__main__":
    demo()
