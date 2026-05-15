"""
MoE Transformer Layer — 手写 Forward & Backward
=================================================

Mixture of Experts (MoE) 的一层实现，包含：
  - Multi-Head Attention
  - Router (Top-K 专家选择)
  - 多个 Expert FFN
  - Sequence CrossEntropy Loss

纯 NumPy 实现，含完整反向传播。
"""

import numpy as np


# =============================================================================
# 基础模块（复用 MHA with Backward）
# =============================================================================

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(dy, y):
    inner = np.sum(dy * y, axis=-1, keepdims=True)
    return y * (dy - inner)


class SequenceCrossEntropyLoss:
    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets
        self.N = logits.shape[0]
        self.probs = softmax(logits, axis=-1)
        correct = self.probs[np.arange(self.N), targets]
        return -np.mean(np.log(correct + 1e-10))

    def backward(self):
        dx = self.probs.copy()
        dx[np.arange(self.N), self.targets] -= 1.0
        return dx / self.N


class LinearLayer:
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


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        lim = np.sqrt(6.0 / (d_model + d_model))
        self.W_q = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_k = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_v = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_o = np.random.uniform(-lim, lim, (d_model, d_model))
        self.b_q = np.zeros((1, d_model))
        self.b_k = np.zeros((1, d_model))
        self.b_v = np.zeros((1, d_model))
        self.b_o = np.zeros((1, d_model))

        for attr in ['grad_W_q', 'grad_W_k', 'grad_W_v', 'grad_W_o',
                     'grad_b_q', 'grad_b_k', 'grad_b_v', 'grad_b_o']:
            setattr(self, attr, np.zeros_like(getattr(self, attr.replace('grad_', ''))))

    def forward(self, X):
        self.X = X
        B, S, D = X.shape
        H, d_k = self.n_heads, self.d_k

        self.Q = X @ self.W_q + self.b_q
        self.K = X @ self.W_k + self.b_k
        self.V = X @ self.W_v + self.b_v

        self.Q_h = self.Q.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
        self.K_h = self.K.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
        self.V_h = self.V.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)

        self.scores = self.Q_h @ self.K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k)
        self.attn = softmax(self.scores, axis=-1)
        self.out_h = self.attn @ self.V_h
        self.concat = self.out_h.transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.concat @ self.W_o + self.b_o

    def backward(self, dOutput):
        B, S, D = dOutput.shape
        H, d_k = self.n_heads, self.d_k

        concat_2d = self.concat.reshape(-1, D)
        dOut_2d = dOutput.reshape(-1, D)
        self.grad_W_o = concat_2d.T @ dOut_2d
        self.grad_b_o = np.sum(dOut_2d, axis=0, keepdims=True)
        dConcat = dOut_2d @ self.W_o.T
        dConcat = dConcat.reshape(B, S, D)

        dOut_h = dConcat.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
        dV_h = self.attn.transpose(0, 1, 3, 2) @ dOut_h
        dAttn = dOut_h @ self.V_h.transpose(0, 1, 3, 2)
        dScores = softmax_backward(dAttn, self.attn)
        dQ_h = dScores @ self.K_h / np.sqrt(d_k)
        dK_h = dScores.transpose(0, 1, 3, 2) @ self.Q_h / np.sqrt(d_k)

        dQ = dQ_h.transpose(0, 2, 1, 3).reshape(B, S, D)
        dK = dK_h.transpose(0, 2, 1, 3).reshape(B, S, D)
        dV = dV_h.transpose(0, 2, 1, 3).reshape(B, S, D)

        X_2d = self.X.reshape(-1, D)
        self.grad_W_q = X_2d.T @ dQ.reshape(-1, D)
        self.grad_W_k = X_2d.T @ dK.reshape(-1, D)
        self.grad_W_v = X_2d.T @ dV.reshape(-1, D)
        self.grad_b_q = np.sum(dQ.reshape(-1, D), axis=0, keepdims=True)
        self.grad_b_k = np.sum(dK.reshape(-1, D), axis=0, keepdims=True)
        self.grad_b_v = np.sum(dV.reshape(-1, D), axis=0, keepdims=True)

        dX = dQ.reshape(-1, D) @ self.W_q.T + dK.reshape(-1, D) @ self.W_k.T + dV.reshape(-1, D) @ self.W_v.T
        return dX.reshape(B, S, D)

    def update(self, lr):
        self.W_q -= lr * self.grad_W_q; self.b_q -= lr * self.grad_b_q
        self.W_k -= lr * self.grad_W_k; self.b_k -= lr * self.grad_b_k
        self.W_v -= lr * self.grad_W_v; self.b_v -= lr * self.grad_b_v
        self.W_o -= lr * self.grad_W_o; self.b_o -= lr * self.grad_b_o


# =============================================================================
# MoE 核心模块
# =============================================================================

class ExpertFFN:
    """单个专家：两层 FFN。"""
    def __init__(self, d_model, d_ff):
        self.fc1 = LinearLayer(d_model, d_ff)
        self.relu = ReLULayer()
        self.fc2 = LinearLayer(d_ff, d_model)

    def forward(self, x):
        self.x_in = x
        h = self.fc1.forward(x)
        h = self.relu.forward(h)
        out = self.fc2.forward(h)
        return out

    def backward(self, dy):
        dh = self.fc2.backward(dy)
        dh = self.relu.backward(dh)
        dx = self.fc1.backward(dh)
        return dx

    def update(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)


class Router:
    """路由层：输出每个 token 对每个专家的偏好分数。"""
    def __init__(self, d_model, num_experts):
        self.linear = LinearLayer(d_model, num_experts)

    def forward(self, x):
        self.logits = self.linear.forward(x)  # (B, S, num_experts)
        self.probs = softmax(self.logits, axis=-1)
        return self.probs

    def backward(self, dy):
        dLogits = softmax_backward(dy, self.probs)
        return self.linear.backward(dLogits)

    def update(self, lr):
        self.linear.update(lr)


class MoELayer:
    """
    MoE 层：Top-K 专家混合 + Load Balancing Loss。

    Forward:
      1. Router 输出概率分布
      2. 对每个 token，选 top-k 专家（soft mask）
      3. 归一化 top-k 门控概率
      4. 每个 token 送入对应的 top-k 专家
      5. 加权求和得到输出
      6. 计算辅助负载均衡损失 L_aux
    """

    def __init__(self, d_model, num_experts=4, top_k=2, d_ff=64,
                 aux_loss_coef=0.3):
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(d_model, num_experts)
        self.experts = [ExpertFFN(d_model, d_ff) for _ in range(num_experts)]
        self.aux_loss_coef = aux_loss_coef
        # 统计专家负载
        self.load_counts = np.zeros(num_experts)
        self.aux_loss = 0.0

    def forward(self, X):
        self.X = X
        B, S, D = X.shape
        T = B * S

        # 1. Router
        self.gates = self.router.forward(X)  # (B, S, num_experts)

        # 2. Top-K mask
        self.topk_mask = np.zeros_like(self.gates)
        for b in range(B):
            for s in range(S):
                top_idx = np.argsort(self.gates[b, s])[-self.top_k:]
                self.topk_mask[b, s, top_idx] = 1.0

        # 3. 归一化 masked gates
        masked_gates = self.gates * self.topk_mask
        gate_sums = np.sum(masked_gates, axis=-1, keepdims=True)
        gate_sums = np.where(gate_sums == 0, 1.0, gate_sums)
        self.norm_gates = masked_gates / gate_sums

        # 4. 计算专家输出
        self.each_expert_out = []
        self.expert_outputs = np.zeros((B, S, D))
        for e in range(self.num_experts):
            out = self.experts[e].forward(X)
            self.each_expert_out.append(out)
            gate_e = self.norm_gates[:, :, e:e+1]
            self.expert_outputs += gate_e * out

        # 5. 统计负载
        for b in range(B):
            for s in range(S):
                top_idx = np.argsort(self.gates[b, s])[-self.top_k:]
                for idx in top_idx:
                    self.load_counts[idx] += 1

        # 6. Load Balancing Loss (Switch Transformer 风格)
        # L_aux = alpha * N * sum_e( f_e * p_e )
        # f_e = 被路由到 e 的 token 比例
        # p_e = 平均 softmax 门控概率
        fraction = self.load_counts / (T * self.top_k)
        mean_probs = np.mean(self.gates, axis=(0, 1))
        self.aux_loss = self.aux_loss_coef * self.num_experts * np.sum(fraction * mean_probs)

        return self.expert_outputs

    def backward(self, dOutput):
        B, S, D = dOutput.shape
        T = B * S

        # --- 主任务梯度 ---
        dRouter_main = np.zeros_like(self.gates)
        dX_experts = np.zeros((B, S, D))

        for e in range(self.num_experts):
            gate_e = self.norm_gates[:, :, e:e+1]
            dExpertOut = dOutput * gate_e
            dExpertIn = self.experts[e].backward(dExpertOut)
            dX_experts += dExpertIn

            # Router 梯度：dOutput * expert_out (只影响被选中的 top-k)
            dRouter_main[:, :, e] = np.sum(dOutput * self.each_expert_out[e], axis=-1)

        # 主任务梯度只通过 top-k 位置传播
        dRouter_main = dRouter_main * self.topk_mask

        # --- 辅助损失梯度 (影响所有位置) ---
        # L_aux = alpha * N * sum_e( f_e * p_e )
        # dL/d(gate_bse) = alpha * N * f_e / T
        fraction = self.load_counts / (T * self.top_k)
        dRouter_aux = np.zeros_like(self.gates)
        for e in range(self.num_experts):
            dRouter_aux[:, :, e] = self.aux_loss_coef * self.num_experts * fraction[e] / T

        dX_router = self.router.backward(dRouter_main + dRouter_aux)

        dX = dX_experts + dX_router
        return dX

    def update(self, lr):
        self.router.update(lr)
        for expert in self.experts:
            expert.update(lr)


# =============================================================================
# 完整 MoE Transformer 模型
# =============================================================================

class MoETransformer:
    """单层 MoE Transformer：Embedding → MHA → MoE → Projection → Loss"""

    def __init__(self, vocab_size, d_model=32, n_heads=4, num_experts=4,
                 top_k=2, d_ff=64, max_len=16, aux_loss_coef=0.3):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_experts = num_experts

        # Embedding
        lim = np.sqrt(6.0 / (vocab_size + d_model))
        self.token_embed = np.random.uniform(-lim, lim, (vocab_size, d_model))
        self.pos_embed = np.random.uniform(-lim, lim, (max_len, d_model))

        # 层
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.moe = MoELayer(d_model, num_experts, top_k, d_ff, aux_loss_coef)
        self.projection = LinearLayer(d_model, vocab_size)
        self.loss_fn = SequenceCrossEntropyLoss()

        # 专家负载历史
        self.expert_load_history = []

    def embed(self, tokens):
        B, S = tokens.shape
        tok_emb = self.token_embed[tokens]
        pos_emb = self.pos_embed[np.arange(S)]
        return tok_emb + pos_emb[np.newaxis, :, :]

    def forward(self, tokens):
        self.tokens = tokens
        self.embedded = self.embed(tokens)

        # MHA + Residual
        mha_out = self.mha.forward(self.embedded)
        self.after_mha = self.embedded + mha_out

        # MoE + Residual
        moe_out = self.moe.forward(self.after_mha)
        self.output = self.after_mha + moe_out

        # Projection
        B, S, D = self.output.shape
        logits = self.projection.forward(self.output)
        return logits.reshape(B * S, self.vocab_size)

    def backward(self, logits, targets, lr):
        """完整训练一步。"""
        ce_loss = self.loss_fn.forward(logits, targets)
        total_loss = ce_loss + self.moe.aux_loss

        # Backward
        dLogits = self.loss_fn.backward()
        B, S = self.tokens.shape
        dLogits = dLogits.reshape(B, S, self.vocab_size)

        dOutput = self.projection.backward(dLogits)
        dAfterMha = dOutput.copy()
        dMoe = dOutput.copy()

        dAfterMha += self.moe.backward(dMoe)
        dEmbedded = dAfterMha.copy()
        dMha = dAfterMha.copy()
        dEmbedded += self.mha.backward(dMha)

        # Embedding 梯度
        np.add.at(self.token_embed, self.tokens, -lr * dEmbedded)
        dPos = np.sum(dEmbedded, axis=0)
        self.pos_embed[:S] -= lr * dPos

        # Update layers
        self.mha.update(lr)
        self.moe.update(lr)
        self.projection.update(lr)

        # 记录专家负载
        self.expert_load_history.append(self.moe.load_counts.copy())
        self.moe.load_counts = np.zeros(self.num_experts)

        return total_loss, ce_loss, self.moe.aux_loss


# =============================================================================
# 训练与测试
# =============================================================================

def generate_copy_task(batch, seq_len, vocab_size):
    tokens = np.random.randint(0, vocab_size, size=(batch, seq_len))
    return tokens, tokens.copy()


def train_moe(vocab_size=8, d_model=32, n_heads=4, num_experts=4, top_k=2,
              seq_len=8, batch=16, epochs=2000, lr=0.05):
    print(f"\n{'='*70}")
    print("MoE Transformer 训练: Copy Task")
    print(f"{'='*70}")
    print(f"配置: vocab={vocab_size}, d_model={d_model}, heads={n_heads}")
    print(f"      experts={num_experts}, top_k={top_k}, seq_len={seq_len}")

    model = MoETransformer(vocab_size, d_model, n_heads, num_experts,
                           top_k, max_len=seq_len)

    losses = []
    for epoch in range(epochs):
        x, y = generate_copy_task(batch, seq_len, vocab_size)
        logits = model.forward(x)
        targets = y.reshape(-1)
        total_loss, ce_loss, aux_loss = model.backward(logits, targets, lr)
        losses.append(total_loss)

        if epoch % 200 == 0 or epoch == epochs - 1:
            preds = np.argmax(logits, axis=-1)
            acc = np.mean(preds == targets)
            print(f"  epoch {epoch:>5}: loss={total_loss:.6f} (ce={ce_loss:.6f}, aux={aux_loss:.6f}), acc={acc:.4f}")

    # 最终测试
    x_test, y_test = generate_copy_task(4, seq_len, vocab_size)
    logits_test = model.forward(x_test)
    preds_test = np.argmax(logits_test, axis=-1).reshape(4, seq_len)
    print(f"\n最终测试:")
    for i in range(4):
        match = np.array_equal(preds_test[i], y_test[i])
        print(f"  输入: {x_test[i]}  预测: {preds_test[i]}  {'✓' if match else '✗'}")

    # 专家负载分析
    print(f"\n专家负载分布分析:")
    if len(model.expert_load_history) > 0:
        total_load = np.sum(model.expert_load_history, axis=0)
        total = np.sum(total_load)
        for e in range(num_experts):
            pct = total_load[e] / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 5)
            print(f"  Expert {e}: {total_load[e]:>8.0f} tokens ({pct:>5.1f}%) {bar}")
        print(f"  理想分布: 每个专家 {(100/num_experts):.1f}%")
        print(f"  负载均衡度: {np.std(total_load/total*100 if total>0 else 0):.2f}% (标准差)")

    return losses, model


def demo():
    print("=" * 70)
    print("MoE Transformer Layer — 手写 Forward & Backward")
    print("=" * 70)
    print()

    # 训练 Copy Task
    losses, model = train_moe(vocab_size=8, d_model=32, n_heads=4,
                               num_experts=4, top_k=2,
                               seq_len=8, batch=16, epochs=2000, lr=0.05)

    # 不同 top_k 对比
    print(f"\n{'='*70}")
    print("对比: 不同 top_k 对训练的影响")
    print(f"{'='*70}")
    print(f"{'top_k':>6} | {'Final Loss':>12} | {'Final Acc':>10} | {'负载均衡度':>10}")
    print("-" * 50)

    for tk in [1, 2, 4]:
        m = MoETransformer(vocab_size=8, d_model=32, n_heads=4,
                           num_experts=4, top_k=tk, max_len=8)
        for _ in range(1000):
            x, y = generate_copy_task(16, 8, 8)
            logits = m.forward(x)
            m.backward(logits, y.reshape(-1), 0.05)

        x_test, y_test = generate_copy_task(16, 8, 8)
        logits_test = m.forward(x_test)
        preds = np.argmax(logits_test, axis=-1)
        acc = np.mean(preds == y_test.reshape(-1))
        loss_fn = SequenceCrossEntropyLoss()
        final_loss = loss_fn.forward(logits_test, y_test.reshape(-1))

        total_load = np.sum(m.expert_load_history, axis=0) if m.expert_load_history else np.zeros(4)
        total = np.sum(total_load)
        balance = np.std(total_load / total * 100) if total > 0 else 0

        print(f"{tk:>6} | {final_loss:>12.6f} | {acc:>10.4f} | {balance:>9.2f}%")

    # 不同专家数量对比
    print(f"\n{'='*70}")
    print("对比: 不同专家数量对训练的影响 (top_k=2)")
    print(f"{'='*70}")
    print(f"{'Experts':>8} | {'Final Loss':>12} | {'Final Acc':>10}")
    print("-" * 40)

    for ne in [2, 4, 8]:
        m = MoETransformer(vocab_size=8, d_model=32, n_heads=4,
                           num_experts=ne, top_k=min(2, ne), max_len=8)
        for _ in range(1000):
            x, y = generate_copy_task(16, 8, 8)
            logits = m.forward(x)
            m.backward(logits, y.reshape(-1), 0.05)

        x_test, y_test = generate_copy_task(16, 8, 8)
        logits_test = m.forward(x_test)
        preds = np.argmax(logits_test, axis=-1)
        acc = np.mean(preds == y_test.reshape(-1))
        loss_fn = SequenceCrossEntropyLoss()
        final_loss = loss_fn.forward(logits_test, y_test.reshape(-1))
        print(f"{ne:>8} | {final_loss:>12.6f} | {acc:>10.4f}")


if __name__ == "__main__":
    demo()
