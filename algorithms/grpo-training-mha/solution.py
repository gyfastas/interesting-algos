"""
GRPO (Group Relative Policy Optimization) 训练 One-Layer MHA Transformer
==========================================================================

核心考点：
  1. Group Sampling — 从旧策略采样一组回答
  2. Group-Relative Advantage — 用组内均值/标准差做 baseline
  3. Importance Sampling Ratio + PPO Clip
  4. KL 散度约束
  5. 以上所有模块的梯度回传（穿过 MHA）

模型：Embedding → MHA(causal) → Residual → FFN → Residual → Projection
任务：给定 prompt [a, b]，生成 response [a+1, b+1]（mod vocab_size）
奖励：每个正确位置 +1（max=2）

纯 NumPy 实现，含完整反向传播 + 数值梯度验证。
"""

import numpy as np
from copy import deepcopy


# =============================================================================
# 基础工具函数与层
# =============================================================================

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(dy, y):
    inner = np.sum(dy * y, axis=-1, keepdims=True)
    return y * (dy - inner)


def gradient_clip(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


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
    """标准多头注意力，含完整反向传播 + Causal Mask。"""

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

        mask = np.tril(np.ones((S, S), dtype=bool))
        self.scores = np.where(mask[None, None, :, :], self.scores, -1e9)
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

        mask = np.tril(np.ones((S, S), dtype=bool))
        dScores = np.where(mask[None, None, :, :], dScores, 0)

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

        dX = (dQ.reshape(-1, D) @ self.W_q.T +
              dK.reshape(-1, D) @ self.W_k.T +
              dV.reshape(-1, D) @ self.W_v.T)
        return dX.reshape(B, S, D)

    def update(self, lr):
        self.W_q -= lr * self.grad_W_q; self.b_q -= lr * self.grad_b_q
        self.W_k -= lr * self.grad_W_k; self.b_k -= lr * self.grad_b_k
        self.W_v -= lr * self.grad_W_v; self.b_v -= lr * self.grad_b_v
        self.W_o -= lr * self.grad_W_o; self.b_o -= lr * self.grad_b_o


class FFN:
    def __init__(self, d_model, d_ff):
        self.fc1 = LinearLayer(d_model, d_ff)
        self.relu = ReLULayer()
        self.fc2 = LinearLayer(d_ff, d_model)

    def forward(self, x):
        h = self.fc1.forward(x)
        h = self.relu.forward(h)
        return self.fc2.forward(h)

    def backward(self, dy):
        dh = self.fc2.backward(dy)
        dh = self.relu.backward(dh)
        return self.fc1.backward(dh)

    def update(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)


class OneLayerTransformer:
    """单层 Transformer：Embedding → MHA → FFN → Projection"""

    def __init__(self, vocab_size, d_model=16, n_heads=4, d_ff=32, max_len=16):
        self.vocab_size = vocab_size
        self.d_model = d_model

        lim = np.sqrt(6.0 / (vocab_size + d_model))
        self.token_embed = np.random.uniform(-lim, lim, (vocab_size, d_model))
        self.pos_embed = np.random.uniform(-lim, lim, (max_len, d_model))

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)
        self.proj = LinearLayer(d_model, vocab_size)

    def embed(self, tokens):
        B, S = tokens.shape
        tok_emb = self.token_embed[tokens]
        pos_emb = self.pos_embed[np.arange(S)]
        return tok_emb + pos_emb[np.newaxis, :, :]

    def forward(self, tokens):
        self.tokens = tokens
        self.embedded = self.embed(tokens)
        mha_out = self.mha.forward(self.embedded)
        self.after_mha = self.embedded + mha_out
        ffn_out = self.ffn.forward(self.after_mha)
        self.output = self.after_mha + ffn_out
        logits = self.proj.forward(self.output)
        return logits

    def backward(self, dLogits):
        dOutput = self.proj.backward(dLogits)
        dAfterMha = dOutput.copy()
        dFfn = dOutput.copy()
        dAfterMha += self.ffn.backward(dFfn)
        dEmbedded = dAfterMha.copy()
        dMha = dAfterMha.copy()
        dEmbedded += self.mha.backward(dMha)
        return dEmbedded

    def clear_gradients(self):
        for attr in ['grad_W_q', 'grad_W_k', 'grad_W_v', 'grad_W_o',
                     'grad_b_q', 'grad_b_k', 'grad_b_v', 'grad_b_o']:
            setattr(self.mha, attr, np.zeros_like(getattr(self.mha, attr)))
        self.ffn.fc1.grad_W.fill(0); self.ffn.fc1.grad_b.fill(0)
        self.ffn.fc2.grad_W.fill(0); self.ffn.fc2.grad_b.fill(0)
        self.proj.grad_W.fill(0); self.proj.grad_b.fill(0)

    def update(self, lr, dEmbed, tokens):
        if dEmbed.ndim == 2:
            self.token_embed -= lr * dEmbed
        else:
            np.add.at(self.token_embed, tokens, -lr * dEmbed)
        self.mha.update(lr)
        self.ffn.update(lr)
        self.proj.update(lr)

    def get_log_probs(self, tokens, targets, mask):
        """计算每个位置 target token 的 log P。"""
        logits = self.forward(tokens)
        B, S, V = logits.shape
        flat = logits.reshape(-1, V)
        flat_t = targets.reshape(-1)
        flat_m = mask.reshape(-1)
        m = np.max(flat, axis=-1, keepdims=True)
        logsm = flat - m - np.log(np.sum(np.exp(flat - m), axis=-1, keepdims=True))
        return logsm[np.arange(B*S), flat_t] * flat_m


# =============================================================================
# GRPO Trainer — 核心考点
# =============================================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization。

    训练流程（每步）：
      1. 采样：从 π_old 对当前 prompt 采样 G 个回答
      2. 奖励：计算每个回答的奖励 r_i
      3. 优势：A_i = (r_i - mean(r)) / (std(r) + ε)
      4. 损失：对每个回答的每个 token t
           ratio_t = π(y_t) / π_old(y_t)
           loss_pg = -min(ratio_t * A_i, clip(ratio_t) * A_i)
           loss_kl = β * (log π(y_t) - log π_ref(y_t))
      5. 梯度回传：穿过 MHA 更新所有参数
    """

    def __init__(self, policy, ref_model, group_size=8, beta=0.01,
                 epsilon=0.2, lr=0.01, grad_clip=1.0, kl_mode='sample'):
        """
        kl_mode: 'sample' | 'forward' | 'reverse'
          - sample: 单点估计 β*(log π(y) - log π_ref(y))，π_ref 不影响梯度方向
          - forward: Forward KL(π||π_ref) = Σ π(v)[log π(v) - log π_ref(v)]
          - reverse: Reverse KL(π_ref||π) = Σ π_ref(v)[log π_ref(v) - log π(v)]
        """
        self.policy = policy
        self.ref_model = ref_model
        self.old_policy = deepcopy(policy)
        self.group_size = group_size
        self.beta = beta
        self.epsilon = epsilon
        self.lr = lr
        self.grad_clip = grad_clip
        self.kl_mode = kl_mode

    def sample_group(self, prompt, resp_len, temperature=1.0, seed=None):
        """
        从 old_policy 自回归采样 G 个回答。
        prompt: (1, P)
        返回 list of (1, resp_len) arrays
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        group = []
        for _ in range(self.group_size):
            response = []
            for t in range(resp_len):
                if response:
                    seq = np.concatenate([prompt, np.array([response])], axis=1)
                else:
                    seq = prompt.copy()
                logits = self.old_policy.forward(seq)
                probs = softmax(logits[0, -1, :] / temperature)
                next_token = rng.choice(self.policy.vocab_size, p=probs)
                response.append(next_token)
            group.append(np.array([response]))
        return group

    def compute_rewards(self, prompt, group_responses):
        """
        奖励函数：response[i] == (prompt[i] + 1) % V 则 +1。
        prompt: (1, P), responses: list of (1, P)
        返回 rewards array (G,)
        """
        P = prompt.shape[1]
        rewards = []
        expected = (prompt[0] + 1) % self.policy.vocab_size
        for resp in group_responses:
            r = np.sum(resp[0] == expected)
            rewards.append(float(r))
        return np.array(rewards)

    def compute_advantages(self, rewards):
        """Group-relative advantage。"""
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-6
        return (rewards - mean_r) / std_r

    def grpo_loss_and_gradients(self, prompt, group_responses, advantages):
        """
        计算 GRPO loss 并累积梯度到 policy。
        返回平均 loss。
        """
        P = prompt.shape[1]
        total_loss = 0.0
        count = 0

        # 梯度累加器
        dW_proj = np.zeros_like(self.policy.proj.W)
        db_proj = np.zeros_like(self.policy.proj.b)
        dW_emb = np.zeros_like(self.policy.token_embed)
        mha_grads = {name: np.zeros_like(getattr(self.policy.mha, f'grad_{name}'))
                     for name in ['W_q', 'W_k', 'W_v', 'W_o',
                                  'b_q', 'b_k', 'b_v', 'b_o']}
        ffn1_W = np.zeros_like(self.policy.ffn.fc1.W)
        ffn1_b = np.zeros_like(self.policy.ffn.fc1.b)
        ffn2_W = np.zeros_like(self.policy.ffn.fc2.W)
        ffn2_b = np.zeros_like(self.policy.ffn.fc2.b)

        for g, (resp, adv) in enumerate(zip(group_responses, advantages)):
            seq = np.concatenate([prompt, resp], axis=1)
            seq_len = seq.shape[1]
            mask = np.zeros((1, seq_len), dtype=bool)
            mask[:, P:] = True

            # 三个模型分别 forward
            logits_pi = self.policy.forward(seq)
            logits_old = self.old_policy.forward(seq)
            logits_ref = self.ref_model.forward(seq)

            B, S, V = logits_pi.shape
            flat_pi = logits_pi.reshape(-1, V)
            flat_old = logits_old.reshape(-1, V)
            flat_ref = logits_ref.reshape(-1, V)
            flat_t = seq.reshape(-1)
            flat_m = mask.reshape(-1)

            # log softmax for all three
            def logsm(z):
                m = np.max(z, axis=-1, keepdims=True)
                return z - m - np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))

            logsm_pi = logsm(flat_pi)
            logsm_old = logsm(flat_old)
            logsm_ref = logsm(flat_ref)

            # 每个 response token 的 log P
            log_pi = logsm_pi[np.arange(B*S), flat_t] * flat_m
            log_pi_old = logsm_old[np.arange(B*S), flat_t] * flat_m
            log_pi_ref = logsm_ref[np.arange(B*S), flat_t] * flat_m

            # 只对 response 部分计算
            resp_positions = np.where(flat_m)[0]

            for pos in resp_positions:
                lp = log_pi[pos]
                lpo = log_pi_old[pos]
                lpr = log_pi_ref[pos]

                ratio = np.exp(lp - lpo)
                ratio_clip = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)

                # PPO-clip objective
                obj1 = ratio * adv
                obj2 = ratio_clip * adv
                loss_pg = -min(obj1, obj2)

                # KL penalty — 三种实现方式
                sm = softmax(flat_pi[pos:pos+1], axis=-1)[0]
                oh = np.zeros_like(sm)
                oh[flat_t[pos]] = 1.0

                if self.kl_mode == 'sample':
                    # 单点估计: β*(log π(y) - log π_ref(y))
                    loss_kl = self.beta * (lp - lpr)
                    # 梯度: β*(one_hot - softmax)
                    # π_ref 不参与梯度（被视为常数）
                    grad_kl = self.beta * (oh - sm)

                elif self.kl_mode == 'forward':
                    # Forward KL: KL(π||π_ref) = Σ_k π(k)[log π(k) - log π_ref(k)]
                    logsm_pi_v = logsm_pi[pos]
                    logsm_ref_v = logsm_ref[pos]
                    kl = np.sum(sm * (logsm_pi_v - logsm_ref_v))
                    loss_kl = self.beta * kl
                    # 梯度: β * π(j) * [log π(j) - log π_ref(j) - KL]
                    grad_kl = self.beta * sm * (logsm_pi_v - logsm_ref_v - kl)

                elif self.kl_mode == 'reverse':
                    # Reverse KL: KL(π_ref||π) = Σ_k π_ref(k)[log π_ref(k) - log π(k)]
                    logsm_pi_v = logsm_pi[pos]
                    logsm_ref_v = logsm_ref[pos]
                    sm_ref = softmax(flat_ref[pos:pos+1], axis=-1)[0]
                    kl = np.sum(sm_ref * (logsm_ref_v - logsm_pi_v))
                    loss_kl = self.beta * kl
                    # 梯度: β * (π(j) - π_ref(j))
                    grad_kl = self.beta * (sm - sm_ref)

                else:
                    raise ValueError(f"Unknown kl_mode: {self.kl_mode}")

                loss_token = loss_pg + loss_kl
                total_loss += loss_token
                count += 1

                # ========== 梯度推导（核心考点）==========
                #
                # log P = log_softmax(logits)[target]
                # d(log P)/d(logits) = one_hot - softmax
                #
                # ratio = exp(log_pi - log_pi_old)
                # dratio/d(logits) = ratio * (one_hot - softmax)
                #
                # PPO clip 判断：
                #   if adv > 0 and ratio > 1+ε: clip 生效，梯度=0
                #   if adv < 0 and ratio < 1-ε: clip 生效，梯度=0
                #   else: grad_pg = -adv * ratio * (one_hot - softmax)
                #
                # KL 模式决定 grad_kl：
                #   sample : β*(one_hot - softmax)      — π_ref 不进入梯度
                #   forward: β*π*(log π - log π_ref - KL) — 完整 Forward KL
                #   reverse: β*(π - π_ref)               — 完整 Reverse KL
                # ==========================================

                clip_active = (adv > 0 and ratio > 1 + self.epsilon) or \
                              (adv < 0 and ratio < 1 - self.epsilon)

                if clip_active:
                    grad_pg = np.zeros_like(sm)
                else:
                    grad_pg = -adv * ratio * (oh - sm)

                grad_logits = grad_pg + grad_kl

                # 构建 (1, S, V) 的 dLogits，只有当前位置非零
                dLogits = np.zeros_like(logits_pi)
                dLogits[0, pos, :] = grad_logits

                # Backward
                self.policy.clear_gradients()
                dEmbed = self.policy.backward(dLogits)

                for name in mha_grads:
                    mha_grads[name] += getattr(self.policy.mha, f'grad_{name}')
                ffn1_W += self.policy.ffn.fc1.grad_W
                ffn1_b += self.policy.ffn.fc1.grad_b
                ffn2_W += self.policy.ffn.fc2.grad_W
                ffn2_b += self.policy.ffn.fc2.grad_b
                dW_proj += self.policy.proj.grad_W
                db_proj += self.policy.proj.grad_b
                np.add.at(dW_emb, self.policy.tokens, dEmbed)

        # 平均并裁剪
        if count == 0:
            return 0.0

        n = count
        for name in mha_grads:
            setattr(self.policy.mha, f'grad_{name}',
                    gradient_clip(mha_grads[name] / n, self.grad_clip))
        self.policy.ffn.fc1.grad_W = gradient_clip(ffn1_W / n, self.grad_clip)
        self.policy.ffn.fc1.grad_b = gradient_clip(ffn1_b / n, self.grad_clip)
        self.policy.ffn.fc2.grad_W = gradient_clip(ffn2_W / n, self.grad_clip)
        self.policy.ffn.fc2.grad_b = gradient_clip(ffn2_b / n, self.grad_clip)
        self.policy.proj.grad_W = gradient_clip(dW_proj / n, self.grad_clip)
        self.policy.proj.grad_b = gradient_clip(db_proj / n, self.grad_clip)
        dEmbed = gradient_clip(dW_emb / n, self.grad_clip)

        self.policy.update(self.lr, dEmbed, self.policy.tokens)

        return total_loss / n

    def update_old_policy(self):
        """更新 old_policy 为当前 policy 的深拷贝。"""
        self.old_policy = deepcopy(self.policy)


# =============================================================================
# 训练与测试
# =============================================================================

def make_prompt_dataset(n, vocab_size, prompt_len=2, seed=123):
    """生成固定 prompt 数据集。"""
    rng = np.random.RandomState(seed)
    return rng.randint(0, vocab_size, size=(n, prompt_len))


def evaluate_accuracy(policy, prompts, resp_len=1):
    """评估模型在 prompt 数据集上的准确率。"""
    correct = 0
    for i in range(len(prompts)):
        p = prompts[i:i+1]
        seq = p.copy()
        all_correct = True
        for t in range(resp_len):
            target = (p[0, 0] + t + 1) % policy.vocab_size
            logits = policy.forward(seq)
            pred = np.argmax(logits[0, -1])
            if pred != target:
                all_correct = False
                break
            seq = np.concatenate([seq, np.array([[target]])], axis=1)
        if all_correct:
            correct += 1
    return correct / len(prompts)


def train_sft(policy, prompts, resp_len=1, lr=0.005, epochs=300, grad_clip=1.0):
    """SFT 预训练。"""
    for epoch in range(epochs):
        mha_grads = {name: np.zeros_like(getattr(policy.mha, f'grad_{name}'))
                     for name in ['W_q', 'W_k', 'W_v', 'W_o',
                                  'b_q', 'b_k', 'b_v', 'b_o']}
        ffn1_W = np.zeros_like(policy.ffn.fc1.W)
        ffn1_b = np.zeros_like(policy.ffn.fc1.b)
        ffn2_W = np.zeros_like(policy.ffn.fc2.W)
        ffn2_b = np.zeros_like(policy.ffn.fc2.b)
        dW_proj = np.zeros_like(policy.proj.W)
        db_proj = np.zeros_like(policy.proj.b)
        dW_emb = np.zeros_like(policy.token_embed)

        for i in range(len(prompts)):
            p = prompts[i:i+1]
            seq = p.copy()
            dLogits = np.zeros((1, 1 + resp_len, policy.vocab_size))
            for t in range(resp_len):
                target = (p[0, 0] + t + 1) % policy.vocab_size
                seq = np.concatenate([seq, np.array([[target]])], axis=1)
            logits = policy.forward(seq)
            sm = softmax(logits, axis=-1)
            for t in range(resp_len):
                target = (p[0, 0] + t + 1) % policy.vocab_size
                pos = 1 + t
                dLogits[0, pos, :] = sm[0, pos, :].copy()
                dLogits[0, pos, target] -= 1.0

            policy.clear_gradients()
            dEmbed = policy.backward(dLogits)
            for name in mha_grads:
                mha_grads[name] += getattr(policy.mha, f'grad_{name}')
            ffn1_W += policy.ffn.fc1.grad_W
            ffn1_b += policy.ffn.fc1.grad_b
            ffn2_W += policy.ffn.fc2.grad_W
            ffn2_b += policy.ffn.fc2.grad_b
            dW_proj += policy.proj.grad_W
            db_proj += policy.proj.grad_b
            np.add.at(dW_emb, policy.tokens, dEmbed)

        n = len(prompts)
        for name in mha_grads:
            setattr(policy.mha, f'grad_{name}', gradient_clip(mha_grads[name] / n, grad_clip))
        policy.ffn.fc1.grad_W = gradient_clip(ffn1_W / n, grad_clip)
        policy.ffn.fc1.grad_b = gradient_clip(ffn1_b / n, grad_clip)
        policy.ffn.fc2.grad_W = gradient_clip(ffn2_W / n, grad_clip)
        policy.ffn.fc2.grad_b = gradient_clip(ffn2_b / n, grad_clip)
        policy.proj.grad_W = gradient_clip(dW_proj / n, grad_clip)
        policy.proj.grad_b = gradient_clip(db_proj / n, grad_clip)
        dEmbed_avg = gradient_clip(dW_emb / n, grad_clip)
        policy.update(lr, dEmbed_avg, policy.tokens)

        if epoch % 100 == 0:
            acc = evaluate_accuracy(policy, prompts, resp_len)
            print(f"  [SFT] epoch {epoch:>3}: acc={acc:.2f}")


def train_grpo(vocab_size=8, d_model=64, n_heads=4, d_ff=128,
               prompt_len=1, resp_len=1,
               group_size=32, beta=0.01, epsilon=0.2,
               lr=0.002, epochs=600, grad_clip=1.0, weight_decay=0.0001,
               kl_mode='sample'):

    print("=" * 70)
    print("GRPO 训练 One-Layer MHA Transformer")
    print("=" * 70)
    print(f"配置: vocab={vocab_size}, d_model={d_model}, heads={n_heads}")
    print(f"      prompt_len={prompt_len}, resp_len={resp_len}, group_size={group_size}")
    print(f"      beta={beta}, epsilon={epsilon}, lr={lr}, epochs={epochs}")
    print(f"      kl_mode={kl_mode}")

    policy = OneLayerTransformer(vocab_size, d_model, n_heads, d_ff,
                                  max_len=prompt_len + resp_len)
    ref_model = deepcopy(policy)
    trainer = GRPOTrainer(policy, ref_model, group_size, beta, epsilon, lr, grad_clip, kl_mode)

    prompts = make_prompt_dataset(8, vocab_size, prompt_len)

    losses = []
    mean_rewards = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_reward = 0.0
        count = 0

        for i in range(len(prompts)):
            prompt = prompts[i:i+1]
            group = trainer.sample_group(prompt, resp_len)
            rewards = trainer.compute_rewards(prompt, group)
            advantages = trainer.compute_advantages(rewards)
            loss = trainer.grpo_loss_and_gradients(prompt, group, advantages)
            epoch_loss += loss
            epoch_reward += np.mean(rewards)
            count += 1

        losses.append(epoch_loss / count)
        mean_rewards.append(epoch_reward / count)

        # Weight decay
        if weight_decay > 0:
            for attr in ['W_q', 'W_k', 'W_v', 'W_o']:
                w = getattr(policy.mha, attr)
                setattr(policy.mha, attr, w * (1 - weight_decay))
            policy.ffn.fc1.W *= (1 - weight_decay)
            policy.ffn.fc2.W *= (1 - weight_decay)
            policy.proj.W *= (1 - weight_decay)

        if epoch % 10 == 0:
            trainer.update_old_policy()

        if epoch % 100 == 0 or epoch == epochs - 1:
            acc = evaluate_accuracy(policy, prompts, resp_len)
            best_acc = max(best_acc, acc)
            print(f"  epoch {epoch:>3}: loss={losses[-1]:.6f}, "
                  f"mean_reward={mean_rewards[-1]:.2f}, acc={acc:.2f}")

    final_acc = evaluate_accuracy(policy, prompts, resp_len)
    print("\n" + "=" * 70)
    print("训练结果总结")
    print("=" * 70)
    print(f"最终平均 Loss:      {losses[-1]:.6f}")
    print(f"最终平均奖励:       {mean_rewards[-1]:.2f} / {resp_len}")
    print(f"最终准确率:         {final_acc:.2f}")
    print(f"最高准确率:         {best_acc:.2f}")

    return losses, mean_rewards, policy


# =============================================================================
# 数值梯度验证 — 核心考点
# =============================================================================

def verify_grpo_gradients():
    """
    验证 GRPO policy gradient 和三种 KL penalty 的梯度推导。
    通过固定采样和固定 old/ref logits，只让 policy logits 变化。
    """
    print("=" * 70)
    print("GRPO 梯度数值验证 — 三种 KL 模式")
    print("=" * 70)

    np.random.seed(42)
    V = 8
    policy = OneLayerTransformer(V, d_model=8, n_heads=2, d_ff=16, max_len=4)

    # 固定输入
    seq = np.array([[1, 2, 3, 4]])
    mask = np.array([[False, False, True, True]])

    # 固定 old 和 ref 的 logits（模拟）
    logits_old = np.random.randn(1, 4, V)
    logits_ref = np.random.randn(1, 4, V)

    adv = 1.5
    beta = 0.1
    epsilon = 0.2

    # 只验证 response 第一个位置 (pos=2)
    pos = 2
    target = seq[0, pos]

    # Precompute ref softmax for forward/reverse KL
    mr = np.max(logits_ref[0, pos])
    logsm_ref_vec = logits_ref[0, pos] - mr - np.log(np.sum(np.exp(logits_ref[0, pos] - mr)))
    sm_ref_vec = softmax(logits_ref[0, pos:pos+1], axis=-1)[0]

    # Precompute old log P
    mo = np.max(logits_old[0, pos])
    logsm_old_vec = logits_old[0, pos] - mo - np.log(np.sum(np.exp(logits_old[0, pos] - mo)))
    lpo = logsm_old_vec[target]

    def make_loss_fn(kl_mode):
        """返回给定 kl_mode 的 loss 函数。"""
        def loss_fn(logits_pi_pos):
            m = np.max(logits_pi_pos)
            logsm_pi_vec = logits_pi_pos - m - np.log(np.sum(np.exp(logits_pi_pos - m)))
            lp = logsm_pi_vec[target]
            sm_pi_vec = np.exp(logsm_pi_vec)

            ratio = np.exp(lp - lpo)
            ratio_clip = np.clip(ratio, 1 - epsilon, 1 + epsilon)
            loss_pg = -min(ratio * adv, ratio_clip * adv)

            if kl_mode == 'sample':
                lpr = logsm_ref_vec[target]
                loss_kl = beta * (lp - lpr)
            elif kl_mode == 'forward':
                kl = np.sum(sm_pi_vec * (logsm_pi_vec - logsm_ref_vec))
                loss_kl = beta * kl
            elif kl_mode == 'reverse':
                kl = np.sum(sm_ref_vec * (logsm_ref_vec - logsm_pi_vec))
                loss_kl = beta * kl
            else:
                raise ValueError(kl_mode)
            return loss_pg + loss_kl
        return loss_fn

    def analytical_grad(kl_mode):
        """计算解析梯度。"""
        logits_pi = policy.forward(seq)
        sm = softmax(logits_pi[0, pos:pos+1], axis=-1)[0]
        oh = np.zeros_like(sm)
        oh[target] = 1.0

        m = np.max(logits_pi[0, pos])
        logsm_pi_vec = logits_pi[0, pos] - m - np.log(np.sum(np.exp(logits_pi[0, pos] - m)))
        lp = logsm_pi_vec[target]
        ratio = np.exp(lp - lpo)

        clip_active = (adv > 0 and ratio > 1 + epsilon) or (adv < 0 and ratio < 1 - epsilon)
        if clip_active:
            grad_pg = np.zeros_like(sm)
        else:
            grad_pg = -adv * ratio * (oh - sm)

        if kl_mode == 'sample':
            grad_kl = beta * (oh - sm)
        elif kl_mode == 'forward':
            kl = np.sum(sm * (logsm_pi_vec - logsm_ref_vec))
            grad_kl = beta * sm * (logsm_pi_vec - logsm_ref_vec - kl)
        elif kl_mode == 'reverse':
            grad_kl = beta * (sm - sm_ref_vec)
        else:
            raise ValueError(kl_mode)

        return grad_pg + grad_kl

    # 数值梯度
    eps = 1e-5
    for kl_mode in ['sample', 'forward', 'reverse']:
        loss_fn = make_loss_fn(kl_mode)
        grad_ana = analytical_grad(kl_mode)

        logits_pi = policy.forward(seq)
        grad_num = np.zeros(V)
        for i in range(V):
            lp_plus = logits_pi[0, pos].copy()
            lp_plus[i] += eps
            grad_num[i] = (loss_fn(lp_plus) - loss_fn(logits_pi[0, pos])) / eps

        rel_err = np.linalg.norm(grad_num - grad_ana) / (np.linalg.norm(grad_num) + 1e-10)
        status = "✓ 通过" if rel_err < 1e-4 else "✗ 错误！"
        print(f"  KL={kl_mode:7s}: rel_err={rel_err:.2e}  {status}")

    print()
    return True


# =============================================================================
# Demo
# =============================================================================

def demo():
    # 1. 梯度验证
    verify_grpo_gradients()

    # 2. 三种 KL 模式对比
    print("=" * 70)
    print("三种 KL 模式训练对比")
    print("=" * 70)

    for kl_mode in ['sample', 'forward', 'reverse']:
        print(f"\n--- KL mode: {kl_mode} ---")
        train_grpo(vocab_size=8, d_model=64, n_heads=4, d_ff=128,
                   prompt_len=1, resp_len=1,
                   group_size=32, beta=0.01, epsilon=0.2,
                   lr=0.002, epochs=300, grad_clip=1.0, weight_decay=0.0001,
                   kl_mode=kl_mode)


if __name__ == "__main__":
    demo()
