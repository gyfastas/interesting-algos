"""
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)
训练 One-Layer MHA Transformer
==============================================================

本文件是 [GRPO 训练 MHA Transformer](../grpo-training-mha/) 的衍生版本，
在保留 GRPO 反向传播框架（穿过 MHA）的基础上，
加入 DAPO 论文 (ByteDance, 2025) 的 4 个核心改进：

  1. Clip-Higher  (解耦的 clip 范围: ε_low < ε_high)
  2. Dynamic Sampling (跳过 zero-advantage 的 prompt)
  3. Token-Level Loss (跨 sample 按 token 等权归一化)
  4. Overlong Reward Shaping (超长回答的 soft penalty)

任务：给定 prompt [a, b]，让模型生成 target = [a+1, b+1, EOS]
      （EOS = token 0, vocab = 9）
      回答必须正好 3 个 token — 太短/太长都会被惩罚
      → 这是演示 Overlong Reward Shaping 的关键设定

模型/工具函数与 GRPO 题保持完全一致（OneLayerTransformer + MHA + FFN），
方便对比训练曲线与梯度推导。

纯 NumPy 实现，含完整反向传播 + 数值梯度验证。
"""

import numpy as np
from copy import deepcopy


# =============================================================================
# 基础工具函数与层（与 GRPO 题保持一致）
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


# =============================================================================
# DAPO Trainer — 核心
# =============================================================================

class DAPOTrainer:
    """
    DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)

    在 GRPO 之上加入 4 项关键改进：
      ① Clip-Higher          — ε_low/ε_high 解耦
      ② Dynamic Sampling     — 跳过 zero-advantage prompts
      ③ Token-Level Loss     — 跨 sample 按 token 等权归一化
      ④ Overlong Reward      — 长度 > max+buffer 施加 soft penalty

    任务设定（变长 response）：
      - vocab = 9，token 0 是 EOS
      - prompt = [a, b]    (a, b ∈ {1, ..., 8})
      - target = [a+1, b+1, 0]  (长度 3)
      - max_resp_len = 3，overlong_buffer = 1
    """

    def __init__(self, policy, ref_model,
                 group_size=8,
                 # ① Clip-Higher
                 epsilon_low=0.2, epsilon_high=0.28,
                 # ② Dynamic Sampling
                 dynamic_sampling=True, max_resample=4,
                 # ④ Overlong Reward Shaping
                 max_resp_len=3, overlong_buffer=1,
                 overlong_soft_penalty=-0.5, overlong_hard_penalty=-1.0,
                 truncation_penalty=-0.5,
                 # 共用 GRPO 项
                 beta=0.01, lr=0.01, grad_clip=1.0, kl_mode='sample'):
        self.policy = policy
        self.ref_model = ref_model
        self.old_policy = deepcopy(policy)
        self.group_size = group_size

        # === 4 个 DAPO 改进的配置 ===
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.dynamic_sampling = dynamic_sampling
        self.max_resample = max_resample
        self.max_resp_len = max_resp_len
        self.overlong_buffer = overlong_buffer
        self.overlong_soft_penalty = overlong_soft_penalty
        self.overlong_hard_penalty = overlong_hard_penalty
        self.truncation_penalty = truncation_penalty

        self.EOS = 0  # 任务设定

        self.beta = beta
        self.lr = lr
        self.grad_clip = grad_clip
        self.kl_mode = kl_mode

    # ------------------------------------------------------------------------
    # 变长自回归采样（与 GRPO 关键区别 #1：支持 EOS 提前终止）
    # ------------------------------------------------------------------------

    def sample_group(self, prompt, temperature=1.0, seed=None):
        """
        从 old_policy 自回归采样 G 个回答，遇到 EOS 提前停止。
        超过 max_resp_len + overlong_buffer 强制截断（视为 overlong hard）。
        prompt: (1, P)
        返回 list of dict: { 'tokens': (1, L), 'length': int, 'truncated': bool, 'overlong': bool }
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        hard_cap = self.max_resp_len + self.overlong_buffer  # hard cap
        group = []
        for _ in range(self.group_size):
            response = []
            for t in range(hard_cap):
                if response:
                    seq = np.concatenate([prompt, np.array([response])], axis=1)
                else:
                    seq = prompt.copy()
                logits = self.old_policy.forward(seq)
                probs = softmax(logits[0, -1, :] / temperature)
                next_token = int(rng.choice(self.policy.vocab_size, p=probs))
                response.append(next_token)
                if next_token == self.EOS:
                    break
            length = len(response)
            overlong = (length > self.max_resp_len)
            truncated = (length < self.max_resp_len)
            group.append({
                'tokens': np.array([response]),
                'length': length,
                'truncated': truncated,
                'overlong': overlong,
            })
        return group

    # ------------------------------------------------------------------------
    # ② Dynamic Sampling — 跳过 zero-advantage 的 prompt
    # ------------------------------------------------------------------------

    def sample_group_with_dynamic(self, prompt, temperature=1.0, seed=None,
                                   verbose=False):
        """
        动态采样：丢掉 advantage 全 0（即 std(rewards)=0）的 group。
        限制 max_resample 次重采。
        返回 (group, advantages, n_resample, n_zero_skipped)
        """
        attempts = 0
        n_zero_skipped = 0
        while True:
            group = self.sample_group(prompt, temperature, seed)
            rewards = self.compute_rewards(prompt, group)
            if not self.dynamic_sampling:
                break
            std = float(np.std(rewards))
            if std > 1e-6:
                break
            n_zero_skipped += 1
            attempts += 1
            if verbose:
                print(f"      [dynamic] resample {attempts}: all-zero group, std={std:.4f}")
            if attempts >= self.max_resample:
                # 实在采不到非平凡 group，跳过这个 prompt
                break
        advantages = self.compute_advantages(rewards)
        return group, advantages, attempts, n_zero_skipped

    # ------------------------------------------------------------------------
    # 奖励函数（带 Overlong Reward Shaping）
    # ------------------------------------------------------------------------

    def compute_rewards(self, prompt, group):
        """
        奖励函数（含 overlong shaping）：
          - length < max_resp_len  → truncation_penalty（未在预期长度结束）
          - length > max_resp_len + overlong_buffer  → overlong_hard_penalty
          - length in (max_resp_len, max_resp_len + buffer] → overlong_soft_penalty
          - length == max_resp_len  → 正确位置数（0, 1, ..., max_resp_len）

        这里的 max_resp_len = 3，buffer = 1，soft 区只在 length=4 时命中。
        """
        a, b = int(prompt[0, 0]), int(prompt[0, 1])
        target = [(a % 8) + 1, (b % 8) + 1, self.EOS]

        rewards = []
        for item in group:
            L = item['length']
            tokens = item['tokens'][0]

            # 长度规则
            if L < self.max_resp_len:
                # 截断惩罚（未到目标长度就停）
                r = self.truncation_penalty
            elif L > self.max_resp_len + self.overlong_buffer:
                # hard overlong（理论上采样阶段已截断，这里兜底）
                r = self.overlong_hard_penalty
            elif L > self.max_resp_len:
                # soft overlong（在 buffer 区间内）
                r = self.overlong_soft_penalty
            else:
                # 长度 == max_resp_len，按位置正确性给奖励
                r = sum(1 for t in range(L) if int(tokens[t]) == target[t])
            rewards.append(float(r))
        return np.array(rewards)

    def compute_advantages(self, rewards):
        """Group-relative advantage（与 GRPO 一致）。"""
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-6
        return (rewards - mean_r) / std_r

    # ------------------------------------------------------------------------
    # DAPO Loss + 梯度（核心考点）
    # ------------------------------------------------------------------------

    def dapo_loss_and_gradients(self, prompt, group, advantages):
        """
        计算 DAPO loss 并累积梯度到 policy。

        与 GRPO 的 3 个核心区别：
          ① clip 范围 = (1-ε_low, 1+ε_high) 而非 (1-ε, 1+ε)
          ② length 可变 — 每个 response 算自己的 loss
          ③ 跨 sample 归一化 = 总 token 数（不是 G 也不是 sample-level 平均）

        返回 (avg_loss, stats) — stats 包含 clip 触发率等可观测指标。
        """
        P = prompt.shape[1]
        total_loss = 0.0
        total_token_count = 0
        clip_activate_count = 0  # 用于统计 clip 触发率

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

        for g, (item, adv) in enumerate(zip(group, advantages)):
            resp = item['tokens']  # (1, L)
            L = item['length']
            if L == 0:
                continue

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

            def logsm(z):
                m = np.max(z, axis=-1, keepdims=True)
                return z - m - np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))

            logsm_pi = logsm(flat_pi)
            logsm_old = logsm(flat_old)
            logsm_ref = logsm(flat_ref)

            # 每个 response token 的 log P（只对 response 部分非零）
            log_pi = logsm_pi[np.arange(B*S), flat_t] * flat_m
            log_pi_old = logsm_old[np.arange(B*S), flat_t] * flat_m
            log_pi_ref = logsm_ref[np.arange(B*S), flat_t] * flat_m

            resp_positions = np.where(flat_m)[0]

            for pos in resp_positions:
                lp = log_pi[pos]
                lpo = log_pi_old[pos]
                lpr = log_pi_ref[pos]

                ratio = np.exp(lp - lpo)

                # ========== ① Clip-Higher：解耦的 clip 范围 ==========
                # GRPO: clip(ρ, 1-ε, 1+ε)
                # DAPO: clip(ρ, 1-ε_low, 1+ε_high)
                ratio_clip = np.clip(ratio, 1.0 - self.epsilon_low,
                                     1.0 + self.epsilon_high)

                obj1 = ratio * adv
                obj2 = ratio_clip * adv
                loss_pg = -min(obj1, obj2)

                # KL penalty（沿用 GRPO 四种实现之一）
                sm = softmax(flat_pi[pos:pos+1], axis=-1)[0]
                oh = np.zeros_like(sm)
                oh[flat_t[pos]] = 1.0

                if self.kl_mode == 'sample':
                    loss_kl = self.beta * (lp - lpr)
                    grad_kl = self.beta * (oh - sm)
                elif self.kl_mode == 'forward':
                    logsm_pi_v = logsm_pi[pos]
                    logsm_ref_v = logsm_ref[pos]
                    kl = np.sum(sm * (logsm_pi_v - logsm_ref_v))
                    loss_kl = self.beta * kl
                    grad_kl = self.beta * sm * (logsm_pi_v - logsm_ref_v - kl)
                elif self.kl_mode == 'reverse':
                    logsm_pi_v = logsm_pi[pos]
                    logsm_ref_v = logsm_ref[pos]
                    sm_ref = softmax(flat_ref[pos:pos+1], axis=-1)[0]
                    kl = np.sum(sm_ref * (logsm_ref_v - logsm_pi_v))
                    loss_kl = self.beta * kl
                    grad_kl = self.beta * (sm - sm_ref)
                elif self.kl_mode == 'k3':
                    ratio_ref = np.exp(lpr - lp)
                    k3 = (lp - lpr) - 1 + ratio_ref
                    loss_kl = self.beta * k3
                    grad_kl = self.beta * (1 - ratio_ref) * (oh - sm)
                else:
                    raise ValueError(f"Unknown kl_mode: {self.kl_mode}")

                loss_token = loss_pg + loss_kl
                total_loss += loss_token
                total_token_count += 1

                # ========== 梯度推导（与 GRPO 相同，仅 clip 范围不同）==========
                # log P = log_softmax(logits)[target]
                # d(log P)/d(logits) = one_hot - softmax
                #
                # ratio = exp(log_pi - log_pi_old)
                # dratio/d(logits) = ratio * (one_hot - softmax)
                #
                # PPO clip 判断（解耦版）：
                #   if adv > 0 and ratio > 1+ε_high: clip 生效，梯度=0
                #   if adv < 0 and ratio < 1-ε_low: clip 生效，梯度=0
                #   else: grad_pg = -adv * ratio * (one_hot - softmax)
                #
                # KL 模式决定 grad_kl（同 GRPO）:
                #   sample : β*(one_hot - softmax)
                #   forward: β*π*[log π - log π_ref - KL]
                #   reverse: β*(π - π_ref)
                #   k3     : β*(1 - π_ref/π)*(one_hot - sm)
                # ==========================================

                # 解耦版 clip 行为判断
                clip_active = ((adv > 0 and ratio > 1.0 + self.epsilon_high) or
                               (adv < 0 and ratio < 1.0 - self.epsilon_low))
                if clip_active:
                    grad_pg = np.zeros_like(sm)
                    clip_activate_count += 1
                else:
                    grad_pg = -adv * ratio * (oh - sm)

                grad_logits = grad_pg + grad_kl

                # 构建 (1, S, V) 的 dLogits
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

        # ========== ③ Token-Level Loss 归一化 ==========
        if total_token_count == 0:
            return 0.0, {'clip_rate': 0.0, 'token_count': 0}

        n = total_token_count
        clip_rate = clip_activate_count / n

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

        stats = {
            'clip_rate': clip_rate,
            'token_count': n,
        }
        return total_loss / n, stats

    def update_old_policy(self):
        """更新 old_policy 为当前 policy 的深拷贝。"""
        self.old_policy = deepcopy(self.policy)


# =============================================================================
# 训练 / 评估工具
# =============================================================================

def make_prompt_dataset(n, vocab_size, seed=123):
    """生成固定 prompt 数据集：a, b ∈ {1, ..., 8} (排除 EOS=0)"""
    rng = np.random.RandomState(seed)
    prompts = []
    for _ in range(n):
        a = rng.randint(1, vocab_size)
        b = rng.randint(1, vocab_size)
        prompts.append((a, b))
    return np.array(prompts)


def evaluate(policy, prompts, max_resp_len=3, eos=0):
    """
    评估：自回归生成直到 EOS 或 max_resp_len。
    返回 (acc, avg_length, ratio_perfect, ratio_truncated, ratio_overlong)。
    """
    perfect = 0  # 完美生成（长度 == max_resp_len 且每 token 都对）
    total_correct_tokens = 0
    total_target_tokens = 0
    len_sum = 0
    n_truncated = 0   # 长度 < max_resp_len
    n_overlong = 0    # 长度 > max_resp_len（已 hard cap 截断）
    n = len(prompts)

    for (a, b) in prompts:
        p = np.array([[a, b]])
        target = [(a % 8) + 1, (b % 8) + 1, eos]
        seq = p.copy()
        generated = []
        for t in range(max_resp_len + 1):  # 最多采 max+1 看是否 overlong
            logits = policy.forward(seq)
            pred = int(np.argmax(logits[0, -1]))
            generated.append(pred)
            if pred == eos:
                break
            seq = np.concatenate([seq, np.array([[pred]])], axis=1)
        L = len(generated)
        len_sum += L
        if L < max_resp_len:
            n_truncated += 1
        elif L > max_resp_len:
            n_overlong += 1
        total_target_tokens += max_resp_len
        # 计算位置正确数
        token_correct = sum(1 for t in range(min(L, max_resp_len))
                            if generated[t] == target[t])
        total_correct_tokens += token_correct
        if L == max_resp_len and all(generated[t] == target[t] for t in range(max_resp_len)):
            perfect += 1

    acc = total_correct_tokens / total_target_tokens
    return {
        'token_acc': acc,
        'avg_length': len_sum / n,
        'perfect_ratio': perfect / n,
        'truncated_ratio': n_truncated / n,
        'overlong_ratio': n_overlong / n,
    }


def train_sft(policy, prompts, max_resp_len=3, eos=0, lr=0.005,
              epochs=300, grad_clip=1.0, teacher_forcing_ratio=0.5,
              use_scheduled_sampling=True):
    """
    SFT 预训练：教模型按 target 生成。

    关键改进 — Scheduled Sampling (DAgger 风格)：
      - teacher_forcing_ratio 概率：用 target token 当输入
      - (1 - teacher_forcing_ratio) 概率：用模型自己上一时刻的预测当输入
      - 这样能缓解 exposure bias，让 SFT 后的自回归性能与训练分布更一致
    """
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

        for (a, b) in prompts:
            p = np.array([[a, b]])
            target = [(a % 8) + 1, (b % 8) + 1, eos]

            # Scheduled sampling: 决定每个位置用 teacher 还是 model 自己的预测
            use_teacher = np.random.random(max_resp_len) < teacher_forcing_ratio
            # 第一个位置必须用 teacher（没有"自己的预测"可用）
            use_teacher[0] = True

            # 自回归构造 input
            input_seq = list(p[0])  # [a, b]
            policy.clear_gradients()  # 提前清，避免 SFT 阶段累积采样过程的状态
            for t in range(max_resp_len):
                seq = np.array([input_seq])
                logits = policy.forward(seq)
                pred = int(np.argmax(logits[0, -1]))
                if use_teacher[t]:
                    next_input = target[t]
                else:
                    next_input = pred
                input_seq.append(next_input)

            # 用最终 input_seq 算 loss
            seq = np.array([input_seq])
            logits = policy.forward(seq)
            sm = softmax(logits, axis=-1)
            dLogits = np.zeros_like(logits)
            for t in range(max_resp_len):
                pos = 2 + t
                dLogits[0, pos, :] = sm[0, pos, :].copy()
                dLogits[0, pos, target[t]] -= 1.0

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
            m = evaluate(policy, prompts, max_resp_len, eos)
            print(f"  [SFT] epoch {epoch:>3}: perfect={m['perfect_ratio']:.2f}, "
                  f"avg_len={m['avg_length']:.2f}, "
                  f"trunc={m['truncated_ratio']:.2f}, over={m['overlong_ratio']:.2f}")


# =============================================================================
# 训练主循环
# =============================================================================

def train_dapo(vocab_size=9, d_model=64, n_heads=4, d_ff=128,
               max_resp_len=3, overlong_buffer=1,
               group_size=16,
               epsilon_low=0.2, epsilon_high=0.28,
               dynamic_sampling=True, max_resample=4,
               overlong_soft_penalty=-0.5, overlong_hard_penalty=-1.0,
               truncation_penalty=-0.5,
               beta=0.01, lr=0.002, epochs=400, grad_clip=1.0,
               weight_decay=0.0001, kl_mode='sample', use_sft=True,
               sft_epochs=300, teacher_forcing_ratio=0.5,
               verbose=False, seed=42):
    """DAPO 完整训练流程：SFT 预热（带 scheduled sampling）→ DAPO 强化学习。"""
    print("=" * 70)
    print("DAPO 训练 One-Layer MHA Transformer")
    print("=" * 70)
    print(f"配置: vocab={vocab_size}, d_model={d_model}, heads={n_heads}")
    print(f"      max_resp_len={max_resp_len}, overlong_buffer={overlong_buffer}")
    print(f"      group_size={group_size}, ε_low={epsilon_low}, ε_high={epsilon_high}")
    print(f"      dynamic_sampling={dynamic_sampling}, max_resample={max_resample}")
    print(f"      beta={beta}, lr={lr}, epochs={epochs}, kl_mode={kl_mode}")

    np.random.seed(seed)
    EOS = 0

    policy = OneLayerTransformer(vocab_size, d_model, n_heads, d_ff,
                                  max_len=2 + max_resp_len + 1)
    ref_model = deepcopy(policy)

    trainer = DAPOTrainer(
        policy, ref_model,
        group_size=group_size,
        epsilon_low=epsilon_low, epsilon_high=epsilon_high,
        dynamic_sampling=dynamic_sampling, max_resample=max_resample,
        max_resp_len=max_resp_len, overlong_buffer=overlong_buffer,
        overlong_soft_penalty=overlong_soft_penalty,
        overlong_hard_penalty=overlong_hard_penalty,
        truncation_penalty=truncation_penalty,
        beta=beta, lr=lr, grad_clip=grad_clip, kl_mode=kl_mode,
    )

    prompts = make_prompt_dataset(8, vocab_size)

    # SFT 预热（带 Scheduled Sampling 缓解 exposure bias）
    if use_sft:
        print("\n[1] SFT 预热（Scheduled Sampling, teacher_forcing_ratio={}）".format(teacher_forcing_ratio))
        train_sft(policy, prompts, max_resp_len, EOS, lr=0.005,
                  epochs=sft_epochs, teacher_forcing_ratio=teacher_forcing_ratio)

    # 强化学习
    print("\n[2] DAPO 强化学习")
    losses = []
    mean_rewards = []
    mean_lengths = []
    clip_rates = []
    n_dynamic_skips = []

    best_perfect = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_len = 0.0
        epoch_clip = 0.0
        epoch_skip = 0
        count = 0

        for (a, b) in prompts:
            prompt = np.array([[a, b]])
            group, advantages, n_resample, n_zero = \
                trainer.sample_group_with_dynamic(prompt, verbose=False)
            if n_resample >= trainer.max_resample and np.std(
                    trainer.compute_rewards(prompt, group)) < 1e-6:
                # 真的没法采到非平凡 group，跳过
                epoch_skip += 1
                continue
            loss, stats = trainer.dapo_loss_and_gradients(prompt, group, advantages)
            rewards = trainer.compute_rewards(prompt, group)
            lengths = [item['length'] for item in group]
            epoch_loss += loss
            epoch_reward += float(np.mean(rewards))
            epoch_len += float(np.mean(lengths))
            epoch_clip += stats['clip_rate']
            epoch_skip += n_zero
            count += 1

        losses.append(epoch_loss / max(count, 1))
        mean_rewards.append(epoch_reward / max(count, 1))
        mean_lengths.append(epoch_len / max(count, 1))
        clip_rates.append(epoch_clip / max(count, 1))
        n_dynamic_skips.append(epoch_skip)

        # Weight decay
        if weight_decay > 0:
            for attr in ['W_q', 'W_k', 'W_v', 'W_o']:
                w = getattr(policy.mha, attr)
                setattr(policy.mha, attr, w * (1 - weight_decay))
            policy.ffn.fc1.W *= (1 - weight_decay)
            policy.ffn.fc2.W *= (1 - weight_decay)
            policy.proj.W *= (1 - weight_decay)

        if epoch % 5 == 0:
            trainer.update_old_policy()

        if epoch % 50 == 0 or epoch == epochs - 1:
            m = evaluate(policy, prompts, max_resp_len, EOS)
            best_perfect = max(best_perfect, m['perfect_ratio'])
            print(f"  epoch {epoch:>3}: loss={losses[-1]:.4f}, "
                  f"reward={mean_rewards[-1]:.2f}, "
                  f"avg_len={mean_lengths[-1]:.2f}, "
                  f"clip_rate={clip_rates[-1]:.2f}, "
                  f"perfect={m['perfect_ratio']:.2f}, "
                  f"trunc={m['truncated_ratio']:.2f}, "
                  f"over={m['overlong_ratio']:.2f}")

    final = evaluate(policy, prompts, max_resp_len, EOS)
    print("\n" + "=" * 70)
    print("DAPO 训练结果总结")
    print("=" * 70)
    print(f"最终 token 准确率:    {final['token_acc']:.3f}")
    print(f"最终完美率:          {final['perfect_ratio']:.3f}")
    print(f"最终平均长度:        {final['avg_length']:.2f} (target=3)")
    print(f"截断率:              {final['truncated_ratio']:.2f}")
    print(f"Overlong 率:         {final['overlong_ratio']:.2f}")
    print(f"最高完美率:          {best_perfect:.3f}")

    return {
        'losses': losses,
        'rewards': mean_rewards,
        'lengths': mean_lengths,
        'clip_rates': clip_rates,
        'final': final,
    }


# =============================================================================
# 数值梯度验证 — 验证 DAPO 的 clip 行为 + KL 梯度
# =============================================================================

def verify_dapo_gradients():
    """
    数值梯度验证。
    重点验证 DAPO 与 GRPO 的两个关键差异：
      ① Clip-Higher：ε_low/ε_high 解耦的 clip 行为
      ② KL 梯度：保持 GRPO 推导不变
    """
    print("=" * 70)
    print("DAPO 梯度数值验证 — Clip-Higher 行为 + 三种 KL 模式")
    print("=" * 70)

    np.random.seed(42)
    V = 9
    policy = OneLayerTransformer(V, d_model=8, n_heads=2, d_ff=16, max_len=6)

    # 固定输入
    seq = np.array([[1, 2, 3, 4, 5]])  # prompt_len=2, response_len=3
    mask = np.array([[False, False, True, True, True]])

    # 固定 old 和 ref 的 logits
    logits_old = np.random.randn(1, 5, V)
    logits_ref = np.random.randn(1, 5, V)

    adv = 1.5
    beta = 0.1
    epsilon_low = 0.2
    epsilon_high = 0.28

    pos = 2  # response 第一个位置
    target = seq[0, pos]

    # Precompute
    mr = np.max(logits_ref[0, pos])
    logsm_ref_vec = logits_ref[0, pos] - mr - np.log(np.sum(np.exp(logits_ref[0, pos] - mr)))
    sm_ref_vec = softmax(logits_ref[0, pos:pos+1], axis=-1)[0]

    mo = np.max(logits_old[0, pos])
    logsm_old_vec = logits_old[0, pos] - mo - np.log(np.sum(np.exp(logits_old[0, pos] - mo)))
    lpo = logsm_old_vec[target]

    lpr = logsm_ref_vec[target]

    def make_loss_fn(kl_mode):
        def loss_fn(logits_pi_pos):
            m = np.max(logits_pi_pos)
            logsm_pi_vec = logits_pi_pos - m - np.log(np.sum(np.exp(logits_pi_pos - m)))
            lp = logsm_pi_vec[target]
            sm_pi_vec = np.exp(logsm_pi_vec)

            ratio = np.exp(lp - lpo)
            # DAPO: 解耦 clip
            ratio_clip = np.clip(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
            loss_pg = -min(ratio * adv, ratio_clip * adv)

            if kl_mode == 'sample':
                loss_kl = beta * (lp - lpr)
            elif kl_mode == 'forward':
                kl = np.sum(sm_pi_vec * (logsm_pi_vec - logsm_ref_vec))
                loss_kl = beta * kl
            elif kl_mode == 'reverse':
                kl = np.sum(sm_ref_vec * (logsm_ref_vec - logsm_pi_vec))
                loss_kl = beta * kl
            elif kl_mode == 'k3':
                ratio_ref = np.exp(lpr - lp)
                k3 = (lp - lpr) - 1 + ratio_ref
                loss_kl = beta * k3
            else:
                raise ValueError(kl_mode)
            return loss_pg + loss_kl
        return loss_fn

    def analytical_grad(kl_mode):
        logits_pi = policy.forward(seq)
        sm = softmax(logits_pi[0, pos:pos+1], axis=-1)[0]
        oh = np.zeros_like(sm)
        oh[target] = 1.0

        m = np.max(logits_pi[0, pos])
        logsm_pi_vec = logits_pi[0, pos] - m - np.log(np.sum(np.exp(logits_pi[0, pos] - m)))
        lp = logsm_pi_vec[target]
        ratio = np.exp(lp - lpo)

        # DAPO clip 行为
        clip_active = ((adv > 0 and ratio > 1.0 + epsilon_high) or
                       (adv < 0 and ratio < 1.0 - epsilon_low))
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
        elif kl_mode == 'k3':
            ratio_ref = np.exp(lpr - lp)
            grad_kl = beta * (1 - ratio_ref) * (oh - sm)
        else:
            raise ValueError(kl_mode)

        return grad_pg + grad_kl

    # 数值梯度
    eps = 1e-5
    for kl_mode in ['sample', 'forward', 'reverse', 'k3']:
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

    # 额外验证：clip 边界行为
    print("\n[Clip-Higher 边界行为验证]")
    print(f"  ε_low={epsilon_low}, ε_high={epsilon_high}")
    # ratio 接近 1+ε_high 但不超过：clip 不触发
    # ratio 略超 1+ε_high：clip 触发（adv>0 时）
    # 用一个人工 ratio 测试
    for r in [0.5, 0.7, 0.95, 1.0, 1.05, 1.2, 1.28, 1.3, 1.5]:
        clip = max(1.0 - epsilon_low, min(r, 1.0 + epsilon_high))
        active = ((adv > 0 and r > 1.0 + epsilon_high) or
                  (adv < 0 and r < 1.0 - epsilon_low))
        print(f"  ratio={r:.2f}  →  clip_value={clip:.2f}  "
              f"clip_active={active}  (adv>0={adv>0})")
    print()
    return True


# =============================================================================
# Demo: DAPO vs GRPO 对比
# =============================================================================

def demo():
    # 1. 梯度验证
    verify_dapo_gradients()

    # 2. DAPO 训练
    print("=" * 70)
    print("DAPO 训练（带 4 项改进）")
    print("=" * 70)
    res_dapo = train_dapo(
        vocab_size=9, d_model=64, n_heads=4, d_ff=128,
        max_resp_len=3, overlong_buffer=1,
        group_size=16,
        epsilon_low=0.2, epsilon_high=0.28,
        dynamic_sampling=True, max_resample=4,
        beta=0.01, lr=0.002, epochs=200, grad_clip=1.0,
        weight_decay=0.0001, kl_mode='sample', use_sft=True,
        sft_epochs=150, seed=42,
    )

    # 3. DAPO 消融：关闭 Clip-Higher
    print("\n" + "=" * 70)
    print("消融 1: ε_low = ε_high = 0.2（对称 clip，等价于 GRPO）")
    print("=" * 70)
    res_ablation_clip = train_dapo(
        vocab_size=9, d_model=64, n_heads=4, d_ff=128,
        max_resp_len=3, overlong_buffer=1,
        group_size=16,
        epsilon_low=0.2, epsilon_high=0.2,  # 对称 → 退化为 GRPO clip
        dynamic_sampling=True, max_resample=4,
        beta=0.01, lr=0.002, epochs=200, grad_clip=1.0,
        weight_decay=0.0001, kl_mode='sample', use_sft=True,
        sft_epochs=150, seed=42,
    )

    # 4. DAPO 消融：关闭 Dynamic Sampling
    print("\n" + "=" * 70)
    print("消融 2: 关闭 Dynamic Sampling")
    print("=" * 70)
    res_ablation_dyn = train_dapo(
        vocab_size=9, d_model=64, n_heads=4, d_ff=128,
        max_resp_len=3, overlong_buffer=1,
        group_size=16,
        epsilon_low=0.2, epsilon_high=0.28,
        dynamic_sampling=False,  # 关闭
        beta=0.01, lr=0.002, epochs=200, grad_clip=1.0,
        weight_decay=0.0001, kl_mode='sample', use_sft=True,
        sft_epochs=150, seed=42,
    )

    # 5. DAPO 消融：关闭 Overlong Reward
    print("\n" + "=" * 70)
    print("消融 3: 关闭 Overlong Reward Shaping")
    print("=" * 70)
    res_ablation_ol = train_dapo(
        vocab_size=9, d_model=64, n_heads=4, d_ff=128,
        max_resp_len=3, overlong_buffer=1,
        group_size=16,
        epsilon_low=0.2, epsilon_high=0.28,
        dynamic_sampling=True, max_resample=4,
        overlong_soft_penalty=0.0,  # 不惩罚
        overlong_hard_penalty=0.0,  # 不惩罚
        truncation_penalty=0.0,     # 不惩罚
        beta=0.01, lr=0.002, epochs=200, grad_clip=1.0,
        weight_decay=0.0001, kl_mode='sample', use_sft=True,
        sft_epochs=150, seed=42,
    )

    # 6. 总结
    print("\n" + "=" * 70)
    print("消融对比总结")
    print("=" * 70)
    print(f"{'配置':<35} {'完美率':>10} {'平均长度':>10} {'Overlong率':>10}")
    print("-" * 70)
    for name, res in [('完整 DAPO', res_dapo),
                      ('消融 Clip-Higher', res_ablation_clip),
                      ('消融 Dynamic Sampling', res_ablation_dyn),
                      ('消融 Overlong Shaping', res_ablation_ol)]:
        f = res['final']
        print(f"{name:<35} {f['perfect_ratio']:>9.2f} {f['avg_length']:>10.2f} "
              f"{f['overlong_ratio']:>10.2f}")


if __name__ == "__main__":
    demo()
