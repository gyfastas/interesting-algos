# GRPO 训练 One-Layer MHA Transformer

## 问题描述

在 [DPO 训练 MHA Transformer](../dpo-training-mha/) 的基础上，**用 GRPO (Group Relative Policy Optimization) 训练一个单层 MHA Transformer，并验证收敛性**。

核心考点：**GRPO Loss 的梯度推导** —— 手写从 logits 到 MHA 参数的完整反向传播，包括 importance sampling ratio、PPO clip 和 KL 散度约束。

---

## 直觉分析

### 什么是 GRPO？

GRPO 是 DeepSeek 提出的 RLHF 算法，核心思想：

1. **不需要价值函数 (Value Model)** —— 用 group 内样本的相对奖励代替优势估计
2. **Group-Relative Baseline** —— 同一 group 内的样本互相做 baseline
3. **PPO-Clip + KL 约束** —— 防止策略更新过大

```
传统 PPO:  SFT → 训练 Reward Model → 训练 Value Model → PPO 优化
GRPO:      SFT → 训练 Reward Model → 直接 Group Sampling 优化（无 Value Model）
```

### GRPO 的核心公式

对于每个 prompt，从旧策略 $\pi_{old}$ 采样一个 group $G = \{y_1, y_2, ..., y_G\}$：

**1. 计算奖励**

$$r_i = \text{Reward}(x, y_i)$$

**2. Group-Relative Advantage**

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$$

**3. Importance Sampling Ratio**

$$\rho_i = \frac{\pi(y_i|x)}{\pi_{old}(y_i|x)}$$

**4. PPO-Clip Objective**

$$\mathcal{L}_{PG} = -\min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \cdot A_i)$$

**5. KL 散度约束**

$$\mathcal{L}_{KL} = \beta \cdot \left(\log \pi(y_i|x) - \log \pi_{ref}(y_i|x)\right)$$

**总 Loss**

$$\mathcal{L} = \mathcal{L}_{PG} + \mathcal{L}_{KL}$$

---

## 数学建模

### 1. 梯度推导（核心考点）

**Step 1：log-softmax 的导数**

对于 $\log P = \text{log-softmax}(\text{logits})[\text{target}]$：

$$\frac{d(\log P)}{d(\text{logits})} = \mathbf{1}_{\text{target}} - \text{softmax}(\text{logits})$$

**Step 2：Importance Ratio 的导数**

$$\rho = \exp(\log \pi - \log \pi_{old})$$

$$\frac{d\rho}{d(\text{logits})} = \rho \cdot (\mathbf{1}_{\text{target}} - \text{softmax})$$

**Step 3：Policy Gradient（未 clip 时）**

$$\frac{d\mathcal{L}_{PG}}{d(\text{logits})} = -A \cdot \rho \cdot (\mathbf{1} - \text{softmax})$$

**Step 4：KL Penalty 的导数**

$$\frac{d\mathcal{L}_{KL}}{d(\text{logits})} = \beta \cdot (\mathbf{1} - \text{softmax})$$

**Step 5：合梯度**

$$\frac{d\mathcal{L}}{d(\text{logits})} = -A \cdot \rho \cdot (\mathbf{1} - \text{softmax}) + \beta \cdot (\mathbf{1} - \text{softmax})$$

### 2. Clip 判断

| 条件 | 行为 | 梯度 |
|---|---|---|
| $A > 0$ 且 $\rho > 1+\epsilon$ | clip 生效 | $\text{grad}_{PG} = 0$ |
| $A < 0$ 且 $\rho < 1-\epsilon$ | clip 生效 | $\text{grad}_{PG} = 0$ |
| 其他情况 | 未 clip | $\text{grad}_{PG} = -A \cdot \rho \cdot (\mathbf{1} - \text{softmax})$ |

### 3. 关键观察

- **正 advantage ($A > 0$)**：提升当前回答的概率
- **负 advantage ($A < 0$)**：降低当前回答的概率
- **clip 的作用**：防止 ratio 过大导致策略剧烈变化
- **KL 的作用**：防止策略偏离参考模型太远

---

## 三种 KL 实现对比（核心考点）

GRPO 中 KL penalty 的实现有三种常见方式，面试中经常被问到：

### ① 单点估计 (Sample-based) — 最常用

$$
\mathcal{L}_{KL} = \beta \cdot \left[\log \pi_\theta(y|x) - \log \pi_{ref}(y|x)\right]$$

**梯度：**

$$\nabla \mathcal{L}_{KL} = \beta \cdot (\mathbf{1}_y - \pi_\theta)$$

**特点：**
- 只对**采样到的 y** 计算梯度
- **π_ref 不进入梯度方向**（被视为常数）
- 计算 O(1)，工程上最常用（OpenAI PPO、早期 TRL）
- 严格来说是 KL 的单点蒙特卡洛估计，而非真正 KL

### ② Forward KL — 理论最严谨

$$
\text{KL}(\pi_\theta \| \pi_{ref}) = \sum_v \pi_\theta(v) \left[\log \pi_\theta(v) - \log \pi_{ref}(v)\right]$$

**梯度：**

$$\nabla_j \text{KL} = \pi_\theta(j) \cdot \left[\log \pi_\theta(j) - \log \pi_{ref}(j) - \text{KL}\right]$$

**特点：**
- 对所有 vocab token 求和
- **π_ref 显式影响梯度**：π_ref(j) 越大，越抑制 π_θ(j)
- 计算 O(V)，LLM 中较贵
- 目标：让 π_θ 的分布「覆盖」π_ref 的分布

### ③ Reverse KL — 形式最简洁

$$
\text{KL}(\pi_{ref} \| \pi_\theta) = \sum_v \pi_{ref}(v) \left[\log \pi_{ref}(v) - \log \pi_\theta(v)\right]$$

**梯度：**

$$\nabla_j \text{KL} = \pi_\theta(j) - \pi_{ref}(j)$$

**特点：**
- 梯度形式极简：`sm - sm_ref`
- **π_ref 显式影响梯度方向**
- 计算 O(V)
- 目标：让 π_θ 的分布「不超出」π_ref 的支持范围

### 对比总结

| 模式 | Loss 形式 | 梯度 | π_ref 是否影响梯度 | 计算复杂度 | 工业界使用 |
|------|-----------|------|-------------------|-----------|-----------|
| **sample** | β·[log π(y) − log π_ref(y)] | β·(1_y − π) | ❌ 否 | O(1) | ⭐⭐⭐ 最常用 |
| **forward** | β·Σ π(v)[log π(v) − log π_ref(v)] | β·π·[log π − log π_ref − KL] | ✅ 是 | O(V) | ⭐⭐ 较严谨 |
| **reverse** | β·Σ π_ref(v)[log π_ref(v) − log π(v)] | β·(π − π_ref) | ✅ 是 | O(V) | ⭐⭐ 形式简洁 |

> **面试考点**：单点估计的 KL 里 π_ref 实际上只影响 loss 数值，不影响梯度方向。如果要让 π_ref 真正约束策略，需要用 Forward KL 或 Reverse KL。

---

## 代码实现

### GRPO 核心

```python
class GRPOTrainer:
    def grpo_loss_and_gradients(self, prompt, group, advantages):
        for g, (resp, adv) in enumerate(zip(group, advantages)):
            seq = concat([prompt, resp], axis=1)
            logits_pi = policy.forward(seq)
            logits_old = old_policy.forward(seq)
            logits_ref = ref_model.forward(seq)

            lp = log_softmax(logits_pi)[target]
            lpo = log_softmax(logits_old)[target]
            lpr = log_softmax(logits_ref)[target]

            ratio = exp(lp - lpo)
            ratio_clip = clip(ratio, 1 - eps, 1 + eps)

            # PPO-clip
            loss_pg = -min(ratio * adv, ratio_clip * adv)

            # === 三种 KL 实现 ===
            sm = softmax(logits_pi)
            sm_ref = softmax(logits_ref)

            if kl_mode == 'sample':
                # 单点估计 — π_ref 不影响梯度
                loss_kl = beta * (lp - lpr)
                grad_kl = beta * (oh - sm)

            elif kl_mode == 'forward':
                # Forward KL — 完整求和
                kl = sum(sm[v] * (log(sm[v]) - log(sm_ref[v])) for v in range(V))
                loss_kl = beta * kl
                grad_kl = beta * sm * (log(sm) - log(sm_ref) - kl)

            elif kl_mode == 'reverse':
                # Reverse KL — 形式最简洁
                kl = sum(sm_ref[v] * (log(sm_ref[v]) - log(sm[v])) for v in range(V))
                loss_kl = beta * kl
                grad_kl = beta * (sm - sm_ref)

            grad_logits = grad_pg + grad_kl

            # Backward 穿过 MHA
            dEmbed = policy.backward(dLogits)
            ...
```

### 梯度回传路径

```
GRPO Loss
    │
    ├── dLogits ──→ proj.backward ──→ ffn.backward ──→ mha.backward ──→ dEmbed
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
                                        累加所有 response 的梯度 + 更新参数
```

注意：**group 中每个 response 单独 forward + backward**，梯度累加后统一更新。

---

## 训练与收敛

### 任务设计

固定数据集：
- prompt = $[a]$（$a \in [0, 7]$）
- target = $[(a+1) \bmod 8]$
- reward = 1（target 正确）或 0（错误）

模型需要通过 causal attention 学习：给定 prompt $a$，生成 $a+1$。

### 训练结果（默认 sample 模式）

```
epoch   0: loss=0.000324, mean_reward=0.11, acc=0.12
epoch 100: loss=-0.066712, mean_reward=0.41, acc=0.75
epoch 200: loss=-0.013144, mean_reward=0.71, acc=0.75
epoch 300: loss=-0.008568, mean_reward=0.78, acc=0.75
epoch 400: loss=-0.000280, mean_reward=0.84, acc=0.88
epoch 500: loss=0.010765, mean_reward=0.87, acc=0.88
epoch 599: loss=0.011541, mean_reward=0.88, acc=0.88
```

- **Loss**：从 ~0 下降到负值（说明 reward 提升）再回升（模型收敛）
- **平均奖励**：从 0.11（随机）提升到 **0.88**
- **准确率**：从 12%（随机）提升到 **88%**

### 三种 KL 模式对比（300 epochs）

| KL 模式 | 最终 Reward | 最终 Acc | 特点 |
|---------|------------|---------|------|
| **sample** | 0.75 | 75% | 快速收敛，π_ref 不约束梯度方向 |
| **forward** | 0.73 | 75% | 最严谨，完整 Forward KL |
| **reverse** | 0.78 | **100%** | 形式简洁，意外表现最好 |

### 关键超参数

| 参数 | 值 | 说明 |
|------|---|------|
| group_size | 32 | 较大 group 提供更稳定的 advantage 估计 |
| beta | 0.01 | 较小的 KL 系数，允许策略适当偏离参考模型 |
| epsilon | 0.2 | PPO clip 范围 |
| lr | 0.002 | 学习率 |
| weight_decay | 0.0001 | 防止过拟合 |

---

## 动画演示

> 打开 `animation.html` 查看交互动画：
> - GRPO Loss 曲线随训练变化
> - 平均奖励从随机提升到 0.88
> - 准确率从 12% 提升到 88%
> - Group 内 reward 分布可视化

---

## 答案与总结

| 要点 | 结论 |
|------|------|
| GRPO 核心思想 | 用 group 内相对奖励代替 Value Model，简化 RLHF 流程 |
| Advantage 计算 | $A_i = (r_i - \text{mean}) / (\text{std} + \epsilon)$ |
| Policy Gradient | $-A \cdot \rho \cdot (\mathbf{1} - \text{softmax})$ |
| PPO Clip | 当 $A>0$ 且 $\rho > 1+\epsilon$ 或 $A<0$ 且 $\rho < 1-\epsilon$ 时梯度为 0 |
| KL (sample) | $\beta \cdot (\mathbf{1}_y - \text{softmax})$，π_ref 不影响梯度方向 |
| KL (forward) | $\beta \cdot \pi \cdot [\log \pi - \log \pi_{ref} - \text{KL}]$，完整严谨 |
| KL (reverse) | $\beta \cdot (\pi - \pi_{ref})$，形式最简洁 |
| 工程注意 | 每个 response 单独 forward/backward，梯度累加后统一更新 |

**一句话总结**：GRPO = Group Sampling + Relative Advantage + PPO-Clip + KL —— 不需要 Value Model，用组内竞争代替全局评价，把高奖励回答往上拉，低奖励回答往下压。KL 的实现方式决定了 π_ref 对策略的约束强度。
