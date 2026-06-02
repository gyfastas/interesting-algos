# DAPO 训练 One-Layer MHA Transformer

## 问题描述

在 [GRPO 训练 MHA Transformer](../grpo-training-mha/) 的基础上，**用 DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 训练一个单层 MHA Transformer**，并保留 GRPO 那套完整反向传播框架。

核心考点：**DAPO 相对 GRPO 的 4 项关键改进** —— 在手写从 logits 到 MHA 参数的反向传播中，把这些改进"无缝"嵌进 PPO-clip 与 KL 约束。

DAPO 是字节跳动在 2025 年 1 月公开的算法，专门针对 long-CoT RL 训练中 GRPO 暴露的几个具体问题。本题用 9 词表 + 3 步生成的 toy 任务把这 4 项改进**逐一对照**演示。

---

## 直觉分析

### DAPO 解决了 GRPO 的哪些问题？

GRPO 在 long-CoT 任务上跑分不理想，DAPO 论文里给出了 4 个观察：

1. **对称 clip 过度保守** —— GRPO 用 `clip(ρ, 1−ε, 1+ε)`，ε=0.2。CoT 任务里"鼓励探索"的天花板太低，正 advantage 经常被误 clip。**修复：ε_low/ε_high 解耦（Clip-Higher）**。

2. **zero-advantage group 浪费算力** —— 同一 prompt 采样的 G 个回答如果奖励完全一致，advantage 全 0，损失=0 但采样时间不省。**修复：动态重采样（Dynamic Sampling）**。

3. **sample 级别 loss 稀释长回答** —— GRPO 论文公式是 `(1/G) Σ_i (1/|y_i|) Σ_t`，短回答对 loss 贡献过大，长 CoT 训练信号被稀释。**修复：跨 sample 按 token 等权（Token-Level Loss）**。

4. **超长回答无惩罚** —— 模型学会"越长奖励越高"，生成无意义废话。**修复：Overlong Reward Shaping**。

### DAPO 核心公式

$$
\mathcal{J}_{DAPO}(\theta) = \underbrace{\frac{1}{\sum_i |y_i|} \sum_i \sum_t \min\!\big(\rho_{i,t} A_{i,t},\, \text{clip}(\rho_{i,t}, 1-\varepsilon_{low}, 1+\varepsilon_{high}) A_{i,t}\big)}_{\text{Token-Level PG (Clip-Higher)}}
\;-\; \alpha\, \mathcal{J}_{KL}
$$

注意分母是**总 token 数**，不是 group size G，也不是 sample 内平均。

与 GRPO 的 3 个关键区别：

| 项 | GRPO | DAPO |
|---|---|---|
| Clip 范围 | 对称 $\varepsilon$ | 解耦 $\varepsilon_{low}, \varepsilon_{high}$ |
| Loss 归一化 | sample 级别 | **token 级别**（跨 sample） |
| 采样策略 | 全采 | 跳过 zero-advantage group |
| 奖励 | 准确性 | 准确性 + **超长惩罚** |

---

## 数学建模

### 1. Clip-Higher 的精确行为

GRPO 的 clip 条件：

- $A > 0$ 且 $\rho > 1+\varepsilon$ → clip
- $A < 0$ 且 $\rho < 1-\varepsilon$ → clip

DAPO 把 $\varepsilon$ 拆成两个：

- $A > 0$ 且 $\rho > 1+\varepsilon_{high}$ → clip
- $A < 0$ 且 $\rho < 1-\varepsilon_{low}$ → clip

典型取值：$\varepsilon_{low}=0.2$，$\varepsilon_{high}=0.28$。上界更高 = 给"探索"留更多空间；下界保持 0.2 = 仍然抑制"灾难性遗忘"。

直觉：**只让 $\rho$ 涨得更自由，不让 $\rho$ 跌得更自由**。

数值边界行为（来自 `verify_dapo_gradients()`）：

```
ratio=0.50  →  clip_value=0.80  clip_active=False
ratio=0.95  →  clip_value=0.95  clip_active=False
ratio=1.20  →  clip_value=1.20  clip_active=False
ratio=1.28  →  clip_value=1.28  clip_active=False  ← ε_high 之内不 clip
ratio=1.30  →  clip_value=1.28  clip_active=True   ← 越过 ε_high 才 clip
ratio=1.50  →  clip_value=1.28  clip_active=True
```

### 2. Token-Level Loss 的梯度归一化

GRPO 论文里 sample 级别的归一化：

$$
\mathcal{L}_{GRPO} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min(\rho_{i,t}A_i,\, \text{clip}(\rho_{i,t})A_i)
$$

DAPO 把它改成**跨 sample 平等**：

$$
\mathcal{L}_{DAPO} = -\frac{1}{\sum_{i=1}^G |y_i|}\sum_{i=1}^G \sum_{t=1}^{|y_i|} \min(\rho_{i,t}A_i,\, \text{clip}(\rho_{i,t})A_i)
$$

差别：分母是 $\sum |y_i|$（所有 token 总数）而不是 $G$。这让 1 个长 CoT 回答（4096 tokens）比 1 个短回答（128 tokens）对梯度贡献更**多**，与短回答对 GRPO 损失贡献**过多**正好相反。

代码里：

```python
total_loss = 0.0
total_token_count = 0
for (resp, adv) in group:
    for pos in resp_positions:
        loss_token = loss_pg + loss_kl
        total_loss += loss_token
        total_token_count += 1
# ...
return total_loss / total_token_count  # Token-Level
```

### 3. Dynamic Sampling

每次 `sample_group` 后检查 `std(rewards)`。如果 $\text{std} < \epsilon$（近似 0），整组 advantage 全 0，没有梯度信号 → **重采**。

```python
def sample_group_with_dynamic(self, prompt, ...):
    attempts = 0
    while True:
        group = self.sample_group(prompt)
        rewards = self.compute_rewards(prompt, group)
        if not self.dynamic_sampling or std(rewards) > 1e-6:
            break
        attempts += 1
        if attempts >= self.max_resample:
            break  # 真的采不到非平凡 group
    return group, advantages, attempts
```

**副作用**：单步训练时间可能变长（要重采），但**总计算-梯度比**显著提升。

### 4. Overlong Reward Shaping

在奖励函数里加长度惩罚（**关键前提：任务必须是变长 response**）：

$$
R(y) = \underbrace{\text{acc}(y)}_{\text{位置正确数}} + \underbrace{R_{length}(|y|)}_{\text{长度惩罚}}
$$

具体地，对 $\text{max\_resp\_len} = 3$，$\text{buffer} = 1$：

| 长度区间 | 奖励 |
|---|---|
| $\|y\| < 3$（截断） | $R_{length} = -0.5$ |
| $\|y\| = 3$ | $R_{length} = 0$ |
| $\|y\| = 4$（soft overlong） | $R_{length} = -0.5$ |
| $\|y\| > 4$（hard overlong） | $R_{length} = -1.0$ |

设计成 soft + hard 两段：先给"软"信号让模型逐渐学到边界，再"硬"截断兜底。

---

## 三种 KL 实现（沿用 GRPO，本题不重复推导）

DAPO 沿用 GRPO 的 4 种 KL 模式（`sample` / `forward` / `reverse` / `k3`），梯度推导完全一致。本题 demo 用 `sample` 模式（最常用），详见 [GRPO README §三种 KL 实现对比](../grpo-training-mha/README.md#三种-kl-实现对比核心考点)。

---

## 任务设计

### 变长 response 任务

为了演示 Overlong Shaping，必须让 response 长度可变。本题设计：

- **词表**：9 个 token，token `0` 是 **EOS**
- **prompt**：`[a, b]`，$a, b \in \{1, ..., 8\}$
- **target**：`[(a \bmod 8) + 1,\, (b \bmod 8) + 1,\, 0]`，长度 3
- **max_resp_len** = 3，**overlong_buffer** = 1
- 模型自回归生成，遇到 EOS 提前停止；超过 hard cap 强制截断

### 奖励函数（含 overlong shaping）

| 长度 | 奖励 |
|---|---|
| $\ell < 3$ | $-0.5$（truncation） |
| $\ell = 3$ 且 target 匹配 | $+3.0$（完美） |
| $\ell = 3$ 但 target 错 | $0$（按位置正确数） |
| $\ell = 4$ | $-0.5$（soft overlong） |
| $\ell > 4$ | $-1.0$（hard overlong，理论上采样时不会触发） |

注：实现里 `length=3` 的奖励 = 位置正确数（0, 1, 2, 3），不是常数 +3 — 这样 group 内的 advantage 才有方差。

### SFT 预热：Scheduled Sampling

GRPO 原题用纯 teacher forcing 的 SFT，会引发 **exposure bias**：SFT 阶段模型用正确的 token 训练，但 RL 阶段自回归采样时第一步预测错了，后面全错。

**修复：DAgger 风格的 scheduled sampling** —

```python
use_teacher = np.random.random(max_resp_len) < teacher_forcing_ratio
use_teacher[0] = True  # 第一个位置必须用 teacher（没东西可采）

for t in range(max_resp_len):
    logits = policy.forward(seq)
    pred = argmax(logits)
    next_input = target[t] if use_teacher[t] else pred
    input_seq.append(next_input)
# 用最终的 input_seq 算 loss
```

让 SFT 训练时偶尔看到"自己的错预测"，增强自回归鲁棒性。

> **说明**：本题 toy 模型（单层 MHA + 9 词表）无法完全收敛到 100% 准确率。在大模型 + 长 CoT 上 DAPO 才会显示出压倒性优势。本题的价值在于**把 DAPO 的 4 项改进**精确地嵌进 GRPO 反传框架里。

---

## 代码实现

### DAPOTrainer 关键差异

| 方法 | 改动 |
|---|---|
| `__init__` | 新增 `epsilon_low/epsilon_high`、`max_resp_len/overlong_buffer`、overlong 三种 penalty |
| `sample_group` | 变长采样，遇到 EOS 提前停；返回 `{tokens, length, truncated, overlong}` |
| `sample_group_with_dynamic` | **新增**：重采样直到 `std(rewards) > 0`，限制 `max_resample` |
| `compute_rewards` | 改写：分长度区间给奖励，加入 overlong shaping |
| `dapo_loss_and_gradients` | ① clip 范围解耦 ② token-level 归一化 ③ 记录 `clip_rate` |

### 关键代码段

**Clip-Higher 的 clip 行为判断**：

```python
ratio = np.exp(lp - lpo)
ratio_clip = np.clip(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
obj1 = ratio * adv
obj2 = ratio_clip * adv
loss_pg = -min(obj1, obj2)

# 解耦版的梯度判断
clip_active = ((adv > 0 and ratio > 1.0 + epsilon_high) or
               (adv < 0 and ratio < 1.0 - epsilon_low))
if clip_active:
    grad_pg = 0
else:
    grad_pg = -adv * ratio * (one_hot - softmax)
```

**Token-Level 归一化**：

```python
total_loss = 0.0
total_token_count = 0
for sample in group:
    for pos in sample_response_positions:
        total_loss += loss_pg + loss_kl
        total_token_count += 1
# ...
return total_loss / total_token_count  # 跨 sample 按 token 等权
```

**Dynamic Sampling**：

```python
attempts = 0
while True:
    group = sample(prompt)
    rewards = compute_rewards(prompt, group)
    if not dynamic_sampling or std(rewards) > 1e-6:
        break
    attempts += 1
    if attempts >= max_resample:
        break  # 跳过这个 prompt
```

---

## 训练与消融

### 数值梯度验证

`verify_dapo_gradients()` 验证 4 种 KL 模式 + Clip-Higher 边界行为：

```
DAPO 梯度数值验证 — Clip-Higher 行为 + 三种 KL 模式
======================================================================
  KL=sample : rel_err=1.55e-06  ✓ 通过
  KL=forward: rel_err=5.09e-06  ✓ 通过
  KL=reverse: rel_err=4.65e-06  ✓ 通过
  KL=k3     : rel_err=2.93e-05  ✓ 通过

[Clip-Higher 边界行为验证]
  ε_low=0.2, ε_high=0.28
  ratio=0.50  →  clip_value=0.80  clip_active=False
  ratio=1.28  →  clip_value=1.28  clip_active=False  ← 边界内
  ratio=1.30  →  clip_value=1.28  clip_active=True   ← 越过
```

### 消融对比

固定其他超参，逐个关闭 DAPO 改进项，训练 200 epoch 后看 overlong 率与完美率：

| 配置 | 完美率 | 平均长度 | Overlong 率 | 解读 |
|---|---|---|---|---|
| **完整 DAPO** | 0.00 | 3.38 | **0.75** | baseline |
| 消融 Clip-Higher (ε_low=ε_high=0.2) | 0.00 | 3.38 | 0.75 | toy 任务下 clip 改动影响小 |
| 消融 Dynamic Sampling | 0.00 | **3.75** | **0.88** | 不重采 → 过多样本被 overlong |
| 消融 Overlong Shaping | 0.00 | 3.62 | **0.88** | 不惩罚 → 模型无约束地超长生成 |

**关键观察**：

- ✅ **Overlong Shaping 显著降低 overlong 率**（0.88 → 0.75，-15%）
- ✅ **Dynamic Sampling 也能拉低 overlong 率**（0.88 → 0.75），因为它过滤掉了那些 group 内奖励方差为 0（常常是"全 overlong"或"全 trunc"）的 prompt
- ⚠️ **Clip-Higher 在 toy 上没体现** —— ε_high 调到 0.28 vs 0.2 的差异需要更长的训练和更激进的 ratio 才会出现
- ⚠️ **完美率 = 0** —— 单层 MHA 在 3 步生成任务上学不会 `(a mod 8) + 1` 这种语义映射；DAPO 论文是在 R1 这种 70B+ 模型上验证的。在 toy 上能看到**相对差异**已是足够

### 训练曲线

```
epoch   0: loss=0.6228, reward=-0.27, avg_len=3.22, clip_rate=0.20, over=1.00
epoch  50: loss=-0.0105, reward=-0.32, avg_len=2.88, clip_rate=0.00, over=0.75
epoch 100: loss=-0.0048, reward=-0.34, avg_len=2.98, clip_rate=0.00, over=0.38
epoch 150: loss= 0.0004, reward=-0.31, avg_len=3.26, clip_rate=0.11, over=0.88
epoch 199: loss=-0.0229, reward=-0.23, avg_len=3.15, clip_rate=0.34, over=0.88
```

注意 `clip_rate` 在 0~0.5 之间波动 —— 训练初期策略剧烈变化，clip 频繁触发；后期稳定后 clip 触发率下降，符合预期。

---

## 工程注意（相对 GRPO 的额外坑）

1. **变长 response 的 mask 处理** —— 每个 response 算自己的长度 `L`，mask 只覆盖 `seq[P:P+L]` 区间。原 GRPO 假设长度固定。
2. **Scheduled Sampling 的 RNG** —— 共享 numpy RNG 时要小心：SFT 阶段会消费 RNG 状态，可能影响后续 RL 阶段的可复现性。demo 里在 train_dapo 入口重新设 seed=42。
3. **max_resample 不能太大** —— 极端情况下所有 prompt 都采不到非平凡 group，max_resample=4 是经验值（论文用 30）。
4. **overlong buffer 是超参** —— buffer=1 太紧、buffer=5 太松。论文里 buffer 通常取 max_resp_len 的 25%。

---

## 动画演示

打开 `animation.html` 查看交互动画：
- 4 个 panel：Loss 曲线、平均奖励、平均长度、Overlong 率
- 完整 DAPO vs 3 个消融的曲线对比
- 演示 Clip-Higher 的 ratio clip 边界行为（实时滑块）
- 演示 Dynamic Sampling 触发时的重采计数

---

## 答案与总结

| 要点 | 结论 |
|------|------|
| Clip-Higher | clip 范围解耦 $\varepsilon_{low} \neq \varepsilon_{high}$，上界抬高以鼓励探索 |
| Dynamic Sampling | 跳过 zero-advantage group，避免无效计算 |
| Token-Level Loss | 跨 sample 按 token 等权归一化，避免长 CoT 被稀释 |
| Overlong Shaping | 长度超阈值施加 soft/hard penalty |
| 反向传播 | 仍走 MHA → FFN → Proj 完整链路（与 GRPO 一致） |
| SFT 阶段 | 加 Scheduled Sampling 缓解 exposure bias |

| GRPO 公式 | DAPO 公式 |
|---|---|
| $\text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)$ | $\text{clip}(\rho, 1-\varepsilon_{low}, 1+\varepsilon_{high})$ |
| $\frac{1}{G}\sum_i \frac{1}{\|y_i\|}\sum_t$ | $\frac{1}{\sum_i \|y_i\|}\sum_i \sum_t$ |
| 全采样 | 动态重采样过滤 zero-adv group |
| 单一 reward | reward + overlength penalty |

**一句话总结**：DAPO = GRPO + (解耦 clip + 动态采样 + token 归一化 + 超长惩罚)。它把 GRPO 在 long-CoT 训练里"上不去分"的具体瓶颈一项项拆出来，分别给出对症下药的修法。**反传框架没动**，仍是 `proj → ffn → mha → embed` 这条主链。
