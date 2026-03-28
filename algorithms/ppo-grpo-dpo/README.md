# PPO vs GRPO vs DPO — RLHF 三大算法对比

## 问题描述

RLHF (Reinforcement Learning from Human Feedback) 是让 LLM 对齐人类偏好的核心技术。三种主流算法：

- **PPO** (2017, OpenAI) — 经典 RL 算法，ChatGPT 使用
- **DPO** (2023, Stanford) — 去掉 reward model，直接从偏好对优化
- **GRPO** (2024, DeepSeek) — 去掉 value function，用组内相对排名，DeepSeek-R1 使用

## RLHF 的大背景

LLM 训练三阶段：

```
Stage 1: Pretraining     → 学习语言能力 (next-token prediction)
Stage 2: SFT             → 学习指令遵循 (supervised fine-tuning)
Stage 3: RLHF/Alignment  → 对齐人类偏好 (PPO / DPO / GRPO)
```

Stage 3 的目标：让模型不只是"能回答"，而是"回答得好"——更有帮助、更安全、更符合人类偏好。

## PPO：经典强化学习路线

### 训练流程

```
           ┌──────────────┐
Prompt ──→ │ Policy (LLM) │ ──→ Response
           └──────────────┘
                  │
                  ▼
           ┌──────────────┐
           │ Reward Model │ ──→ Reward score
           └──────────────┘
                  │
                  ▼
           ┌──────────────┐
           │ Value Model  │ ──→ Baseline V(s)
           └──────────────┘
                  │
                  ▼
         Advantage = R - V(s)
                  │
                  ▼
        PPO Clipped Objective
```

需要 **4 个模型同时在显存中**：
1. **Policy** (当前要训练的 LLM)
2. **Reference Policy** (冻��的 SFT 模型，防止偏离太远)
3. **Reward Model** (从人类偏好数据训练的打分器)
4. **Value Model** (估计每个状态的期望回报)

### 核心公式

$$L^{PPO} = -\mathbb{E}\left[\min\left(r_t A_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中：
- $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ — 重要性采样比
- $A_t = R_t - V(s_t)$ — Advantage (reward - baseline)
- $\epsilon = 0.2$ — clip 范围

**Clip 的作用**：当 ratio 偏离 1 太远时，截断梯度。防止一步更新太大导致策略崩溃。

加上 KL 惩罚防止偏离参考模型太远：

$$L = L^{PPO} + \beta \cdot KL(\pi_\theta \| \pi_{ref})$$

## DPO：去掉 Reward Model

### 核心洞察

DPO 的数学推导发现：**最优 policy 和 reward model 之间有闭式关系**：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + C$$

所以不需要单独训练 reward model——直接用偏好对 (chosen, rejected) 优化 policy。

### 训练流程

```
           ┌──────────────┐
           │ Policy (LLM) │ ──→ log π_θ(y_w|x), log π_θ(y_l|x)
           └──────────────┘
                  │
           ┌──────────────┐
           │  Ref Policy  │ ──→ log π_ref(y_w|x), log π_ref(y_l|x)
           └──────────────┘
                  │
                  ▼
           DPO Loss (一个公式搞定)
```

只需要 **2 个模型**：Policy + Reference Policy。

### 核心公式

$$L^{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \cdot \left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$$

其中 $y_w$ 是 chosen (win)，$y_l$ 是 rejected (lose)。

**直觉**：让 policy 相对 ref model，给 chosen 更高的概率、给 rejected 更低的概率。$\beta$ 控制偏离 ref 的程度。

### 隐式 Reward

DPO 训好后，可以从 policy 反推出隐式 reward：

$$\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

## GRPO：去掉 Value Function

### 核心洞察

PPO 的 value function 训练难度大、不稳定，而��对 LLM 场景来说显存开销巨大（又一个和 policy 同等规模的模型）。

GRPO 的思路：**每个 prompt 采样一组（G 个）响应，用组内 reward 的相对排名代替 value baseline**。

### 训练流程

```
           ┌──────────────┐      G 个响应
Prompt ──→ │ Policy (LLM) │ ──→ {y₁, y₂, ..., y_G}
           └──────────────┘
                  │
           ┌──────────────┐
           │ Reward Model │ ──→ {r₁, r₂, ..., r_G}
           └──────────────┘
                  │
                  ▼
        组内归一化: A_i = (r_i - mean) / std
                  │
                  ▼
        PPO-style Clipped Objective
```

需要 **3 个模型**：Policy + Reference Policy + Reward Model（没有 Value Model）。

### 核心公式

$$L^{GRPO} = -\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(r_i A_i, \; \text{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i\right)\right] + \beta \cdot KL$$

Advantage 的计算不需要 Value function：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

就是组内 z-score——同一个 prompt 的 G 个响应中，哪些比平均好、哪些比平均差。

### GRPO 的 KL 惩罚

GRPO 用逐 token 的 KL 散度（而非序列级）：

$$KL = \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log\frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1\right]$$

## 三者对比

| | PPO | DPO | GRPO |
|---|---|---|---|
| **年份** | 2017 | 2023 | 2024 |
| **需要 Reward Model** | 是 | 否 | 是 |
| **需要 Value Function** | 是 | 否 | 否 |
| **显存中的模型数** | 4 | 2 | 3 |
| **训练数据** | prompt → 在线采样 | 离线偏好对 (chosen, rejected) | prompt → 在线采样 G 个 |
| **Advantage 来源** | R - V(s) | 隐式 (chosen vs rejected) | 组内 z-score |
| **在线/离线** | 在线 | 离线 | 在线 |
| **实现复杂度** | 高 | 低 | 中 |
| **代表模型** | ChatGPT, Claude | Llama 2-Chat, Zephyr | DeepSeek-R1 |

### 优缺点

**PPO**
- 优点：理论成熟，效果上限高，能持续在线探索
- 缺点：4 个模型显存巨大，训练不稳定，调参困难，实现复杂

**DPO**
- 优点：简单优雅，只需偏好对数据，2 个模型就够
- 缺点：离线算法，受限于已有数据质量；不能探索新策略；长回答偏好信号弱

**GRPO**
- 优点：去掉 value model 省显存，组内排名更稳定，在线探索
- 缺点：每个 prompt 需要 G 次采样（计算成本），依赖 reward model 质量

### 共识

三种算法有几个共同的核心设计：

1. **KL 约束**：都限制 policy 不要偏离 reference model 太远
2. **相对比较**：都是在比较"更好 vs 更差"，不是绝对评分
3. **Reference Model**：都需要一个冻结的 SFT 模型作为锚点
4. **目标一致**：最终都是让 $\pi_\theta(y_w) > \pi_\theta(y_l)$（偏好的回答概率更高）

## 代码

三个最小可运行示例：
- `ppo.py` — PPO with clipped surrogate + value function
- `dpo.py` — DPO with preference pairs
- `grpo.py` — GRPO with group relative ranking

## 动画演示

> 打开 `animation.html` 查看交互式公式对比和流程图。

## 答案与总结

| 要点 | 结论 |
|------|------|
| PPO | 经典 RL，4 个模型，效果好但工程复杂 |
| DPO | 去掉 RM + Value，只需偏好对，简单但受限于离线数据 |
| GRPO | 去掉 Value，组内排名当 advantage，DeepSeek 的实用选择 |
| 共识 | 都要 KL 约束 + 相对比较 + reference model |

**一句话总结**：PPO 是"完整版强化学习"，DPO 是"不要 RL 直接优化"，GRPO 是"RL 但更省"。
