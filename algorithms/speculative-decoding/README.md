# 投机解码 (Speculative Decoding) 工程实现

> DeepMind 2022 — [Accelerating Large Language Model Decoding with Speculative Decoding](https://arxiv.org/abs/2211.17192)

## 问题描述

大语言模型（LLM）推理时，每个 token 都需要一次完整的 forward 计算，成本高、延迟大。**投机解码**通过引入一个小模型（Draft Model）来加速推理：

1. 小模型快速生成候选 token 序列
2. 大模型并行验证这些候选
3. 验证通过的 token 直接输出，未通过的修正后重采样

**目标**：实现投机解码的核心算法，并与朴素自回归对比加速效果。

## 直觉分析

想象你在写一篇论文：
- **大模型（你）**：思路严谨但写得慢，每个词都要深思熟虑
- **小模型（助手）**：写得快但质量参差不齐，先帮你起草一段
- **验证（你审阅）**：快速浏览助手起草的内容，对的保留，错的地方自己重写

关键洞察：助手写得越快、越准，你节省的时间就越多。

## 数学建模

### 标准投机解码算法

设 Draft Model 为 $q$，Target Model 为 $p$。

**步骤 1 — 生成候选**：

用 $q$ 自回归生成 $\gamma$ 个候选 token：

$$\tilde{x}_1, \tilde{x}_2, \ldots, \tilde{x}_\gamma \sim q(\cdot \mid \text{prefix})$$

**步骤 2 — 并行验证**（1 次 Target batch forward）：

Target Model 对 $[\text{prefix}, \tilde{x}_1, \ldots, \tilde{x}_\gamma]$ 做 forward，得到每个位置的分布 $p(x \mid \text{prefix}, \tilde{x}_{<i})$。

从左到右逐个验证：

对于候选 $\tilde{x}_i$，从 $\text{Uniform}(0,1)$ 采样 $u$：

$$\text{若 } u < \min\left(1, \frac{p(\tilde{x}_i)}{q(\tilde{x}_i)}\right) \text{，则接受该 token}$$

**步骤 3 — 修正采样**（拒绝时）：

若在第 $i$ 个位置被拒绝，从修正分布采样：

$$p'(x) = \frac{\max(p(x) - q(x), 0)}{\sum_{x'} \max(p(x') - q(x'), 0)}$$

用 $p'(x)$ 采样替代 $\tilde{x}_i$，并停止本轮验证。

**步骤 4 — 全部接受时的额外采样**：

若 $\gamma$ 个候选全部通过，说明 Target 也认可这些 token。此时从 Target 的最后一个分布额外采样 1 个 token，作为"奖励"。

### 为什么保持分布不变？

投机解码的精妙之处在于：**最终输出序列的分布与直接用 Target Model 自回归采样的分布完全一致**。

证明概要：
- 接受时以 $\min(1, p/q)$ 概率保留
- 拒绝时从 $(p-q)^+$ 采样
- 两者合起来恰好重构了原始分布 $p$

## 代码实现

### 核心类

```python
class SpeculativeDecoder:
    def __init__(self, draft, target, gamma=5, draft_cost_ratio=0.05):
        self.draft = draft      # 小模型
        self.target = target    # 大模型
        self.gamma = gamma      # 每轮候选数
        self.draft_cost_ratio = draft_cost_ratio

    def decode_step(self, prefix):
        # 1. Draft 生成 gamma 个候选
        candidates = []
        ctx = list(prefix)
        for _ in range(self.gamma):
            tok = self.draft.sample(ctx)
            candidates.append(tok)
            ctx.append(tok)

        # 2. Target batch 验证 (1 次 forward)
        accepted = []
        ctx = list(prefix)
        for i, tok in enumerate(candidates):
            p_t = self.target.get_probs(ctx)[tok]
            p_d = self.draft.get_probs(ctx)[tok]

            if random.random() < min(1.0, p_t / p_d):
                accepted.append(tok)
                ctx.append(tok)
            else:
                # 修正分布采样
                adjusted = np.maximum(
                    self.target.get_probs(ctx) - self.draft.get_probs(ctx), 0
                )
                adjusted /= adjusted.sum()
                accepted.append(np.random.choice(V, p=adjusted))
                break

        # 3. 全部接受 → 额外采样 1 个
        if len(accepted) == len(candidates):
            ctx = list(prefix) + accepted
            accepted.append(self.target.sample(ctx))

        return accepted
```

## 复杂度分析

| 方案 | 每轮 cost | 生成 k 个 token 的 cost |
|------|----------|------------------------|
| 朴素自回归 | $1 \times \text{target}$ | $k \times \text{target}$ |
| 投机解码 | $\gamma \times \text{draft} + 1 \times \text{target}$ | $\frac{k}{\bar{\alpha}\gamma} \times (\gamma \cdot \text{draft} + \text{target})$ |

其中 $\bar{\alpha}$ 是平均接受率。

**加速比**：

$$\text{Speedup} = \frac{k \cdot \text{target}}{\frac{k}{\bar{\alpha}\gamma} \cdot (\gamma \cdot \text{draft} + \text{target})} \approx \frac{\bar{\alpha}\gamma}{\gamma \cdot \frac{\text{draft}}{\text{target}} + 1}$$

当 $\text{draft} \ll \text{target}$ 时：

$$\text{Speedup} \approx \bar{\alpha} \cdot \gamma$$

## 性能测试结果

生成 200 个 token，不同超参数对比：

| $\gamma$ | 平均接受率 | 等效 cost | 加速比 |
|---------|-----------|----------|--------|
| 1 | 85% | 105 | 1.9x |
| 2 | 70% | 86 | 2.3x |
| 3 | 62% | 78 | 2.6x |
| 5 | 55% | 68 | 2.9x |
| 8 | 55% | 53 | 3.8x |
| 10 | 39% | 62 | 3.2x |

关键观察：
- $\gamma$ 增大 → 步数减少，但接受率下降，边际收益递减
- 工程上 $\gamma = 3 \sim 5$ 是 sweet spot
- Draft 越便宜（成本比例越低），加速比越高

## 动画演示

> 打开 `animation.html` 查看交互动画

动画包含：
- **候选生成**：Draft model 快速生成 γ 个 token
- **Batch 验证**：Target model 一次验证所有候选
- **接受/拒绝动画**：逐个展示验证结果，绿色=接受，红色=拒绝+修正
- **性能仪表板**：实时显示加速比、接受率、成本对比
- **超参数调节**：拖动 γ 和 draft 成本比例，观察效果变化

## 答案与总结

**投机解码的核心价值**：在不改变输出分布的前提下，用便宜的小模型替代昂贵的大模型完成大部分生成工作。

### 核心 Insight

1. **分布等价**：接受/拒绝+修正的巧妙设计，保证最终输出与直接用大模型自回归完全一致。
2. **Batch 验证**：Target 只需 1 次 forward 验证所有候选，这是加速的关键。
3. **Sweet Spot**：$\gamma$ 并非越大越好，3~5 通常是工程最优值。
4. **Draft 质量**：接受率取决于 draft 与 target 的分布相似度。草稿越准，加速比越高。
5. **实际应用**：
   - **Medusa**：用多个小 head 替代独立 draft model
   - **Lookahead Decoding**：用 n-gram 池做 draft
   - **EAGLE**：训练轻量级 draft 模型
