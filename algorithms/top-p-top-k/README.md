# Top-p & Top-k 采样：LLM 解码策略

## 问题描述

**LLM 生成文本时，如何从 logits 中选下一个 token？手写 Top-k、Top-p (nucleus)、Temperature 采样。**

Greedy decoding（每次选概率最大的 token）会导致重复、无聊的输出。采样策略让模型在"确定性"和"创造性"之间取得平衡。

## 直觉分析

### 为什么不直接 argmax？

```
logits = [猫: 5.2, 狗: 4.8, 鱼: 3.1, 桌子: 0.5, 量子: 0.01, ...]

Greedy:  永远选「猫」→ 重复、死板
采样:    按概率随机选 → 可能选到「量子」→ 胡说八道
Top-k:   只从前 k 个候选中采样 → 排除长尾噪声
Top-p:   只从累积概率达到 p 的候选中采样 → 自适应候选集
```

### 三种策略的直觉

| 策略 | 直觉 | 类比 |
|------|------|------|
| **Temperature** | 调节分布的"锐度" | 调音量旋钮 |
| **Top-k** | 只看前 k 名 | 只考虑排名前 k 的候选人 |
| **Top-p** | 只看累积概率达到 p 的 | 只考虑"主流"选项，忽略极小概率的 |

## Temperature 采样

### 原理

Temperature 不改变候选集，而是改变概率分布的形状：

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- $T = 1$：标准 softmax，正常分布
- $T → 0$：分布趋向 one-hot（argmax），确定性输出
- $T > 1$：分布变平，增加随机性
- $T → ∞$：均匀分布，完全随机

### 为什么有效

Temperature 本质是对 logits 做缩放：$z_i / T$。

- $T < 1$：放大 logits 差异 → softmax 后高概率更高、低概率更低
- $T > 1$：���小 logits 差异 → softmax 后分布更均匀

```
logits = [5.0, 4.0, 1.0]

T=0.5: softmax([10, 8, 2])   = [0.88, 0.12, 0.00]  ← 很确定
T=1.0: softmax([5, 4, 1])    = [0.71, 0.26, 0.03]  ← 正常
T=2.0: softmax([2.5, 2, 0.5]) = [0.47, 0.38, 0.15]  ← 很随机
```

## Top-k 采样

### 算法

```
Step 1: 对 logits 排序，取前 k 个
Step 2: 把其余 token 的 logits 设为 -∞
Step 3: 在前 k 个 token 上做 softmax + 采样
```

### 优点与缺点

- **优点**：简单有效，截断长尾噪声
- **缺点**：k 是固定的。当分布很尖锐时（模型很确定），k=50 也包含太多无意义 token；当分布很平坦时，k=50 可能不够

```
场景 1 (模型很确定):
  probs = [0.9, 0.05, 0.02, 0.01, ...]
  Top-50 包含 47 个几乎 0 概率的 token → 浪费

场景 2 (模型不确定):
  probs = [0.05, 0.04, 0.04, 0.03, ...]  ← 很多 token 都合理
  Top-50 可能排除了一些合理候选 → 截断过多
```

## Top-p (Nucleus) 采样

### 算法

```
Step 1: 对 logits 做 softmax 得到概率
Step 2: 按概率从大到小排序
Step 3: 计算累积概率 (cumsum)
Step 4: 找到累积概率 ≥ p 的最小集合
Step 5: 在这个集合中重新归一化 + 采样
```

### 为什么比 Top-k 好

Top-p 的候选集大小是**自适应的**：

```
场景 1 (p=0.9, 模型很确定):
  probs = [0.9, 0.05, 0.02, ...]
  cumsum = [0.9, 0.95, ...]
  → 只需 1 个 token 就达到 p=0.9 → 候选集很小

场景 2 (p=0.9, 模型不确定):
  probs = [0.1, 0.08, 0.07, 0.06, ...]
  cumsum = [0.1, 0.18, 0.25, 0.31, ...]
  → 需要很多 token 才达到 0.9 → 候选集很大
```

**核心优势：候选集大小随模型的"确定程度"自动调节。**

## 组合使用

实际推理中，这些策略通���组合使用：

```python
# 典型配置
logits = model(input_ids)

# Step 1: Temperature 调节分布形状
logits = logits / temperature      # 先缩放

# Step 2: Top-k 粗筛
# 只保留前 k 个

# Step 3: Top-p 精筛
# 在 Top-k 结果上再做 nucleus sampling

# Step 4: 采样
token = sample(logits)
```

顺序很重要：**Temperature → Top-k → Top-p → Sample**。

## 核心代码

```python
def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    完整的采样流程: Temperature → Top-k → Top-p → Sample
    """
    # 1. Temperature
    if temperature != 1.0:
        logits = logits / temperature

    # 2. Top-k: 只保留前 k 个
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # 3. Top-p: 累积概率截断
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # 移除累积概率超过 p 的 token (保留第一个超过的)
        remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        # 还原到原始顺序
        logits = sorted_logits.scatter(-1, sorted_idx.argsort(-1), sorted_logits)

    # 4. 采样
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## 各策略参数的典型值

| 参数 | 含义 | 典型范围 | 常用值 |
|------|------|----------|--------|
| `temperature` | 分布锐度 | 0.0 ~ 2.0 | 0.7 ~ 1.0 |
| `top_k` | 候选集大小 | 0 ~ 100 | 50 |
| `top_p` | 累积概率阈值 | 0.0 ~ 1.0 | 0.9 ~ 0.95 |

不同任务的推荐配置：

| 任务 | temperature | top_k | top_p | 原因 |
|------|------------|-------|-------|------|
| 代码生成 | 0.0 ~ 0.2 | — | — | 需要确定性，不要创造性 |
| 翻译 | 0.3 ~ 0.5 | — | 0.9 | 准确性优先 |
| 对话 | 0.7 ~ 1.0 | 50 | 0.9 | 自然且多样 |
| 创意写作 | 1.0 ~ 1.5 | 100 | 0.95 | 最大化创造性 |

## Repetition Penalty

采样之外，还有一个重要的技巧：重复惩罚。

```python
# 对已生成的 token 降低 logits
for token_id in generated_ids:
    if logits[token_id] > 0:
        logits[token_id] /= repetition_penalty  # > 1.0
    else:
        logits[token_id] *= repetition_penalty
```

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化 Temperature / Top-k / Top-p 对概率分布的影响。

## 答案与总结

| 要点 | 结论 |
|------|------|
| Temperature | 缩放 logits，控制分布的"锐度" |
| Top-k | 固定保留前 k 个 token，简单但不自适应 |
| Top-p | 保留累积概率达到 p 的最小集合，自适应候选集大小 |
| 组合使用 | Temperature → Top-k → Top-p → Sample |
| 核心优势 | Top-p 比 Top-k 更好，因为候选集大小随模型确定程度自动调节 |

**一句话总结**：Temperature 调分布形状，Top-k 砍固定数量的尾巴，Top-p 按累积概率自适应砍尾巴——实际推理三者组合使用。
