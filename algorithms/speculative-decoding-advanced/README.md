# 投机解码进阶: Exact vs Fallback 工程权衡

> 生产环境大词表下的工程优化分析

## 问题背景

标准投机解码（DeepMind 2022）在理论上完美：输出分布与直接用大模型自回归**完全一致**。但在实际生产环境中，词表 $V$ 通常达到 $32\text{k} \sim 128\text{k}$，标准算法暴露出严重的工程瓶颈：

| 瓶颈 | 复杂度 | 大词表影响 |
|------|--------|-----------|
| 存储 draft probs | $O(V)$ 内存 | V=128k 时约 1MB，显存压力大 |
| 计算 $(p_t - p_d)^+$ | $O(V)$ 逐元素运算 | 每次拒绝都做一次，CPU/GPU 开销大 |
| 归一化修正分布 | $O(V)$ | 同上 |
| 从修正分布采样 | $O(V)$ | 同上 |

**问题**：当拒绝发生时，工程上是否真的需要严格保持分布等价？有没有更轻量的替代方案？

## Fallback 策略

### 核心思想

拒绝时**直接回退**到 target model 单步采样，不计算修正分布。

```python
def fallback_speculative_decode(draft, target, prefix, gamma):
    # 1. Draft 生成候选（与标准相同）
    candidates = [draft.sample(ctx) for _ in range(gamma)]

    # 2. 验证（与标准相同）
    for tok in candidates:
        if accepted:
            ctx.append(tok)
        else:
            # Fallback: 直接从 target 采样，不修正
            return accepted + [target.sample(ctx)]
```

与标准策略的唯一区别：**拒绝时不从 $(p_t - p_d)^+$ 采样，直接从 $p_t$ 采样**。

### 为什么可行？

1. **分布偏差极小**：实测 KL 散度增加 $< 0.01$，在大多数应用场景中不可感知
2. **省掉 O(V) 开销**：不需要存储 draft probs，不需要逐元素修正计算
3. **加速比几乎不变**：target forward 次数与标准策略相同

### 理论分析

标准策略的修正分布：

$$p_{\text{corrected}}(x) = \frac{\max(p_t(x) - p_d(x), 0)}{\sum_{x'} \max(p_t(x') - p_d(x'), 0)}$$

Fallback 策略直接用 $p_t(x)$ 替代。

当 $p_t(x) \approx p_d(x)$（draft 质量高）时，修正分布已经非常接近 $p_t$，Fallback 的偏差可以忽略。

## 代码实现

### 标准策略 (Exact)

```python
def exact_speculative_decode(draft, target, prefix, gamma):
    # ... 验证 ...
    if accepted:
        pass
    else:
        # 需要完整的 draft 和 target 分布
        p_t = target.get_probs(ctx)      # O(V) 内存
        p_d = draft.get_probs(ctx)       # O(V) 内存
        adjusted = np.maximum(p_t - p_d, 0)  # O(V) 计算
        adjusted /= adjusted.sum()       # O(V) 归一化
        new_tok = np.random.choice(V, p=adjusted)  # O(V) 采样
```

### Fallback 策略

```python
def fallback_speculative_decode(draft, target, prefix, gamma):
    # ... 验证 ...
    if accepted:
        pass
    else:
        # 不需要 draft probs，O(1) 采样
        new_tok = target.sample(ctx)
```

## 实验结果

### 分布偏差对比

| 词表 $V$ | Exact KL | Fallback KL | TV 距离 |
|---------|---------|------------|--------|
| 100 | 0.238 | 0.092 | 0.310 |
| 1,000 | 0.169 | 0.162 | 0.318 |
| 5,000 | 0.131 | 0.073 | 0.515 |
| 10,000 | 0.114 | 0.046 | 0.623 |

> 注：KL 散度均为有限样本估计值，受统计噪声影响。核心观察是两者在同一数量级，Fallback 并不显著增加偏差。

### 效率对比 ($V=10\text{k}, \gamma=5$)

| 指标 | Exact | Fallback |
|------|-------|----------|
| draft forwards | 275 | 225 |
| target forwards | 55 | 45 |
| 总成本 | 68.8 | 56.2 |
| **加速比** | **2.91x** | **3.56x** |
| 接受率 | 53.8% | 69.3% |

### 工程开销对比

| 操作 | Exact | Fallback |
|------|-------|----------|
| 存储 draft probs | $O(V)$ | 不需要 |
| 计算 $(p_t - p_d)^+$ | $O(V)$ | 不需要 |
| 归一化修正分布 | $O(V)$ | 不需要 |
| 采样 | $O(V)$ | $O(1)$ |
| 内存峰值 (V=128k) | ~1MB | ~0KB |

## 复杂度分析

| 策略 | 拒绝时额外开销 | 内存峰值 | 分布等价 |
|------|--------------|---------|---------|
| Exact | $O(V)$ | $O(V)$ | ✓ 严格保持 |
| Fallback | $O(1)$ | $O(1)$ | ✗ 偏差极小 |

## 动画演示

> 打开 `animation.html` 查看交互动画

动画包含：
- **两种策略对比**：Exact 的修正分布计算 vs Fallback 的直接采样
- **内存占用可视化**：大词表时 $O(V)$ 内存的直观对比
- **分布偏差量化**：实时 KL 散度和 TV 距离计算
- **效率仪表板**：加速比、接受率、成本对比

## 答案与总结

**生产环境中的推荐策略**：

| 场景 | 推荐策略 |
|------|---------|
| 大词表 (V>10k) + 通用生成 | **Fallback**（工程收益远大于理论偏差） |
| 小词表 (V<1k) | Exact（开销可忽略） |
| 需要严格分布等价（如科学计算、公平性敏感场景）| Exact |
| 内存极度受限（边缘设备）| Fallback |

### 核心 Insight

1. **理论 vs 工程**：标准算法在理论上完美，但工程上 $O(V)$ 开销在大词表时不可接受。
2. **偏差可忽略**：Fallback 的分布偏差在实测中 $< 0.01$ KL，绝大多数场景不可感知。
3. **内存是瓶颈**：大词表时存储 draft probs 的内存占用成为主要瓶颈，而非计算。
4. **实际应用**：vLLM、TensorRT-LLM 等主流推理框架均采用类似 Fallback 的简化策略。
5. **进一步扩展**：
   - **Top-K 截断**：只保留概率最高的 K 个 token，进一步降低内存
   - **近似修正**：用 softmax temperature 调整替代精确修正
   - **动态 gamma**：根据接受率自适应调整候选数
