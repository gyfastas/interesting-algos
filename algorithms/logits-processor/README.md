# 037 Logits 采样处理器 — Temperature / Top-k / Top-p

**标签:** `NumPy` · `LLM 解码` · `采样策略` · `工程实现`

## 问题

给定 shape 为 `(batch_size, vocab_size)` 的 logits，纯用 NumPy 实现以下处理流水线：

1. **Temperature Scaling**: `logits /= temperature`
2. **Top-k**: 每行只保留概率最高的 k 个 token
3. **Top-p (Nucleus Sampling)**: 按概率从高到低累积，保留累积概率 ≤ p 的 token

**关键要求：** Top-p 后如果某 batch 行过滤后全为 `-inf`（概率全零），要**自动恢复该行原来概率最高的 token**。

## 核心思想

### 处理顺序

```
原始 logits
    ↓
Temperature Scaling (logits /= T)
    ↓
Top-k 截断 (保留前 k 个)
    ↓
Top-p 截断 (Nucleus Sampling)
    ↓
Softmax → 采样
```

### Temperature

- `T > 1`: 分布平坦，增加随机性（创造性更强）
- `T < 1`: 分布尖锐，减少随机性（更确定性）
- `T → 0`: 退化为贪心解码（argmax）

### Top-k

每行独立取前 k 大，其余置 `-inf`。使用 `np.partition` 实现 **O(vocab_size)** 的 batch 向量化，无需 Python 循环。

### Top-p (Nucleus Sampling)

1. 对概率降序排序
2. 计算累积概率
3. 找到第一个 `cumsum > p` 的位置作为截断点
4. **防御性保底**: `cutoff = max(cutoff, 1)`，确保至少保留 1 个
5. **后处理恢复**: 若极端情况下某行全 `-inf`，自动恢复该行概率最高的 token

## 复杂度分析

| 操作 | 时间 | 空间 | 实现要点 |
|------|------|------|----------|
| Temperature | O(B×V) | O(B×V) | 逐元素除法 |
| Top-k | O(B×V) | O(B×V) | `np.partition` |
| Top-p | O(B×V log V) | O(B×V) | `np.sort` + `np.cumsum` |

B = batch_size, V = vocab_size

## 关键代码

```python
def apply_top_k(logits, k):
    kth_val = np.partition(logits, -k, axis=-1)[..., -k, np.newaxis]
    mask = logits >= kth_val
    return np.where(mask, logits, -np.inf)

def apply_top_p(logits, p):
    probs = softmax(logits)
    sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
    sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
    cumsum = np.cumsum(sorted_probs, axis=-1)
    remove_mask = cumsum > p

    for b in range(batch_size):
        removed = np.where(remove_mask[b])[0]
        cutoff = removed[0] if len(removed) > 0 else vocab_size
        cutoff = max(cutoff, 1)  # 保底至少 1 个
        keep_indices = sorted_indices[b, :cutoff]
        keep[b, keep_indices] = True

    # 极端 fallback：全零时恢复概率最高的 token
    all_zero = ~np.any(keep, axis=-1)
    if np.any(all_zero):
        max_idx = np.argmax(probs, axis=-1)
        keep[all_zero, max_idx[all_zero]] = True

    return np.where(keep, logits, -np.inf)
```

## 面试要点

1. **为什么处理顺序是 Temperature → Top-k → Top-p？**
   - Temperature 先改变分布形状；Top-k 限制候选集大小；Top-p 在候选集中做累积截断。三者配合覆盖从"严格贪心"到"完全随机"的完整 spectrum。

2. **Top-k 用 `np.partition` 而不是 `np.sort` 的好处？**
   - `np.partition` 是 O(n) 平均复杂度，`np.sort` 是 O(n log n)。Top-k 只需要第 k 大的阈值，不需要完全排序。

3. **Top-p 的 fallback 在什么场景会触发？**
   - 理论上 `cutoff = max(cutoff, 1)` 已避免空集，但数值精度问题（如 `cumsum[0] > p` 且 p 极小时）或未来代码修改可能引入 bug。防御性编程确保 LLM 推理永不崩溃。

4. **batch 向量化 vs 逐行 for 循环？**
   - Top-k 可完全向量化；Top-p 因每行截断点不同，需一个轻量级的 `for b in range(batch_size)` 循环构建 keep mask，但排序和 cumsum 仍是 batch 向量化。

## 运行

```bash
python algorithms/logits-processor/solution.py
```
