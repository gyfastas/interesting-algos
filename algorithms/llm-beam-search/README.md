# LLM 生成中的 Beam Search

**类型：** 搜索算法 · 解码策略
**难度：** ⭐⭐
**关键结论：** Greedy 可能比全局最优差 **30%+**，Beam k=2 往往足够

---

## 问题背景

大语言模型（LLM）在生成文本时是**自回归**的：每次生成一个 token，条件概率为：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

我们的目标是找到联合概率最大的完整序列：

$$\hat{x} = \arg\max_{x} \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

**难点：** 词表大小 $|V|$ 通常达 3 万～10 万，逐步展开所有可能性是指数级的 $O(|V|^T)$。

---

## 为什么 Greedy 会失败？

Greedy（贪心）搜索每步只选当前条件概率最高的 token，**看不到未来**。

用一个具体例子说明：

```
起始 <BOS>
         ├─ "The"  (P=0.52) ← Greedy 选这里
         │    ├─ "small" (P=0.60)
         │    │    └─ "mouse" (P=0.90) → 联合概率 0.52×0.60×0.90 = 0.2808
         │    └─ "big"   (P=0.40)
         │         └─ "house" (P=0.80) → 联合概率 0.52×0.40×0.80 = 0.1664
         │
         └─ "A"    (P=0.48) ← Greedy 没选，但这里藏着最优路径！
              ├─ "quick"     (P=0.15)
              │    └─ "fox"  (P=0.95) → 联合概率 0.48×0.15×0.95 = 0.0684
              └─ "legendary" (P=0.85)
                   └─ "hero" (P=0.99) → 联合概率 0.48×0.85×0.99 = 0.4039 ★最优
```

| 方法 | 结果 | 联合概率 |
|------|------|---------|
| Greedy | "The small mouse" | 0.2808 |
| **Beam k=2** | **"A legendary hero"** | **0.4039 ★** |

Greedy 在第一步被 "The"(0.52) 的高局部概率诱导，错过了 "A legendary hero" 这条更优路径。

---

## Beam Search 算法

**核心思想：** 每步同时维护 $k$ 条最优前缀（beam），用空间换全局质量。

```
初始: beams = [("", log_P=0)]

每一步:
  candidates = []
  for (seq, log_P) in beams:
      for (token, p) in LM(seq):
          candidates.append((seq + token, log_P + log(p)))

  beams = top-k(candidates)   # 只保留 k 条最优

直到所有 beam 都以 <EOS> 结束
```

**为什么用 log 求和而非概率相乘？**

连乘多个小概率会导致浮点数下溢。log 概率相加数值更稳定：

$$\log P(x_1, \ldots, x_T) = \sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

### 逐步演示（k=2）

```
Step 0: beams = [("", 0)]

Step 1: 展开 → [The(-0.654), A(-0.734)]
        top-2: [The(-0.654), A(-0.734)]

Step 2: 展开:
        The → small(-1.165), big(-1.570)
        A   → legendary(-1.571), quick(-2.644)
        top-2: [The small(-1.165), A legendary(-1.571)]

Step 3: 展开:
        The small  → mouse(-1.270★greedy), cat(-3.467)
        A legendary → hero(-1.571+(-0.010)=-0.906★), fail(-5.502)
        top-2: [A legendary hero(-0.906), The small mouse(-1.270)]

Step 4: 所有 beam → <EOS>
结果: "A legendary hero" log_P=-0.906, P=0.4039
```

---

## 算法对比

| 算法 | 时间复杂度 | 空间复杂度 | 质量 |
|------|-----------|-----------|------|
| Greedy | $O(V \cdot T)$ | $O(T)$ | 局部最优 |
| Beam Search (k) | $O(k \cdot V \cdot T)$ | $O(k \cdot T)$ | 近似全局最优 |
| Best-First Search | $O(V^T)$ 最差 | $O(V^T)$ | 精确全局最优 |
| 穷举 | $O(V^T)$ | $O(V^T)$ | 精确全局最优 |

**实践中的 beam width 选取：**
- $k=1$：退化为 Greedy
- $k=4\sim12$：翻译、摘要等任务常用
- $k$ 过大：生成结果趋向"安全但无聊"（length penalty 问题）

---

## 代码实现

### Beam Search 核心

```python
def beam_search(lm, beam_width=2, max_steps=20):
    beams = [(0.0, [])]   # (log_prob, tokens)
    finished = []

    for _ in range(max_steps):
        candidates = []
        for log_p, seq in beams:
            for token, prob in lm(seq).items():
                new_log_p = log_p + math.log(prob)
                if token == '<EOS>':
                    finished.append((new_log_p, seq))
                else:
                    candidates.append((new_log_p, seq + [token]))

        # 剪枝：只保留 top-k
        beams = sorted(candidates, key=lambda x: -x[0])[:beam_width]

    return sorted(finished, key=lambda x: -x[0])
```

运行 [`solution.py`](./solution.py) 查看完整实现，包含 Greedy、Beam Search、Best-First 三种算法及对比。

---

## 动画演示

打开 [`animation.html`](./animation.html) 查看：
- 🌳 展开树的逐步动画
- ✂️ 剪枝过程可视化
- 🎚️ 可调节 beam width（k=1~4）

```bash
open algorithms/llm-beam-search/animation.html
```

---

## 实际 LLM 中的注意事项

1. **Length Penalty**：长序列的 log-prob 天然更低，需归一化 $\frac{\log P}{T^\alpha}$
2. **Top-p / Top-k Sampling**：推理时常与 beam search 结合或替代，增加多样性
3. **Diverse Beam Search**：强制 beam 之间有差异，避免输出高度相似的 k 个结果
4. **Early Stopping**：第一个 beam 出现 `<EOS>` 时，可以停止其他 beam 的展开

---

## 总结

Beam Search 是 LLM 推理解码的核心算法之一。它用 $O(k)$ 倍的计算代价，显著提升输出质量。理解它的本质——**在搜索树上维护 k 条最优前缀**——有助于理解和调试各种解码策略。
