# 手写 SwiGLU 激活函数与 FFN

## 问题描述

面试常见：**SwiGLU 是什么？为什么现代大模型都用它替代 ReLU FFN？手写一个 SwiGLU FFN。**

SwiGLU 不只是一个激活函数——它改变了 Transformer FFN 的整个结构，从 2 个矩阵变成 3 个矩阵。理解它需要先搞清楚三个概念：SiLU (Swish)、GLU (门控)、以及它们的组合。

## 直觉分析

### 标准 Transformer FFN 长什么样？

GPT-2 / BERT 的 FFN 非常简单：

```
x → W1 → ReLU/GELU → W2 → output
     ↑ up-project    ↑ down-project
   (D → 4D)        (4D → D)
```

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

两个矩阵，中间一个激活函数。$d_{ff} = 4D$ 是惯例。

### 问题在哪？

ReLU 把所有负值直接归零——**硬截断，信息永久丢失**。GELU 稍好一些，是"软截断"，但本质上还是 element-wise 的开/关决策。

**更根本的问题**：激活函数作用在 $W_1 x$ 的每个元素上，决定"这个维度要不要保留"。但这个决定是**用同一个值既做判断又做输出**——判断和信息传递混在一起了。

### GLU 的解法：判断和输出分开

GLU (Gated Linear Unit) 的核心思想：**用一条路做判断（gate），另一条路传信息（value），然后相乘**。

```
        ┌─ W_gate → σ(·) ─┐
x ──────┤                  ⊙ ──→ output
        └─ W_up   ─────── ┘
```

Gate 路径决定"开/关"，Up 路径传递实际信息。这就是**门控机��**。

### SwiGLU = SiLU + GLU

SwiGLU 把 GLU 中 gate 路径的 sigmoid 换成 SiLU (Swish)：

$$\text{SwiGLU}(x) = \text{SiLU}(W_{gate} x) \odot (W_{up} x)$$

然后再经过一个 down projection 降回原始维度：

$$\text{SwiGLU-FFN}(x) = W_{down} \cdot [\text{SiLU}(W_{gate} x) \odot (W_{up} x)]$$

## 数学公式

### 几个激活函数

**ReLU**：

$$\text{ReLU}(x) = \max(0, x)$$

**Sigmoid**：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**SiLU (Swish)**：

$$\text{SiLU}(x) = x \cdot \sigma(x)$$

SiLU 的性质：
- 非单调：在 $x \approx -1.28$ 处有个小的负值"凹陷"
- 处处可导：不像 ReLU 在 0 处有拐点
- $x \gg 0$ 时 $\approx x$，$x \ll 0$ 时 $\approx 0$，但负值区间不完全归零

**GELU**：

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi$ 是标准正态分布的 CDF。和 SiLU 形状非常接近，但理论动机不同（dropout 的连续近似）。

### GLU 系列

**GLU (原始)**：

$$\text{GLU}(x) = \sigma(W_{gate} x) \odot (W_{up} x)$$

**SwiGLU**：

$$\text{SwiGLU}(x) = \text{SiLU}(W_{gate} x) \odot (W_{up} x)$$

**GeGLU**：

$$\text{GeGLU}(x) = \text{GELU}(W_{gate} x) \odot (W_{up} x)$$

Shazeer (2020) 的论文测试了多种 GLU 变体，SwiGLU 和 GeGLU 效果最好。

### 完整的 SwiGLU FFN

$$\text{FFN}_{SwiGLU}(x) = W_{down} \cdot [\text{SiLU}(W_{gate} \cdot x) \odot (W_{up} \cdot x)]$$

三个权重矩阵：
- $W_{gate} \in \mathbb{R}^{d_{ff} \times D}$ — 门控投影
- $W_{up} \in \mathbb{R}^{d_{ff} \times D}$ — 上投影（信息通路）
- $W_{down} \in \mathbb{R}^{D \times d_{ff}}$ — 下投影

### 参数量对齐

标准 FFN 有 2 个矩阵：$2 \times D \times d_{ff}$

SwiGLU FFN 有 3 个矩阵：$3 \times D \times d_{ff}$

为了保持总参数量一致，SwiGLU 需要缩小 $d_{ff}$：

$$d_{ff}^{SwiGLU} = \frac{2}{3} \times d_{ff}^{std} = \frac{2}{3} \times 4D = \frac{8D}{3}$$

实际实现中会 round 到 256 的倍数（方便 GPU tensor core 对齐）。

## 代码实现

### 标准 FFN（2 矩阵）

```python
class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)       # up
        self.w2 = nn.Linear(d_ff, d_model)        # down
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
```

### SwiGLU FFN（3 矩阵）

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # 门控
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)  # 信息
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # 降维

    def forward(self, x):
        gate = F.silu(self.w_gate(x))   # 门控信号
        up   = self.w_up(x)             # 信息流
        return self.w_down(gate * up)   # 门控 + 降维
```

### Fused 写法（Llama 风格）

实际推理中，gate 和 up 可以合并成一个矩阵，减少一次 kernel launch：

```python
class SwiGLUFFNFused(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_down    = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate_up = self.w_gate_up(x)            # (B, S, 2*d_ff)
        gate, up = gate_up.chunk(2, dim=-1)    # 各 (B, S, d_ff)
        return self.w_down(F.silu(gate) * up)
```

## 为什么 SwiGLU 比 ReLU FFN 好？

### 1. 门控分离了"判断"和"信息"

标准 FFN 中，$\text{ReLU}(W_1 x)$ 用同一个值既决定激活与否，又作为激活值传递。

SwiGLU 中，$W_{gate} x$ 和 $W_{up} x$ 是**两条独立的线性变换**：
- Gate 可以学到"这个特征重要吗？"
- Up 可以学到"这个特征的值是什么？"
- 两者相乘实现**选择性信息传递**

这和 LSTM 的门控思想一脉相承。

### 2. SiLU 比 ReLU 的梯度更好

| 性质 | ReLU | SiLU |
|------|------|------|
| 负区域梯度 | 0（死神经元） | 小但非零 |
| 零点导数 | 不存在 | 0.5 |
| 平滑性 | 不可导（x=0） | 处处可导 |
| 非单调 | 单调 | 非单调（有小负区间） |

SiLU 在负值区域有微小的负输出，这��负梯度能流回来，缓解了"死神经元"问题。

### 3. 实验效果一致更好

Shazeer (2020) 在同等参数量下对比：

```
T5 模型, 同等参数量和训练 tokens:
  ReLU FFN     → perplexity baseline
  GELU FFN     → 略好于 ReLU
  SwiGLU FFN   → 明显好于 GELU（约等于多训练 15% tokens 的效果）
```

### 4. 所有现代大模型都用了

| 模型 | FFN 类型 | 激活函数 |
|------|----------|---------|
| GPT-2, BERT | 标准 FFN (2矩阵) | GELU |
| GPT-3 | 标准 FFN (2矩阵) | GELU |
| PaLM | SwiGLU FFN (3矩阵) | SiLU |
| Llama 1/2/3 | SwiGLU FFN (3矩阵) | SiLU |
| DeepSeek-V2/V3 | SwiGLU FFN (3矩阵) | SiLU |
| Qwen 2.5 | SwiGLU FFN (3矩阵) | SiLU |
| Gemma | GeGLU FFN (3矩阵) | GELU |

## SwiGLU FFN 在 Transformer 中的位置

```
┌──────────────────────────────┐
│      Transformer Block       │
│                              │
│  x ─→ RMSNorm ─→ Attention ─┼─→ + ─→ RMSNorm ─→ SwiGLU FFN ─→ + ─→ out
│       │                      │   ↑                               ↑
│       └──────────────────────┼───┘   └───────────────────────────┘
│            residual          │              residual
└──────────────────────────────┘
```

每个 Transformer Block 有两次 residual + norm：
1. Attention 前：RMSNorm + Self-Attention + residual
2. FFN 前：RMSNorm + SwiGLU FFN + residual

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化激活函数曲线、门控机制和 FFN 结构对比。

## 答案与总结

| 要点 | 结论 |
|------|------|
| SiLU (Swish) | $x \cdot \sigma(x)$，平滑版 ReLU，负区间有微小非零输出 |
| GLU 门控 | 用两条路分别做"判断"和"信息传递"，相乘实现选择性通过 |
| SwiGLU | SiLU + GLU 的组合，FFN 从 2 矩阵变成 3 矩阵 |
| d_ff 调整 | 为保持参数量一致，$d_{ff}$ 缩小到 $\frac{8}{3}D$ |
| 为什么好 | 门控分离判断与信息、SiLU 梯度更好、实验一致优于 GELU FFN |

**一句话总结**：SwiGLU 让 FFN 从"算��了激活一下"变成"一路判断、一路传信息、门控选择性通过"——多了一个矩阵，但效果明显更好。
