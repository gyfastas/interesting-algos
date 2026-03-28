# 手写多头注意力机制：MHA vs GQA vs MLA

## 问题描述

面试高频题：**手写一个 Multi-Head Attention 的 forward**。

但光会写 MHA 还不够——现代大模型早已不用标准 MHA 了。Llama 2/3 用的是 **GQA**，DeepSeek-V2/V3 用的是 **MLA**。它们解决的是同一个问题：**推理时 KV Cache 太大了**。

本题从最简 MHA 出发，讲清楚三种注意力机制的核心区别和演进逻辑。

## 直觉分析

先回顾标准 MHA 的计算流程：

```
输入 x: (B, S, D)
         │
    ┌─────┼─────┐
    W_q   W_k   W_v       ← 三个独立的线性投影
    │     │     │
    Q     K     V          ← 各 (B, S, D)
    │     │     │
   拆成 n_heads 个头      ← 各 (B, H, S, d_k), d_k = D/H
    │     │     │
    └──┬──┘     │
   QK^T/√d_k   │          ← 注意力分数 (B, H, S, S)
       │        │
    softmax     │
       │        │
       └───×────┘          ← 加权求和
           │
       concat heads        ← (B, S, D)
           │
          W_o              ← 输出投影
           │
       输出: (B, S, D)
```

**问题在哪？** 自回归推理时，每生成一个 token，都要用到之前所有 token 的 K 和 V。这些 K/V 必须缓存在显存里（KV Cache），而且**每一层、每一个头**都要单独存。

以一个 32 层、32 头、d_model=4096 的模型为例，序列长度 4096 时：

$$\text{MHA KV Cache} = 2 \times L \times H \times d_k \times S \times 2\text{B} = 2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 2.15\text{ GB}$$

这还只是**一个请求**的缓存。高并发时，KV Cache 往往比模型参数本身还占显存。

## 数学建模

### 1. 标准 MHA

给定输入 $X \in \mathbb{R}^{S \times D}$：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$。

将 Q, K, V 拆分为 $H$ 个头，每个头维度 $d_k = D/H$：

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

**KV Cache 大小**：每层缓存 K 和 V，各 $H \times d_k = D$ 维，共 $2D$ per token per layer。

### 2. GQA (Grouped Query Attention)

GQA 的核心改动：**Q 还是 $H$ 个头，但 K/V 只有 $H_{kv}$ 个头**（$H_{kv} < H$）。

$$W_K \in \mathbb{R}^{D \times (H_{kv} \cdot d_k)}, \quad W_V \in \mathbb{R}^{D \times (H_{kv} \cdot d_k)}$$

每 $H / H_{kv}$ 个 Q 头共享同一组 KV 头。计算时将 KV 头 repeat 到和 Q 对齐：

```python
K = K.repeat_interleave(n_rep, dim=1)  # (B, H_kv, S, d_k) -> (B, H, S, d_k)
```

**特殊情况**：
- $H_{kv} = H$ → 标准 MHA
- $H_{kv} = 1$ → MQA (Multi-Query Attention)

**KV Cache 节省比例**：$H_{kv} / H$。Llama 2 70B 用 8 个 KV 头 / 64 个 Q 头，节省 87.5%。

### 3. MLA (Multi-Head Latent Attention)

MLA 来自 DeepSeek-V2 论文，思路完全不同：**不是减少 KV 头数，而是把 KV 压缩到一个低维 latent 向量**。

KV 联合压缩：

$$c^{KV}_t = W_{DKV} \cdot x_t \in \mathbb{R}^{d_c}$$

$$K_t = W_{UK} \cdot c^{KV}_t, \quad V_t = W_{UV} \cdot c^{KV}_t$$

其中 $d_c \ll 2 H d_k$。推理时，**只需缓存 $c^{KV}_t$**，不需要存完整的 K 和 V。

Q 侧同样做低秩压缩（减少训练时的激活内存）：

$$c^Q_t = W_{DQ} \cdot x_t, \quad Q_t = W_{UQ} \cdot c^Q_t$$

**KV Cache 大小**：每个 token 每层只存 $d_c$ 维（加上解耦 RoPE 的 $d_{rope}$ 维），而不是 $2Hd_k$ 维。

## 三种机制对比

| | MHA | GQA | MLA |
|---|---|---|---|
| **Q 头数** | $H$ | $H$ | $H$ |
| **KV 头数** | $H$ | $H_{kv}$ (< $H$) | $H$（从 latent 恢复） |
| **KV Cache / token / layer** | $2Hd_k$ | $2H_{kv}d_k$ | $d_c$ (+$d_{rope}$) |
| **压缩方式** | 无 | 减少头数 | 低秩压缩 |
| **代表模型** | GPT-2, BERT | Llama 2/3, Gemma | DeepSeek-V2/V3 |
| **质量损失** | baseline | 轻微 | 极小 |

### KV Cache 数值对比

以 32 层、32 头、d_model=4096、seq_len=4096、fp16 为例：

| 方案 | KV Cache | 占 MHA 比例 |
|------|----------|------------|
| MHA (32 KV heads) | 2.15 GB | 100% |
| GQA (8 KV heads) | 0.54 GB | 25% |
| MQA (1 KV head) | 0.07 GB | 3.1% |
| MLA (d_c=512) | 0.13 GB | 6.2% |

## 代码实现

### MHA — 最简 forward

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # 默认 causal mask（decoder-only 标配）
        if mask is None:
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

### GQA — 关键改动：KV 头数更少 + repeat

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_rep = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)   # 更小
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)   # 更小
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 核心：KV 头 repeat 对齐到 Q 头数
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is None:
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

### MLA — 关键改动：KV 联合低秩压缩

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_c):
        super().__init__()
        self.d_k = d_model // n_heads
        self.W_dq  = nn.Linear(d_model, d_c, bias=False)              # Q 下投影
        self.W_uq  = nn.Linear(d_c, n_heads * self.d_k, bias=False)   # Q 上投影
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)              # KV 联合下投影
        self.W_uk  = nn.Linear(d_c, n_heads * self.d_k, bias=False)   # K 上投影
        self.W_uv  = nn.Linear(d_c, n_heads * self.d_k, bias=False)   # V 上投影
        self.W_o   = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        c_q  = self.W_dq(x)                    # (B, S, d_c)
        Q    = self.W_uq(c_q).view(...)        # -> (B, H, S, d_k)

        c_kv = self.W_dkv(x)                   # (B, S, d_c) ← 只缓存这个！
        K    = self.W_uk(c_kv).view(...)        # -> (B, H, S, d_k)
        V    = self.W_uv(c_kv).view(...)        # -> (B, H, S, d_k)

        # 后面和 MHA 完全一样
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

## MLA 的好处

**1. KV Cache 大幅压缩**

MLA 的 KV Cache 只需存 $c^{KV}_t$（维度 $d_c$），而不是完整的 K 和 V（维度 $2Hd_k$）。DeepSeek-V2 中 $d_c = 512$，而 $2Hd_k = 2 \times 128 \times 128 = 32768$，**压缩比超过 98%**。

**2. 质量几乎不损失**

和 GQA 不同（直接砍头数，信息容量降低），MLA 保留了全部 $H$ 个注意力头。低秩压缩利用的是 KV 表征中的冗余——不同头的 K/V 之间存在大量线性相关性，一个 $d_c$ 维的 latent 向量就能恢复出完整的多头 KV。

**3. 推理时矩阵吸收 (Weight Absorption)**

推理时可以把 $W_{UK}$ 和 $W_{UQ}$ 合并到注意力计算中，直接用 $c^Q$ 和 $c^{KV}$ 算注意力分数，避免显式恢复 K。这意味着推理 FLOPs 也能降低。

**4. 训练时减少激活内存**

Q 侧也做低秩压缩，反向传播时只需存 $c^Q$（$d_c$ 维）而非完整 Q（$Hd_k$ 维），降低训练的峰值显存。

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化三种注意力机制的数据流和 KV Cache 对比。

## 答案与总结

| 要点 | 结论 |
|------|------|
| MHA 的问题 | KV Cache 随层数×头数×序列长度线性增长，推理显存瓶颈 |
| GQA 的思路 | 减少 KV 头数，多个 Q 头共享一组 KV，简单有效 |
| MLA 的思路 | KV 联合低秩压缩到 latent 向量，推理只缓存 latent |
| MLA 的优势 | 压缩比更高（>98%）、保留全部注意力头、可做矩阵吸收加速推理 |

**一句话总结**：GQA 是"少存几份 KV"，MLA 是"把 KV 压缩了再存"——后者在同等缓存预算下能保留更多信息。
