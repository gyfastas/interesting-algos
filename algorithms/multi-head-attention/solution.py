"""
手写多头注意力机制 (MHA) + GQA / MLA 对比

重点：类定义和 forward 写清楚，用 PyTorch 实现。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 标准 Multi-Head Attention (MHA)
# ============================================================
class MultiHeadAttention(nn.Module):
    """标准多头注意力，最简实现。"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (B, S, D)
        B, S, _ = x.shape

        # 线性投影 + 拆分多头: (B, S, D) -> (B, n_heads, S, d_k)
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, H, S, S)

        # 默认使用 causal mask（下三角），decoder-only 标配
        if mask is None:
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        # 加权求和 + 合并多头
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)  # (B, S, D)
        return self.W_o(out)


# ============================================================
# 2. Grouped Query Attention (GQA)
#    - n_kv_heads 组 KV，每组被 n_heads/n_kv_heads 个 Q 头共享
#    - n_kv_heads=1 退化为 MQA, n_kv_heads=n_heads 退化为 MHA
# ============================================================
class GroupedQueryAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 每个 KV 头被几个 Q 头共享
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)     # 比 MHA 小
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)     # 比 MHA 小
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 关键：将 KV 头复制 n_rep 次，对齐到 Q 的头数
        # (B, n_kv_heads, S, d_k) -> (B, n_heads, S, d_k)
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is None:
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)


# ============================================================
# 3. Multi-Head Latent Attention (MLA) — DeepSeek-V2
#    核心思想：把 KV 压缩成低秩 latent 向量 c_t，推理时只缓存 c_t
#    KV cache 从 2*H*d_k 降到 d_c（压缩比可达 >93%）
# ============================================================
class MultiHeadLatentAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_c: int, d_rope: int = 0):
        """
        d_c:    latent 压缩维度（远小于 2 * n_heads * d_k）
        d_rope: 解耦 RoPE 的维度（简化版设为 0 不加 RoPE）
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_c = d_c

        # Q 侧也做低秩压缩（训练时减少激活内存）
        self.W_dq = nn.Linear(d_model, d_c, bias=False)     # Q 下投影
        self.W_uq = nn.Linear(d_c, n_heads * self.d_k, bias=False)  # Q 上投影

        # KV 联合压缩 — 这是 MLA 的核心
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)    # KV 下投影（推理时缓存这一层的输出 c_t）
        self.W_uk = nn.Linear(d_c, n_heads * self.d_k, bias=False)  # K 上投影（absorbed into W_o during inference）
        self.W_uv = nn.Linear(d_c, n_heads * self.d_k, bias=False)  # V 上投影

        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        # Q 低秩压缩: x -> c_q -> Q
        c_q = self.W_dq(x)                                             # (B, S, d_c)
        Q = self.W_uq(c_q).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # KV 联合低秩压缩: x -> c_kv -> K, V
        c_kv = self.W_dkv(x)                                           # (B, S, d_c) ← 推理时只需缓存这个
        K = self.W_uk(c_kv).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_uv(c_kv).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # 后面和标准 MHA 一样
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is None:
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)


# ============================================================
# KV Cache 大小对比
# ============================================================
def kv_cache_comparison():
    """打印典型配置下各方案的 KV cache 大小。"""
    d_model = 4096
    n_heads = 32
    d_k = d_model // n_heads  # 128
    n_layers = 32
    seq_len = 4096
    B = 2  # bytes per param (fp16)

    mha  = 2 * n_layers * n_heads * d_k * seq_len * B
    gqa  = 2 * n_layers * 8 * d_k * seq_len * B       # 8 KV heads (Llama 2 style)
    mqa  = 2 * n_layers * 1 * d_k * seq_len * B       # 1 KV head
    mla  = n_layers * 512 * seq_len * B                # d_c=512, 只存 c_t

    print("=" * 55)
    print(f"KV Cache 对比 (seq={seq_len}, {n_layers}层, d={d_model}, fp16)")
    print("=" * 55)
    print(f"MHA  (32 KV heads):  {mha/1e9:.2f} GB  (100%)")
    print(f"GQA  ( 8 KV heads):  {gqa/1e9:.2f} GB  ({gqa/mha*100:5.1f}%)")
    print(f"MQA  ( 1 KV head ):  {mqa/1e9:.2f} GB  ({mqa/mha*100:5.1f}%)")
    print(f"MLA  (d_c=512    ):  {mla/1e9:.2f} GB  ({mla/mha*100:5.1f}%)")


if __name__ == "__main__":
    kv_cache_comparison()
