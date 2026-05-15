# MoE Transformer Layer — 手写 Forward & Backward

> **扩展题**：在 [MHA 手写 Forward & Backward](../multi-head-attention-with-backward/) 的基础上，加入 **Mixture of Experts (MoE)** 层，实现完整的手写反向传播。

## 问题描述

实现一个单层 MoE Transformer，包含：

1. **Token + Positional Embedding**
2. **Multi-Head Attention**（保留残差连接）
3. **MoE 层**：
   - Router：线性投影 + Softmax
   - Top-K 专家选择（Soft Mask）
   - 多个 Expert FFN（每个专家独立的两层前馈网络）
   - 加权求和输出
4. **投影层**：将 d_model 映射到词表维度
5. **Sequence CrossEntropy Loss**

**要求**：纯 NumPy 实现，包含完整的反向传播（手写 backward）。

## 核心概念

### MoE 路由

Router 对序列中每个 token 输出对 `num_experts` 个专家的偏好概率：

```
gates = softmax(X @ W_r + b_r)   # (batch, seq, num_experts)
```

对每个 token，只保留 Top-K 的概率，其余置 0 并重新归一化：

```
mask[i,j] = 1  if j in topk(gates[i]) else 0
norm_gates = (gates * mask) / sum(gates * mask, axis=-1)
```

### Expert 计算

每个 token 的输出是其所选 K 个专家输出的加权平均：

```
output[b,s] = sum_e( norm_gates[b,s,e] * Expert_e(X[b,s]) )
```

### 负载均衡损失

为了防止所有 token 都涌向少数专家（**routing collapse**），加入辅助损失：

```
f_e      = 被路由到专家 e 的 token 比例
p_e      = 所有 token 对专家 e 的平均门控概率
L_aux    = α · num_experts · Σ_e( f_e · p_e )
```

- `f_e` 大且 `p_e` 大 → 高惩罚 → push `p_e` 下降
- `f_e` 小且 `p_e` 小 → 低惩罚 → 允许 `p_e` 上升
- 最优：所有专家均匀分担，`f_e = p_e = 1 / num_experts`

### 反向传播链

从 `SequenceCrossEntropyLoss` 的合并梯度 `(softmax(logits) - one_hot) / N` 开始：

1. `dLoss/dProjection` → `dLoss/dMoE_output`
2. MoE backward：
   - **专家梯度**：每个专家只收到被分配 token 的梯度（稀疏）
   - **Router 梯度**：
     - 主任务梯度：通过 Top-K mask 传播
     - 辅助损失梯度：传播到所有位置，推动负载均衡
3. `dLoss/dMHA_output` → MHA backward（复用之前的实现）
4. `dLoss/dEmbedding` → 更新 Token & Positional Embedding

## 复杂度分析

| 模块 | 前向 | 反向 | 空间 |
|------|------|------|------|
| Embedding | O(B·S·D) | O(B·S·D) | O(V·D + S·D) |
| MHA | O(B·S²·D) | O(B·S²·D) | O(B·S·D) |
| MoE (含 E 个专家) | O(B·S·E·D·d_ff) | O(B·S·E·D·d_ff) | O(B·S·D·E) |
| Projection | O(B·S·D·V) | O(B·S·D·V) | O(D·V) |

> 注：MoE 的前向/反向复杂度与激活专家数成正比。Top-K 选择下，每个 token 只激活 K 个专家，因此实际复杂度可降至 `O(B·S·K·D·d_ff)`。

## 运行结果

```
======================================================================
MoE Transformer 训练: Copy Task
======================================================================
配置: vocab=8, d_model=32, heads=4
      experts=4, top_k=2, seq_len=8
  epoch     0: loss=2.523083 (ce=2.207845, aux=0.315237), acc=0.1250
  epoch   200: loss=0.388938 (ce=0.068669, aux=0.320269), acc=1.0000
  epoch   400: loss=0.316329 (ce=0.015061, aux=0.301268), acc=1.0000
  epoch   800: loss=0.303972 (ce=0.005504, aux=0.298467), acc=1.0000
  epoch  1200: loss=0.303754 (ce=0.002602, aux=0.301151), acc=1.0000
  epoch  1600: loss=0.303056 (ce=0.001828, aux=0.301228), acc=1.0000
  epoch  1999: loss=0.303036 (ce=0.001704, aux=0.301332), acc=1.0000

最终测试:
  输入: [5 7 7 6 3 6 0 5]  预测: [5 7 7 6 3 6 0 5]  ✓
  输入: [1 0 0 7 7 2 0 0]  预测: [1 0 0 7 7 2 0 0]  ✓
  输入: [5 7 1 5 1 4 2 0]  预测: [5 7 1 5 1 4 2 0]  ✓
  输入: [1 2 1 7 1 3 5 7]  预测: [1 2 1 7 1 3 5 7]  ✓

专家负载分布分析:
  Expert 0:   127718 tokens ( 24.9%) ████
  Expert 1:   120549 tokens ( 23.5%) ████
  Expert 2:   130087 tokens ( 25.4%) █████
  Expert 3:   133646 tokens ( 26.1%) █████
  理想分布: 每个专家 25.0%
  负载均衡度: 0.94% (标准差)
```

### 关键观察

1. **收敛**：200 epochs 内 Copy Task 准确率即达到 100%
2. **负载均衡**：加入 `L_aux` 后，4 个专家的标准差仅 0.94%，接近理想均匀分布
3. **无 L_aux 时的灾难**：如果不加辅助损失，99% 的 token 会涌向 2 个专家，另外 2 个专家几乎不被激活（routing collapse）

### 超参数对比

**不同 `aux_loss_coef` 对负载均衡的影响**（训练 1000 epochs）：

| α | 负载均衡度 (std) | Expert 0 | Expert 1 | Expert 2 | Expert 3 |
|---|------------------|----------|----------|----------|----------|
| 0.01 | 24.35% | 0.4% | 49.2% | 49.5% | 0.9% |
| 0.05 | 17.13% | 39.7% | 1.6% | 43.0% | 15.7% |
| 0.10 | 6.47% | 17.1% | 25.7% | 22.4% | 34.9% |
| **0.30** | **1.75%** | **28.0%** | **24.1%** | **23.4%** | **24.4%** |
| 0.50 | 0.77% | 23.9% | 25.7% | 24.6% | 25.7% |
| 1.00 | 0.42% | 24.7% | 25.2% | 24.5% | 25.6% |

## 核心代码

```python
# Router + Top-K 门控
gates = router.forward(X)                        # (B,S,E)
topk_mask = zeros_like(gates)
for each token: topk_mask[token, topk(gates)] = 1
norm_gates = (gates * topk_mask) / sum(gates * topk_mask)

# 专家加权输出
output = sum_e( norm_gates[..., e] * Expert_e(X) )

# Load Balancing Loss
f_e = load_counts[e] / (T * top_k)
p_e = mean(gates[..., e])
L_aux = alpha * num_experts * sum_e(f_e * p_e)

# Backward 关键：
# 1. 主任务梯度只通过 top-k mask 传播
# 2. 辅助损失梯度传播到所有位置，推动均衡
dRouter_main = dOutput * expert_out * topk_mask
dRouter_aux  = alpha * N * f_e / T   (对所有位置)
dX_router = router.backward(dRouter_main + dRouter_aux)
```

## 扩展思考

1. **Sparse MoE**：实际实现中不需要让所有 token 经过所有专家。可以只对 Top-K 专家计算前向/反向，复杂度从 `O(E)` 降到 `O(K)`。
2. **Expert Choice**：另一种路由策略是让每个专家选择 Top-K token（而非每个 token 选 Top-K 专家），天然保证负载均衡。
3. **Shared Expert**：如 DeepSeekMoE，保留 1 个始终激活的共享专家 + N 个路由专家，兼顾泛化与特化。
4. **Capacity Factor**：限制每个专家每轮处理的最大 token 数，超出部分被「溢出」到下一专家或丢弃。
