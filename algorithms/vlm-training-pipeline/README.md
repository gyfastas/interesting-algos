# VLM 训练流水线：ViT 解耦 PP — 消除 Stage 0 瓶颈

## 问题描述

**在大 MoE VLM 训练中 (如 300B+ MoE LLM + 400M ViT)，ViT 和 LLM 的 size 极度不均衡。如果把 ViT 放在 PP stage 0，ViT 会成为流水线瓶颈。如何通过 ViT 全量 forward + LLM micro-batch + ViT recompute 的方式来消除这个瓶颈？**

这不只是"省显存"——真正的动机是**消除 Pipeline Parallelism 中 ViT 作为 stage 0 的 bubble 瓶颈**，让大规模 MoE VLM 训练的吞吐量不被一个小 ViT 卡住。

> **参考**: [LongCat-Flash Technical Report (2601.16725)](https://arxiv.org/abs/2601.16725) — Meituan LongCat 团队在大规模 MoE VLM 训练中采用了 ViT decoupling 策��，将 ViT 从 PP 流水线中解耦出来，避免 vision encoder 成为 pipeline stage 的瓶颈。

## 核心问题：PP Stage 不均衡

### 大 MoE VLM 的并行策略

典型配置 (DeepSeek-VL2, InternVL-2.5 等):

```
模型:  ViT (400M) + Projector (20M) + MoE LLM (236B active 21B)
并行:  TP=8, PP=8, EP=64, DP=...
卡数:  512+ GPUs
```

PP (Pipeline Parallelism) 把模型按层切成多个 stage：

```
PP stage 0: ViT + Projector + LLM layers 0-4
PP stage 1: LLM layers 5-9
PP stage 2: LLM layers 10-14
...
PP stage 7: LLM layers 35-39 + lm_head
```

### 问题：Stage 0 的 ViT 瓶颈

**每个 PP stage 应该计算量大致相等**，否则快的 stage 要等慢的 stage → PP bubble。

但实际上：

```
Stage 0 (ViT + ���层 LLM):
  ViT forward:  400M 参数的 ViT, 处理 micro_batch 的图像
  LLM layers:   每层 ~MoE expert 路由 + 计算
  总时间:       T_vit + T_llm_partial

Stage 1-7 (纯 LLM layers):
  LLM layers:   每层同样的计算量
  总时间:       T_llm_partial

问题: T_vit 是额外的! Stage 0 比其他 stage 慢 → 全局等 stage 0 → bubble!
```

更具体地：

| | Stage 0 | Stage 1-7 |
|---|---------|-----------|
| LLM 层数 | 5 层 | 5 层 |
| 额外负担 | **ViT forward + backward** | 无 |
| 计算时间 | T + T_vit | T |
| 状态 | **瓶颈** | 等待 |

### bubble 有多大？

```
假设: 8 PP stages, 每 stage LLM 计算 10ms
  平衡时: 每 stage 10ms, 总 pipeline 延迟 = 80ms
  Stage 0 有 ViT: Stage 0 = 10 + 5 = 15ms
  其他 stage: 10ms, 但要等 Stage 0

每个 micro-batch 的 bubble: 5ms × 7 stages = 35ms 浪费
8 个 micro-batch: 35 × 8 = 280ms bubble

PP 效率: 80ms / (80 + 35) ≈ 70% → 浪费 30%!
```

**ViT 只有 400M 参数 (占总参数 < 0.2%)，却导致了 30% 的训练效率损失。**

## 解决方案：ViT 解耦 PP

### 核心思想

**把 ViT 从 PP 流水线中拿出来，让它独立运行。**

```
朴素 PP (ViT 在 Stage 0 内):
  Stage 0: [ViT + LLM_0-4] ← 瓶颈!
  Stage 1: [LLM_5-9]
  ...

优化 PP (ViT 解耦):
  所有卡: ViT 全量 forward (独立于 PP, no_grad)
  Stage 0: [LLM_0-4]       ← 和其他 stage 均衡了!
  Stage 1: [LLM_5-9]
  ...
  每个 micro-batch 的 ViT recompute 分摊到每个 stage 的 backward 中
```

### 为什么可以把 ViT 拿出来？

1. **ViT 很小**：每张 GPU 都能放一份完整 ViT (400M = 0.8GB bf16)
2. **ViT 不参与 PP 切分**：所有 PP stage 的卡上都 replicate 一份完整 ViT
3. **ViT forward 和 PP 流水线解耦**：先全量过 ViT，再开始 PP 流水线

### 三阶段流水线

```
Phase 1: ViT 全量 Forward (所有 PP rank 并行, no_grad)
  ├─ PP rank 0: vit(all_images) → all_visual_tokens
  ├─ PP rank 1: vit(all_images) → all_visual_tokens  (同样的计算!)
  └─ PP rank 7: vit(all_images) → all_visual_tokens
  每张卡都有完整的 visual tokens, 不需要跨 PP 通信

Phase 2: PP 流水线 (1F1B schedule)
  对每个 micro-batch:
    Stage 0: 从 all_visual_tokens 取对应的 visual tokens
             拼接 text embeddings → LLM layers 0-4 → send to stage 1
    Stage 1: recv → LLM layers 5-9 → send
    ...
    Stage 7: recv → LLM layers 35-39 → loss

  Backward: ViT recompute 发生在 gradient 回传到 stage 0 时
    Stage 0 backward: recv grad → LLM layers 0-4 backward
                      → 需要 visual tokens 的梯度
                      → ViT recompute (这个 MB 的图像, 有 grad)
                      → backward 传到 ViT

Phase 3: All-reduce ViT 梯度 + Optimizer Step
  ViT 在每张卡上都有副本 → 梯度需要 all-reduce
  (如果 ViT freeze 则不需要)
```

### 关键：ViT Recompute 不在关键路径上

```
朴素 (ViT 在 PP 关键路径):
  [ViT fwd] → [LLM stage 0 fwd] → [stage 1 fwd] → ... → [loss]
  ↑ 这段是额外的, 阻塞了整条 pipeline

优化 (ViT 解耦):
  Phase 1 (预处理):  [ViT fwd (no_grad)] → visual tokens 就绪
  Phase 2 (PP 流水):  [LLM stage 0] → [stage 1] → ... → [loss]
                       ↑ 不含 ViT, 各 stage 均衡!

  Backward 时的 recompute:
    Stage 7 backward → ... → Stage 0 backward
    此时 Stage 0 做 ViT recompute:
    - 但这时 Stage 1-7 正在做前面 micro-batch 的 forward
    - ViT recompute 和后续 stage 的 forward 重叠!
    - 所以 recompute 的开销被 pipeline 的 bubble 吸收了
```

**ViT recompute 的时间 ≤ PP bubble 的时间 → 净开销为零。**

## 为什么在 MoE 中更关键

MoE 模型的特点让这个优化更重要：

| 因素 | MoE 的影响 |
|------|-----------|
| **参数量巨大** | 236B 参数 → PP stage 更多 → bubble 比例更高 |
| **Expert Parallelism** | EP=64 → 已经用了很多 GPU → 通信开销大 |
| **Active 参数少** | 每 token 只激活 21B → 计算密度低 → 更怕 bubble |
| **ViT 比例更极端** | 400M / 236B = 0.17% → 微不足道的 ViT 导致严重的 PP 不均衡 |
| **4D 并行** | TP × PP × EP × DP → 任何一个维度的 bubble 被其他维度放大 |

```
MoE 236B, TP=8, PP=16, EP=64:
  每个 PP stage: ~14.75B / 16 ≈ 0.92B active 参数
  ViT: 400M = 0.4B → 相当于半个 PP stage!
  如果 ViT 在 stage 0: stage 0 的计算量是其他 stage 的 ~1.4x
  → PP bubble = 40% × 15 stages = 巨大浪费
```

## ViT Freeze 时的简化

很多 VLM 训练 ViT 是 freeze 的 (只训练 projector + LLM)：

```python
# ViT freeze → 不需要 backward → 不需要 recompute!
# Phase 1 就是全部了

with torch.no_grad():
    all_visual_tokens = vit(all_images)    # 全量, no_grad
    all_visual_tokens = projector(all_visual_tokens)  # projector 可能要 grad

# Phase 2: PP 流水线只跑 LLM
for micro_batch in pp_schedule:
    visual = all_visual_tokens[mb_indices]
    llm_forward_backward(visual, text)     # projector 的 grad 在这里
```

ViT freeze 时，Phase 1 的 visual tokens 可以直接给所有 micro-batch 用，连 recompute 都不需要。这是最简单也最常见的场景。

## 对比总结

| 方案 | ViT 在 PP 关键路径? | PP 均衡? | ViT forward 次数 |
|------|---------------------|----------|-----------------|
| 朴素 (ViT 在 stage 0) | 是 | 不均衡 (stage 0 瓶颈) | N (每个 MB 1 次) |
| ViT 解耦 (freeze) | 否 | 均衡 | 1 (全量, no_grad) |
| ViT 解耦 (trainable) | 否 | 均衡 | 1 + N (全量 + recompute) |

## 核心代码

```python
def vlm_train_step_pp_decoupled(
    batch, vit, projector, llm_pp_stages,
    optimizer, pp_schedule,
):
    """ViT 解耦 PP 的训练步"""
    images, text_ids, labels = batch

    # === Phase 1: ViT 全量 (解耦于 PP, 所有 rank 并行) ===
    with torch.no_grad():
        all_visual = vit(images)  # 每张卡独立跑

    # === Phase 2: PP 流水线 (1F1B schedule) ===
    # ViT 不在 pipeline 中! stage 0 和其他 stage 均衡
    for mb_idx in pp_schedule:
        s, e = mb_idx * mb_size, (mb_idx + 1) * mb_size

        if is_first_stage:
            # Stage 0: ViT recompute (如果 ViT trainable)
            mb_visual = vit(images[s:e])          # recompute, with grad
            mb_visual = projector(mb_visual)
            hidden = concat(mb_visual, embed(text_ids[s:e]))
            send_to_next_stage(llm_layers(hidden))

        elif is_last_stage:
            hidden = recv_from_prev_stage()
            logits = llm_layers(hidden)
            loss = compute_loss(logits, labels[s:e])
            (loss / num_mb).backward()

        else:  # middle stages
            hidden = recv_from_prev_stage()
            send_to_next_stage(llm_layers(hidden))

    # === Phase 3: ViT all-reduce + optimizer ===
    all_reduce_vit_grads(vit)  # 因为 ViT 是 replicated 的
    optimizer.step()
```

## 动画演示

> 打开 `animation.html` 查看 PP 流水线中 ViT 瓶颈的可视化，以及解耦方案如何消除 bubble。

## 答案与总结

| 要点 | 结论 |
|------|------|
| **真正的动机** | 不只是省显存——是消除 PP stage 0 的 ViT 瓶颈 |
| PP 不均衡 | ViT 在 stage 0 → stage 0 比其他 stage 慢 → 全局 bubble |
| MoE 更严重 | 400M ViT / 236B MoE → 极端不均衡 → PP bubble 可达 30%+ |
| 解耦方案 | ViT replicate 到所有卡，独立全量 forward，不参与 PP 切分 |
| Recompute | backward 时 ViT recompute 的开销被 PP bubble 吸收 → 净零开销 |
| ViT freeze | 最简单：全量 no_grad → visual tokens 直接复用，无 recompute |

**一句话总结**：大 MoE VLM 训练中，把 ViT 从 PP 流水线中解耦出来——每张卡 replicate 一份小 ViT 独立跑全量 forward，LLM 的 PP 流水线各 stage 恢复均衡，ViT recompute 开销被 pipeline bubble 吸收，把 0.2% 参数量的 ViT 从全局瓶颈变成零开销。
