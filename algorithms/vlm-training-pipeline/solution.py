"""
VLM 训练流水线 — ViT 解耦 PP，消除 Stage 0 瓶颈

核心问题:
  大 MoE VLM (如 400M ViT + 236B MoE LLM) 中，ViT 作为 PP stage 0
  导致 stage 不均衡 → PP bubble → 30%+ 训练效率损失

解决方案:
  Phase 1: ViT replicate 到所有 PP rank, 全量 no_grad forward (解耦 PP)
  Phase 2: PP 流水线只跑 LLM (各 stage 均衡), ViT recompute 在 backward
  Phase 3: ViT grad all-reduce + optimizer step

关键洞察: ViT recompute 的开销 ≤ PP bubble → 净零开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 简化的 ViT (Vision Transformer)
# ============================================================
class SimpleViT(nn.Module):
    """
    简化 ViT: patch embed → transformer blocks → output visual tokens

    实际的 ViT 还有 cls token, position embedding 等，这里简化。
    关键是: 输入 (B, 3, H, W) → 输出 (B, num_patches, vit_dim)
    """

    def __init__(self, image_size=336, patch_size=14, vit_dim=1024, num_layers=4, num_heads=8):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2  # 576 for 336/14
        self.patch_embed = nn.Conv2d(3, vit_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, vit_dim) * 0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=vit_dim, nhead=num_heads, dim_feedforward=vit_dim * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(vit_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, vit_dim)"""
        # Patch embedding
        x = self.patch_embed(images)          # (B, vit_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)     # (B, num_patches, vit_dim)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


# ============================================================
# 2. Projector (ViT dim → LLM dim)
# ============================================================
class VisionProjector(nn.Module):
    """
    将 ViT 的 visual tokens 投影到 LLM 的嵌入空间。

    常见实现:
    - 单层 Linear (LLaVA-1.0)
    - 2层 MLP + GELU (LLaVA-1.5, InternVL)
    - Perceiver Resampler (Flamingo, Qwen-VL)
    """

    def __init__(self, vit_dim=1024, llm_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vit_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, vit_dim) → (B, num_patches, llm_dim)"""
        return self.mlp(visual_tokens)


# ============================================================
# 3. 简化的 LLM
# ============================================================
class SimpleLLM(nn.Module):
    """
    简化的 Decoder-only LLM。

    输入: visual tokens + text tokens → 拼接后过 transformer → logits
    实际 LLM 还有 KV cache, RoPE 等，这里简化。
    """

    def __init__(self, vocab_size=32000, llm_dim=4096, num_layers=4, num_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, llm_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=llm_dim, nhead=num_heads, dim_feedforward=llm_dim * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(llm_dim)
        self.lm_head = nn.Linear(llm_dim, vocab_size, bias=False)

    def forward(
        self,
        visual_tokens: torch.Tensor,  # (B, num_patches, llm_dim)
        input_ids: torch.Tensor,      # (B, text_seq_len)
    ) -> torch.Tensor:
        """返回 logits: (B, total_seq_len, vocab_size)"""
        text_embeds = self.token_embed(input_ids)  # (B, text_seq_len, llm_dim)

        # 拼接: [visual_tokens | text_tokens]
        hidden = torch.cat([visual_tokens, text_embeds], dim=1)

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


# ============================================================
# 4. 朴素训练 (每个 micro-batch 独立过 ViT + LLM)
# ============================================================
def naive_train_step(
    batch: dict,
    vit: SimpleViT,
    projector: VisionProjector,
    llm: SimpleLLM,
    optimizer: torch.optim.Optimizer,
    num_micro_batches: int = 4,
) -> float:
    """
    朴素方案: 每个 micro-batch 独立过完整的 ViT → Projector → LLM

    问题:
    - ViT 被过了 num_micro_batches 次 (全部都需要存激活)
    - 不高效: ViT 的计算是冗余的 (不同 micro-batch 之间 ViT 权重没变)
    """
    images = batch['images']
    input_ids = batch['input_ids']
    labels = batch['labels']

    B = images.size(0)
    mb_size = B // num_micro_batches
    total_loss = 0.0

    for i in range(num_micro_batches):
        s, e = i * mb_size, (i + 1) * mb_size

        # ViT forward (每个 micro-batch 都过一遍 — 浪费!)
        visual_tokens = vit(images[s:e])
        visual_tokens = projector(visual_tokens)

        # LLM forward
        logits = llm(visual_tokens, input_ids[s:e])

        # Loss (只在 text 部分计算)
        num_visual = visual_tokens.size(1)
        text_logits = logits[:, num_visual:-1, :]  # shift
        text_labels = labels[s:e, 1:]               # shift

        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
        )

        (loss / num_micro_batches).backward()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()
    return total_loss / num_micro_batches


# ============================================================
# 5. 优化训练 (ViT 全量 + LLM Micro-Batch + ViT Recompute)
# ============================================================
def optimized_train_step(
    batch: dict,
    vit: SimpleViT,
    projector: VisionProjector,
    llm: SimpleLLM,
    optimizer: torch.optim.Optimizer,
    num_micro_batches: int = 4,
) -> float:
    """
    优化方案:

    Phase 1: ViT 全量 no_grad forward
      - 目的: 拿到所有 visual tokens 的值 (不需要梯度图)
      - 优势: 只过一次 ViT, 不存中间激活

    Phase 2: 每个 micro-batch
      - ViT recompute (有梯度, 只对当前 micro-batch 的图像)
      - Projector forward (有梯度)
      - LLM forward + backward
      - 梯度累积

    Phase 3: optimizer.step()

    显存对比:
      朴素: 存全量 ViT 激活 (或每个 mb 独立存)
      优化: 只存当前 micro-batch 的 ViT 激活
    """
    images = batch['images']
    input_ids = batch['input_ids']
    labels = batch['labels']

    B = images.size(0)
    mb_size = B // num_micro_batches
    total_loss = 0.0

    # ============================================
    # Phase 1: ViT 全量 forward (NO GRAD)
    # ============================================
    # 目的: 只拿值，不建梯度图，不存中间激活
    # 这些 visual tokens 的值可以用于构造 LLM 输入序列
    # (比如确定每个样本的序列长度，做 packing 等)
    with torch.no_grad():
        all_visual_tokens = vit(images)
        # all_visual_tokens: (global_B, num_patches, vit_dim)
        # 这是 detached 的，没有梯度图

    # 可以在这里用 all_visual_tokens 做一些预处理:
    # - 确定 packing 策略
    # - 排序使相近长度的样本在一起
    # - 预计算 attention mask 等

    # ============================================
    # Phase 2: LLM Micro-Batch + ViT Recompute
    # ============================================
    for i in range(num_micro_batches):
        s, e = i * mb_size, (i + 1) * mb_size

        # --- ViT Recompute (WITH GRAD) ---
        # 对当前 micro-batch 的图像重新过 ViT
        # 这次有梯度图! backward 可以把梯度传回 ViT 和 projector
        mb_visual = vit(images[s:e])
        mb_visual = projector(mb_visual)
        # mb_visual: (mb_size, num_patches, llm_dim)

        # --- LLM Forward ---
        logits = llm(mb_visual, input_ids[s:e])

        # --- Loss ---
        num_visual = mb_visual.size(1)
        text_logits = logits[:, num_visual:-1, :]
        text_labels = labels[s:e, 1:]

        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
        )

        # --- Backward (梯度累积) ---
        # 除以 num_micro_batches 确保梯度的期望值正确
        (loss / num_micro_batches).backward()
        total_loss += loss.item()

    # ============================================
    # Phase 3: Optimizer Step
    # ============================================
    optimizer.step()
    optimizer.zero_grad()

    return total_loss / num_micro_batches


# ============================================================
# 6. 带 Activation Checkpointing 的进一步优化
# ============================================================
def optimized_train_step_with_checkpoint(
    batch: dict,
    vit: SimpleViT,
    projector: VisionProjector,
    llm: SimpleLLM,
    optimizer: torch.optim.Optimizer,
    num_micro_batches: int = 4,
) -> float:
    """
    进一步优化: 用 torch.utils.checkpoint 自动管理 ViT 的 recompute。

    torch.utils.checkpoint 在 forward 时不保存中间激活，
    在 backward 时自动重新 forward 来计算梯度。
    效果和手动 recompute 一样，但代码更简洁。
    """
    from torch.utils.checkpoint import checkpoint

    images = batch['images']
    input_ids = batch['input_ids']
    labels = batch['labels']

    B = images.size(0)
    mb_size = B // num_micro_batches
    total_loss = 0.0

    for i in range(num_micro_batches):
        s, e = i * mb_size, (i + 1) * mb_size

        # checkpoint 自动处理 recompute:
        # forward 时只存输入和输出，不存中间激活
        # backward 时重新 forward 来计算中间激活和梯度
        mb_visual = checkpoint(
            lambda imgs: projector(vit(imgs)),
            images[s:e],
            use_reentrant=False,
        )

        logits = llm(mb_visual, input_ids[s:e])

        num_visual = mb_visual.size(1)
        text_logits = logits[:, num_visual:-1, :]
        text_labels = labels[s:e, 1:]

        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
        )

        (loss / num_micro_batches).backward()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()
    return total_loss / num_micro_batches


# ============================================================
# 7. TP-Aware 版本 (ViT 不切分, LLM 切分)
# ============================================================
def tp_aware_train_step(
    batch: dict,
    vit: SimpleViT,           # 每张卡上完整的 ViT (不做 TP)
    projector: VisionProjector,
    llm: SimpleLLM,           # TP 切分后的 LLM (只有 1/tp_size 参数)
    optimizer: torch.optim.Optimizer,
    num_micro_batches: int = 4,
    tp_size: int = 8,
    tp_rank: int = 0,
):
    """
    TP (Tensor Parallelism) 感知的训练步:

    - ViT: 每张卡都有完整权重，处理全量图像
      → 不需要 all-gather visual tokens! 每张卡本地就有
    - LLM: 权重按 TP 切分，需要 all-reduce 梯度
    - Projector: 可以不切分 (小) 或跟 LLM 一起切分

    关键优势:
    ViT 不做 TP → 省去 visual tokens 的 all-gather 通信
    ViT 参数量 << LLM → 每卡多存一份完整 ViT 的开销很小
    """
    images = batch['images']
    input_ids = batch['input_ids']
    labels = batch['labels']

    B = images.size(0)
    mb_size = B // num_micro_batches
    total_loss = 0.0

    # Phase 1: ViT 全量 forward (每张卡独立跑, 结果相同)
    with torch.no_grad():
        all_visual_tokens = vit(images)
        # 每张 TP 卡上都有完整的 visual tokens
        # 不需要 all-gather!

    # Phase 2: LLM micro-batch (LLM 内部做 TP 通信)
    for i in range(num_micro_batches):
        s, e = i * mb_size, (i + 1) * mb_size

        # ViT recompute (每张卡独立, 结果相同)
        mb_visual = vit(images[s:e])
        mb_visual = projector(mb_visual)

        # LLM forward (内部有 all-reduce 等 TP 通信)
        logits = llm(mb_visual, input_ids[s:e])

        num_visual = mb_visual.size(1)
        text_logits = logits[:, num_visual:-1, :]
        text_labels = labels[s:e, 1:]

        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
        )

        (loss / num_micro_batches).backward()
        total_loss += loss.item()

    # 注意: ViT 梯度每张卡独立计算，结果相同 (因为输入和参数都相同)
    # 所以 ViT 不需要 all-reduce 梯度!
    # LLM 的梯度由 TP 内部机制自动处理

    optimizer.step()
    optimizer.zero_grad()
    return total_loss / num_micro_batches


# ============================================================
# 8. PP-Decoupled 版本 (核心: 解决 PP stage 不均衡)
# ============================================================
def pp_decoupled_train_step(
    batch: dict,
    vit: SimpleViT,           # replicated 在所有 PP rank 上
    projector: VisionProjector,
    llm_stage_layers: list,   # PP 切分��当前 rank 的 LLM layers
    optimizer: torch.optim.Optimizer,
    num_micro_batches: int = 8,
    pp_rank: int = 0,
    pp_size: int = 8,
    is_first_stage: bool = True,
    is_last_stage: bool = False,
    send_fn=None,             # PP 通信: 发给下一个 stage
    recv_fn=None,             # PP 通信: 从上一个 stage 接收
):
    """
    PP-Decoupled VLM 训练步 — 消除 ViT 在 Stage 0 的瓶颈。

    核心问题:
      朴素 PP: ViT 在 Stage 0 → Stage 0 比其他 stage 慢 → 全局 PP bubble
      例: 400M ViT + 236B MoE (PP=8)
        每个 stage: ~30B params, Stage 0 额外 +400M (ViT)
        Stage 0 慢 ~1-2% (计算), 但 PP bubble 被放大到 ~15%+

    解决方案:
      1. ViT replicate 到所有 PP rank (每卡多 0.8GB)
      2. Phase 1: 所有 rank 并行跑 ViT 全量 forward (no_grad, 解耦于 PP)
      3. Phase 2: PP 1F1B 流水线只跑 LLM layers (各 stage 均衡!)
         - Stage 0 从本地 ViT 副本拿 visual tokens, 不走 PP 通信
         - Backward 时 Stage 0 做 ViT recompute
      4. ViT recompute 的开销被 PP bubble 吸收 → 净零开销

    为什么在 MoE 中更关键:
      - MoE 参数巨大 (236B) → PP stage 更多 → bubble 比例更大
      - MoE active 参数少 (21B) → 计算密度低 → 更怕任何 bubble
      - EP=64 已用大量 GPU → 通信开销大 → 不能再加 PP bubble
      - ViT/LLM 比例极端 (400M/236B = 0.17%) → 微小 ViT 导致严重不均衡
    """
    images = batch['images']
    input_ids = batch['input_ids']
    labels = batch['labels']

    B = images.size(0)
    mb_size = B // num_micro_batches

    # ================================================================
    # Phase 1: ViT 全量 Forward (所有 PP rank 并行, 解耦于 PP 流水线)
    # ================================================================
    # 关键: 这一步不走 PP 通信, 每张卡独立跑
    # 因为 ViT 是 replicated 的, 所有卡算出来结果一样
    # 时间: O(ViT_forward(全量)) — 只有一次, 且和 PP 流水线解耦
    with torch.no_grad():
        all_visual_tokens = vit(images)
        # (global_B, num_patches, vit_dim)
        # detached, 无梯度图 → 不占额外显存

    # 可在此做预处理: packing 排序, attention mask 预计算等
    # 因为已经知道每张图产出多少 visual tokens

    # ================================================================
    # Phase 2: PP 1F1B 流水线 (ViT 不在关键路径上!)
    # ================================================================
    # 简化的 1F1B schedule (实际�� Megatron/DeepSpeed 的 schedule)
    total_loss = 0.0

    for mb_idx in range(num_micro_batches):
        s, e = mb_idx * mb_size, (mb_idx + 1) * mb_size

        if is_first_stage:
            # === Stage 0: 构造输入 ===
            # 方案 A (ViT freeze): 直接用 Phase 1 的 visual tokens
            # 方案 B (ViT trainable): recompute with grad
            if vit.training:
                # ViT Recompute (有梯度) — 只对这个 MB 的图像
                # 时间: O(ViT_forward(mb_size)) — 很小
                # 这段时间里, Stage 1-7 在处理前面 MB 的 backward
                # → recompute 开销被 PP bubble 吸收!
                mb_visual = vit(images[s:e])
            else:
                # ViT freeze: 直接用 Phase 1 的值, 不需要 recompute
                mb_visual = all_visual_tokens[s:e]

            mb_visual = projector(mb_visual)

            # 构造 LLM 输入: [visual_tokens | text_embeddings]
            # 过当前 stage 的 LLM layers
            text_embeds = llm_stage_layers[0].token_embed(input_ids[s:e]) \
                if hasattr(llm_stage_layers[0], 'token_embed') else None
            hidden = torch.cat([mb_visual, text_embeds], dim=1) if text_embeds is not None else mb_visual

            for layer in llm_stage_layers:
                hidden = layer(hidden)

            # 发给下一个 PP stage
            if send_fn:
                send_fn(hidden)

        elif is_last_stage:
            # === Last Stage: 接收 + loss ===
            hidden = recv_fn() if recv_fn else torch.zeros(1)

            for layer in llm_stage_layers:
                hidden = layer(hidden)

            # 计算 loss + backward
            # logits = lm_head(hidden)
            # loss = cross_entropy(logits, labels[s:e])
            # (loss / num_micro_batches).backward()

        else:
            # === Middle Stage: 接收 → LLM layers → 发送 ===
            hidden = recv_fn() if recv_fn else torch.zeros(1)
            for layer in llm_stage_layers:
                hidden = layer(hidden)
            if send_fn:
                send_fn(hidden)

    # ================================================================
    # Phase 3: ViT Gradient All-Reduce + Optimizer Step
    # ================================================================
    # ViT 在每张卡上独立计算梯度 (相同输入+参数 → 梯度相同)
    # 但如果用了数据并行 (DP), 不同 DP rank 的输入不同 → 需要 all-reduce
    if vit.training:
        # all_reduce_vit_grads(vit)  # 跨 DP ranks
        pass

    optimizer.step()
    optimizer.zero_grad()
    return total_loss / num_micro_batches


# ============================================================
# 演示
# ============================================================
def demo():
    print("=" * 65)
    print("VLM 训练流水线 — ViT 全量 + LLM Micro-Batch + Recompute")
    print("=" * 65)

    device = 'cpu'  # 演示用 CPU
    B = 8  # global batch size

    # 创建模型 (小规模演示)
    vit = SimpleViT(image_size=56, patch_size=14, vit_dim=128, num_layers=2, num_heads=4).to(device)
    projector = VisionProjector(vit_dim=128, llm_dim=256).to(device)
    llm = SimpleLLM(vocab_size=1000, llm_dim=256, num_layers=2, num_heads=4).to(device)

    # 模拟数据
    images = torch.randn(B, 3, 56, 56, device=device)
    input_ids = torch.randint(0, 1000, (B, 32), device=device)
    labels = input_ids.clone()
    labels[:, :10] = -100  # 前 10 个 token 是 user turn
    batch = {'images': images, 'input_ids': input_ids, 'labels': labels}

    # 参数量统计
    vit_params = sum(p.numel() for p in vit.parameters())
    proj_params = sum(p.numel() for p in projector.parameters())
    llm_params = sum(p.numel() for p in llm.parameters())
    print(f"\nViT:       {vit_params:>10,} params ({vit_params/1e6:.1f}M)")
    print(f"Projector: {proj_params:>10,} params ({proj_params/1e6:.1f}M)")
    print(f"LLM:       {llm_params:>10,} params ({llm_params/1e6:.1f}M)")
    print(f"比例: ViT/LLM = {vit_params/llm_params:.1%}")

    # 朴素 vs 优化
    print(f"\n--- 朴素方案 (每个 micro-batch 独立过 ViT) ---")
    optimizer = torch.optim.AdamW(
        list(vit.parameters()) + list(projector.parameters()) + list(llm.parameters()),
        lr=1e-4,
    )
    loss = naive_train_step(batch, vit, projector, llm, optimizer, num_micro_batches=4)
    print(f"Loss: {loss:.4f}")

    print(f"\n--- 优化方案 (ViT 全量 + Recompute) ---")
    optimizer = torch.optim.AdamW(
        list(vit.parameters()) + list(projector.parameters()) + list(llm.parameters()),
        lr=1e-4,
    )
    loss = optimized_train_step(batch, vit, projector, llm, optimizer, num_micro_batches=4)
    print(f"Loss: {loss:.4f}")

    # 显存分析
    print(f"\n--- 显存分析 (以 SigLIP-400M + LLaMA-72B 为例) ---")
    # ViT 激活估算
    num_patches = (336 // 14) ** 2  # 576
    vit_dim_real = 1152
    bytes_per_element = 2  # bf16
    per_image_vit_act = num_patches * vit_dim_real * bytes_per_element
    print(f"  每张图 ViT 激活: {per_image_vit_act / 1024:.1f} KB")
    print(f"  全量 batch=32: {32 * per_image_vit_act / 1024 / 1024:.1f} MB ← 很小")
    print(f"  但 ViT 内部中间激活 (attention maps 等) 远大于输出")
    print(f"  存全量中间激活 (32张) ≈ 数 GB → 用 recompute 避免!")
    print(f"")
    print(f"  优化后: 只存 1 个 micro-batch (8张) 的中间激活 ≈ 几百 MB")
    print(f"  代价: 多 4 次 ViT forward (400M 的 ViT 很快)")


if __name__ == "__main__":
    demo()
