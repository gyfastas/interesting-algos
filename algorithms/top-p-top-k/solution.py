"""
Top-p & Top-k 采样 — LLM 解码策略手写实现

包含: Temperature / Top-k / Top-p / 组合采样 / 简易 generate 循环
"""

import torch
import torch.nn.functional as F


# ============================================================
# 1. Temperature 采样
# ============================================================
def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Temperature 采样: logits / T → softmax → 采样

    T < 1: 分布更尖锐 (更确定)
    T = 1: 标准 softmax
    T > 1: 分布更平坦 (更随机)
    T → 0: 退化为 greedy (argmax)
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ============================================================
# 2. Top-k 采样
# ============================================================
def top_k_sample(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """
    Top-k 采样: 只保留概率最大的前 k 个 token，其余设为 -inf。

    算法:
    1. torch.topk 找到前 k 个 logits 及其索引
    2. 创建全 -inf 的 tensor
    3. 把前 k 个 logits 放回对��位置
    4. softmax + 采样
    """
    k = min(k, logits.size(-1))

    # 找到第 k 大的值作为阈值
    top_k_values = torch.topk(logits, k, dim=-1).values
    threshold = top_k_values[..., -1:]  # 第 k 大的值

    # 小于阈值的设为 -inf
    filtered_logits = logits.masked_fill(logits < threshold, float('-inf'))

    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ============================================================
# 3. Top-p (Nucleus) 采样
# ============================================================
def top_p_sample(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Top-p (Nucleus) 采样: 保留累积概率 ≥ p 的最小 token 集合。

    算法:
    1. 对 logits 排序 (降序)
    2. 计算 softmax 概率的 cumsum
    3. 找到 cumsum ≥ p 的位置，之后的 token 设为 -inf
    4. 还原到原始顺序
    5. softmax + 采样

    关键: 候选集大小是自适应的
    - 模型很确定 → 少数 token 就达到 p → 候选集小
    - 模型不确定 → 需要很多 token 达到 p → 候选集大
    """
    # 降序排列
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # 创建 mask: 移除累积概率超过 p 的 token
    # 注意: 要保留第一个使 cumsum >= p 的 token
    # 所以用 cumsum - 当前概率 >= p 来判断 (即之前的已经够了)
    remove_mask = (cumulative_probs - sorted_probs) >= p
    sorted_logits[remove_mask] = float('-inf')

    # 还原到原始顺序
    original_logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)

    probs = F.softmax(original_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ============================================================
# 4. 组合采样 (Temperature + Top-k + Top-p)
# ============================================================
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    完整采样流程: Temperature → Top-k → Top-p → Sample

    顺序很重要:
    1. Temperature 先调节分布形状
    2. Top-k 粗筛，砍掉明显不可能的
    3. Top-p 精筛，自适应保留合理候选
    4. 最后 multinomial 采样
    """
    # Greedy
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    # Step 1: Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Step 2: Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Step 3: Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        remove_mask = (cumulative_probs - sorted_probs) >= top_p
        sorted_logits[remove_mask] = float('-inf')

        logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)

    # Step 4: Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ============================================================
# 5. Repetition Penalty
# ============================================================
def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    """
    对已生成的 token 施加重复惩罚:
    - logit > 0: 除以 penalty (降低概率)
    - logit < 0: 乘以 penalty (更负，更不可能)

    penalty = 1.0 时无惩罚
    典型值: 1.1 ~ 1.3
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    logits = logits.clone()
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits


# ============================================================
# 6. 简易 Generate 循环
# ============================================================
@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    简易自回归生成循环。

    每一步:
    1. 模型前向 → 取最后一个位置的 logits
    2. 施加 repetition penalty
    3. Temperature + Top-k + Top-p 采样
    4. 拼接到 input_ids
    5. 遇到 eos 或达到 max_new_tokens 停止
    """
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # 前向传播，取最后一个 token 的 logits
        outputs = model(generated)
        # outputs 可能是 logits tensor 或有 .logits 属性的对象
        if hasattr(outputs, 'logits'):
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
        else:
            logits = outputs[:, -1, :]

        # Repetition penalty
        if repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                logits[b] = apply_repetition_penalty(
                    logits[b],
                    generated[b].tolist(),
                    repetition_penalty,
                )

        # 采样
        next_token = sample_next_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
        )  # (batch, 1)

        generated = torch.cat([generated, next_token], dim=-1)

        # EOS 检查
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated


# ============================================================
# 演示 (不需要实际模型, 用随机 logits 展示采样效果)
# ============================================================
def demo():
    torch.manual_seed(42)
    vocab_size = 10
    token_names = ['猫', '狗', '鱼', '鸟', '兔', '蛇', '鼠', '虎', '龙', '马']

    # 模拟模型输出的 logits
    logits = torch.tensor([5.0, 4.2, 3.0, 2.1, 1.5, 0.8, 0.3, -0.5, -1.0, -2.0])

    print("=" * 60)
    print("原始 logits 和 softmax 概率")
    print("=" * 60)
    probs = F.softmax(logits, dim=-1)
    for i, (name, l, p) in enumerate(zip(token_names, logits, probs)):
        bar = '█' * int(p * 50)
        print(f"  {name}: logit={l:>5.1f}  prob={p:.3f}  {bar}")

    print(f"\n{'=' * 60}")
    print("Temperature 对分布的影响")
    print("=" * 60)
    for T in [0.3, 0.7, 1.0, 1.5, 2.0]:
        scaled = F.softmax(logits / T, dim=-1)
        top1 = scaled.max().item()
        entropy = -(scaled * scaled.log()).sum().item()
        print(f"  T={T:.1f}: top1_prob={top1:.3f}  entropy={entropy:.2f}  分布={'尖锐' if T < 1 else '平坦' if T > 1 else '标准'}")

    print(f"\n{'=' * 60}")
    print("Top-k 采样 (k=3)")
    print("=" * 60)
    k = 3
    top_k_vals = torch.topk(logits, k)
    print(f"  候选: {[token_names[i] for i in top_k_vals.indices.tolist()]}")
    filtered_probs = F.softmax(logits.masked_fill(
        logits < top_k_vals.values[-1], float('-inf')
    ), dim=-1)
    for i in top_k_vals.indices.tolist():
        print(f"    {token_names[i]}: {filtered_probs[i]:.3f}")

    print(f"\n{'=' * 60}")
    print("Top-p 采样 (p=0.9)")
    print("=" * 60)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    print("  排序后:")
    for i, (idx, sp, cs) in enumerate(zip(sorted_idx, sorted_probs, cumsum)):
        name = token_names[idx]
        marker = " ← 截断" if cs.item() >= 0.9 and (i == 0 or cumsum[i-1].item() < 0.9) else ""
        in_set = "✓" if (cs - sp).item() < 0.9 else "✗"
        print(f"    [{in_set}] {name}: prob={sp:.3f}  cumsum={cs:.3f}{marker}")

    print(f"\n{'=' * 60}")
    print("采样 1000 次对比")
    print("=" * 60)
    counts = {s: {'greedy': 0, 'top_k': 0, 'top_p': 0, 'combined': 0} for s in token_names}
    for _ in range(1000):
        g = logits.argmax().item()
        counts[token_names[g]]['greedy'] += 1

        tk = top_k_sample(logits.unsqueeze(0), k=3).item()
        counts[token_names[tk]]['top_k'] += 1

        tp = top_p_sample(logits.unsqueeze(0), p=0.9).item()
        counts[token_names[tp]]['top_p'] += 1

        cb = sample_next_token(logits.unsqueeze(0), temperature=0.7, top_k=5, top_p=0.9).item()
        counts[token_names[cb]]['combined'] += 1

    print(f"  {'Token':>4} {'Greedy':>8} {'Top-k=3':>8} {'Top-p=.9':>9} {'T=.7+k=5+p=.9':>15}")
    print(f"  {'-'*48}")
    for name in token_names:
        c = counts[name]
        print(f"  {name:>4} {c['greedy']:>8} {c['top_k']:>8} {c['top_p']:>9} {c['combined']:>15}")


if __name__ == "__main__":
    demo()
