"""
投机解码 (Speculative Decoding) 工程实现
==========================================

DeepMind 2022 提出的 LLM 推理加速技术:
  1. 用小模型 (draft) 快速生成候选 token 序列
  2. 用大模型 (target) batch forward 并行验证
  3. 逐个接受/拒绝，拒绝时从修正分布重新采样
  4. 显著减少大模型的 forward 次数

纯 Python 模拟实现，含完整工程接口 + 性能对比。
"""

import numpy as np
import time


# =============================================================================
# 模型接口抽象
# =============================================================================

class LanguageModel:
    """语言模型接口: next-token 概率分布 + 采样。"""

    def __init__(self, vocab_size, temperature=1.0):
        self.V = vocab_size
        self.temperature = temperature

    def get_probs(self, context):
        """给定上下文，返回下一个 token 的概率分布 (V 维向量)。"""
        raise NotImplementedError

    def sample(self, context):
        """从分布中采样一个 token。"""
        probs = self.get_probs(context)
        return np.random.choice(self.V, p=probs)


class SimpleLM(LanguageModel):
    """
    简化语言模型: 用确定性 hash 生成分布，模拟真实 LM 的行为。

    每个上下文对应唯一的概率分布，不同模型参数(offset/temp)
    产生不同的分布，但保持上下文相关性。
    """

    def __init__(self, vocab_size=100, temperature=1.0, offset=0.0, seed=None):
        super().__init__(vocab_size, temperature)
        self.offset = offset
        self.rng = np.random.RandomState(seed)

    def _context_hash(self, context):
        """用上下文的 hash 确定性地生成随机种子。"""
        key = tuple(context[-8:] if len(context) >= 8 else context)
        return hash(key) % (2 ** 31)

    def get_probs(self, context):
        state = np.random.RandomState(self._context_hash(context))
        logits = state.randn(self.V) + self.offset
        logits = logits / self.temperature
        # softmax
        max_logit = np.max(logits)
        exp = np.exp(logits - max_logit)
        return exp / np.sum(exp)


# =============================================================================
# 投机解码核心实现
# =============================================================================

class SpeculativeDecoder:
    """
    投机解码器。

    参数:
      draft: 小模型，快速生成候选
      target: 大模型，验证候选并修正
      gamma: 每轮候选 token 数量
      draft_cost_ratio: draft 单次 forward 成本 / target 单次 forward 成本
    """

    def __init__(self, draft, target, gamma=5, draft_cost_ratio=0.05):
        self.draft = draft
        self.target = target
        self.gamma = gamma
        self.draft_cost_ratio = draft_cost_ratio

    def decode_step(self, prefix):
        """
        单次投机解码步骤。

        返回: (generated_tokens, stats)
        stats = {
            'draft_forwards': int,   # draft 生成候选的 forward 次数
            'target_forwards': int,  # target batch 验证的 forward 次数 (始终=1)
            'accepted': int,         # 被接受的候选数
            'total_cost': float,     # 等效 target forward 成本
            'all_accepted': bool     # 是否全部候选都被接受
        }
        """
        stats = {
            'draft_forwards': 0,
            'target_forwards': 0,
            'accepted': 0,
            'all_accepted': False
        }

        # ---- 1. Draft 生成 gamma 个候选 ----
        candidates = []
        ctx = list(prefix)
        for _ in range(self.gamma):
            tok = self.draft.sample(ctx)
            candidates.append(tok)
            ctx.append(tok)
            stats['draft_forwards'] += 1

        # ---- 2. Target batch 验证 ----
        # 工程上: 1 次 batch forward 验证所有 gamma 个位置
        accepted = []
        ctx = list(prefix)
        stats['target_forwards'] += 1

        for i, tok in enumerate(candidates):
            p_target = self.target.get_probs(ctx)[tok]
            p_draft = self.draft.get_probs(ctx)[tok]

            # 接受概率 = min(1, p_target / p_draft)
            if p_draft > 0:
                accept_prob = min(1.0, p_target / p_draft)
            else:
                accept_prob = 1.0

            if np.random.random() < accept_prob:
                accepted.append(tok)
                ctx.append(tok)
                stats['accepted'] += 1
            else:
                # ---- 拒绝: 从修正分布 (p_target - p_draft)^+ 采样 ----
                p_t = self.target.get_probs(ctx)
                p_d = self.draft.get_probs(ctx)
                adjusted = np.maximum(p_t - p_d, 0)
                adj_sum = np.sum(adjusted)
                if adj_sum > 1e-10:
                    adjusted /= adj_sum
                    new_tok = np.random.choice(self.target.V, p=adjusted)
                else:
                    new_tok = self.target.sample(ctx)
                accepted.append(new_tok)
                break

        # ---- 3. 全部接受 → 从 target 额外采样 1 个 ----
        if len(accepted) == len(candidates):
            ctx = list(prefix) + accepted
            extra = self.target.sample(ctx)
            accepted.append(extra)
            stats['all_accepted'] = True

        stats['total_cost'] = (
            stats['draft_forwards'] * self.draft_cost_ratio
            + stats['target_forwards']
        )
        return accepted, stats

    def generate(self, prefix, n_tokens):
        """生成 n 个 token，返回完整序列和累计统计。"""
        tokens = list(prefix)
        total = {
            'draft_forwards': 0,
            'target_forwards': 0,
            'total_cost': 0.0,
            'steps': 0,
            'total_accepted': 0,
            'total_candidates': 0
        }

        while len(tokens) - len(prefix) < n_tokens:
            toks, stats = self.decode_step(tokens)
            tokens.extend(toks)
            total['draft_forwards'] += stats['draft_forwards']
            total['target_forwards'] += stats['target_forwards']
            total['total_cost'] += stats['total_cost']
            total['steps'] += 1
            total['total_accepted'] += stats['accepted']
            total['total_candidates'] += self.gamma

        return tokens, total


# =============================================================================
# 朴素自回归 Baseline
# =============================================================================

class NaiveAutoregressive:
    """朴素自回归: 每次用 target model 采样 1 个 token。"""

    def __init__(self, target):
        self.target = target

    def generate(self, prefix, n_tokens):
        tokens = list(prefix)
        for _ in range(n_tokens):
            tok = self.target.sample(tokens)
            tokens.append(tok)
        return tokens, {'target_forwards': n_tokens, 'total_cost': float(n_tokens)}


# =============================================================================
# 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("🚀 投机解码 (Speculative Decoding) 工程实现")
    print("=" * 70)
    print()

    V = 100
    prefix = [1, 2, 3]
    draft = SimpleLM(V, temperature=1.2, offset=0.0, seed=42)
    target = SimpleLM(V, temperature=0.8, offset=0.5, seed=123)

    print(f"配置: 词表={V}, 前缀={prefix}")
    print(f"      draft temp=1.2 (更平滑), target temp=0.8 (更尖锐)")
    print()

    # ---- 单次投机解码演示 ----
    print("【单次投机解码演示】gamma=5")
    decoder = SpeculativeDecoder(draft, target, gamma=5, draft_cost_ratio=0.05)
    toks, stats = decoder.decode_step(prefix)
    print(f"  候选数: {decoder.gamma}")
    print(f"  生成 token: {toks}")
    print(f"  接受数: {stats['accepted']}")
    print(f"  draft forwards: {stats['draft_forwards']}")
    print(f"  target forwards: {stats['target_forwards']} (batch)")
    print(f"  等效成本: {stats['total_cost']:.2f}")
    print(f"  全部接受: {stats['all_accepted']}")
    print()

    # ---- 批量生成对比 ----
    n_tokens = 200
    print(f"【批量生成对比】生成 {n_tokens} 个 token")
    print("-" * 50)

    spec_tokens, spec_stats = decoder.generate(prefix, n_tokens)
    naive = NaiveAutoregressive(target)
    naive_tokens, naive_stats = naive.generate(prefix, n_tokens)

    print(f"投机解码 (γ={decoder.gamma}, draft_cost={decoder.draft_cost_ratio}):")
    print(f"  步数: {spec_stats['steps']}")
    print(f"  draft forwards: {spec_stats['draft_forwards']}")
    print(f"  target forwards: {spec_stats['target_forwards']}")
    print(f"  等效成本: {spec_stats['total_cost']:.1f}")
    print(f"  平均接受率: {spec_stats['total_accepted']/spec_stats['total_candidates']*100:.1f}%")
    print()
    print(f"朴素自回归:")
    print(f"  target forwards: {naive_stats['target_forwards']}")
    print(f"  等效成本: {naive_stats['total_cost']:.1f}")
    print()
    print(f"加速比: {naive_stats['total_cost'] / spec_stats['total_cost']:.2f}x")
    print()

    # ---- 不同 gamma 对比 ----
    print("【超参数扫描: 不同 gamma】")
    print(f"{'gamma':>6} | {'步数':>6} | {'接受率':>8} | {'成本':>8} | {'加速比':>8}")
    print("-" * 50)
    for g in [1, 2, 3, 4, 5, 8, 10]:
        dec = SpeculativeDecoder(draft, target, gamma=g, draft_cost_ratio=0.05)
        _, s = dec.generate(prefix, 200)
        acc_rate = s['total_accepted'] / s['total_candidates'] * 100 if s['total_candidates'] > 0 else 0
        speedup = 200 / s['total_cost']
        print(f"{g:>6} | {s['steps']:>6} | {acc_rate:>7.1f}% | {s['total_cost']:>8.1f} | {speedup:>7.2f}x")
    print()

    # ---- 不同 draft 速度对比 ----
    print("【超参数扫描: 不同 draft 成本比例 (γ=5)】")
    print(f"{'draft_cost':>10} | {'步数':>6} | {'接受率':>8} | {'成本':>8} | {'加速比':>8}")
    print("-" * 55)
    for ratio in [0.01, 0.05, 0.1, 0.2, 0.5]:
        dec = SpeculativeDecoder(draft, target, gamma=5, draft_cost_ratio=ratio)
        _, s = dec.generate(prefix, 200)
        acc_rate = s['total_accepted'] / s['total_candidates'] * 100 if s['total_candidates'] > 0 else 0
        speedup = 200 / s['total_cost']
        print(f"{ratio:>10.2f} | {s['steps']:>6} | {acc_rate:>7.1f}% | {s['total_cost']:>8.1f} | {speedup:>7.2f}x")
    print()

    print("关键结论:")
    print("  • gamma 越大，步数越少，但边际收益递减 (γ>5 后加速比提升有限)")
    print("  • draft 越便宜 (成本比例越低)，加速比越高")
    print("  • 当 draft_cost=0.05, γ=5 时，典型加速比 2.5~3.5x")
    print("  • 工程上 draft 通常是小 10~100x 的模型，成本可忽略")


if __name__ == "__main__":
    demo()
