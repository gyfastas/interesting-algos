"""
GRPO (Group Relative Policy Optimization) — 最小示例

核心: 去掉 value function，用同一 prompt 的多个采样的组内相对排名作为 advantage
来源: DeepSeek-Math / DeepSeek-R1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=-1)


def compute_grpo_loss(
    policy: PolicyModel,
    ref_policy: PolicyModel,        # 参考模型 (frozen)
    prompts: torch.Tensor,          # (B, S_prompt)
    group_responses: torch.Tensor,  # (B, G, S_resp) — 每个 prompt G 个采样
    group_rewards: torch.Tensor,    # (B, G) — 每个响应的 reward
    clip_eps: float = 0.2,
    kl_coeff: float = 0.04,
):
    """
    GRPO 损失函数。

    关键区别 vs PPO:
    1. 没有 value model — advantage 直接用组内 reward 的 z-score
    2. 每个 prompt 采样 G 个响应，组内相对排名
    3. KL 惩罚逐 token 计算（而不是序列级）
    """
    B, G, S = group_responses.shape

    # 1. 组内归一化 reward → advantage（无需 value function！）
    #    A_i = (r_i - mean(r)) / std(r)  在同一 prompt 的 G 个样本���
    group_mean = group_rewards.mean(dim=-1, keepdim=True)
    group_std = group_rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
    advantages = (group_rewards - group_mean) / group_std   # (B, G)

    # 2. 计算 policy 和 ref_policy 的 log-prob
    responses_flat = group_responses.view(B * G, S)         # (B*G, S)

    curr_log_probs = policy(responses_flat)                  # (B*G, S, V)
    curr_action_lp = curr_log_probs.gather(
        2, responses_flat.unsqueeze(2)
    ).squeeze(2)                                             # (B*G, S)

    with torch.no_grad():
        ref_log_probs = ref_policy(responses_flat)
        ref_action_lp = ref_log_probs.gather(
            2, responses_flat.unsqueeze(2)
        ).squeeze(2)                                         # (B*G, S)

    # 3. 序列级 log-prob
    curr_seq_lp = curr_action_lp.sum(dim=-1).view(B, G)     # (B, G)
    ref_seq_lp = ref_action_lp.sum(dim=-1).view(B, G)       # (B, G)

    # 4. PPO-style clipped ratio（用 ref_policy 作为 π_old）
    ratio = torch.exp(curr_seq_lp - ref_seq_lp.detach())
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 5. 逐 token KL 惩罚
    #    KL = Σ_t [π_ref(a_t|s_t) - π_θ(a_t|s_t) + π_θ/π_ref - 1]
    #    简化为: mean(ref_lp - curr_lp)
    kl_per_token = (ref_action_lp - curr_action_lp).mean()

    total_loss = policy_loss + kl_coeff * kl_per_token
    return total_loss, {
        'policy_loss': policy_loss.item(),
        'kl': kl_per_token.item(),
        'advantage_std': advantages.std().item(),
        'ratio_mean': ratio.mean().item(),
    }


def demo():
    torch.manual_seed(42)
    V, B, G, S = 100, 4, 8, 6   # G=8 个采样 per prompt

    policy = PolicyModel(V)
    ref_policy = PolicyModel(V)
    ref_policy.load_state_dict(policy.state_dict())  # 初始化相同
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 模拟数据
    prompts = torch.randint(0, V, (B, 4))
    group_responses = torch.randint(0, V, (B, G, S))
    # 模拟 reward: 每组内有好有坏
    group_rewards = torch.randn(B, G)

    print("=" * 50)
    print(f"GRPO 最小示例 (G={G} samples per prompt)")
    print("=" * 50)

    for step in range(5):
        loss, info = compute_grpo_loss(
            policy, ref_policy,
            prompts, group_responses, group_rewards,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step}: loss={loss.item():.4f}, "
              f"policy={info['policy_loss']:.4f}, "
              f"kl={info['kl']:.4f}, "
              f"adv_std={info['advantage_std']:.4f}")


if __name__ == "__main__":
    demo()
