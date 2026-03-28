"""
PPO (Proximal Policy Optimization) — 最小 RLHF 示例

核心: clipped surrogate objective + value function baseline
流程: policy → 采样 → reward model 打分 → 计算 advantage → 更新 policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyModel(nn.Module):
    """简化的 policy (语言模型)，输出 token 的 log-probabilities。"""
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


class ValueModel(nn.Module):
    """价值函数 V(s)，估计当前状态的期望回报。PPO 特有。"""
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_ppo_loss(
    policy: PolicyModel,
    value_model: ValueModel,
    ref_log_probs: torch.Tensor,   # 参考模型 (SFT) 的 log-prob
    states: torch.Tensor,           # 输入 token ids: (B, S)
    actions: torch.Tensor,          # 采样的 token ids: (B, S)
    rewards: torch.Tensor,          # reward model 给的分数: (B,)
    clip_eps: float = 0.2,
    kl_coeff: float = 0.01,
    vf_coeff: float = 0.5,
):
    """
    PPO-Clip 损失函数。

    L_clip = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    其中 r_t = π_θ(a|s) / π_old(a|s) 是重要性采样比
    """
    B, S = states.shape

    # 1. 当前 policy 的 log-prob
    log_probs = policy(states)                                  # (B, S, V)
    action_log_probs = log_probs.gather(2, actions.unsqueeze(2)).squeeze(2)  # (B, S)
    curr_log_probs = action_log_probs.sum(dim=-1)               # (B,) 序列级

    # 2. 旧 policy 的 log-prob（采样时记录的）
    old_log_probs = ref_log_probs  # 简化：用 ref model 的

    # 3. Value function 估计
    values = value_model(states).mean(dim=-1)                   # (B,)

    # 4. Advantage = reward - value baseline
    advantages = rewards - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 5. PPO Clipped Surrogate Objective
    ratio = torch.exp(curr_log_probs - old_log_probs.detach())  # r_t = π/π_old
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 6. Value function loss
    value_loss = F.mse_loss(values, rewards)

    # 7. KL penalty (对参考模型)
    kl = (old_log_probs - curr_log_probs).mean()

    total_loss = policy_loss + vf_coeff * value_loss + kl_coeff * kl
    return total_loss, {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'kl': kl.item(),
        'ratio_mean': ratio.mean().item(),
    }


def demo():
    torch.manual_seed(42)
    V, B, S = 100, 4, 8

    policy = PolicyModel(V)
    value_model = ValueModel(V)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_model.parameters()), lr=1e-3
    )

    # 模拟数据
    states = torch.randint(0, V, (B, S))
    actions = torch.randint(0, V, (B, S))
    rewards = torch.randn(B)                    # reward model 的分数
    ref_log_probs = torch.randn(B) * 0.1 - 5   # 参考模型的 log-prob

    print("=" * 50)
    print("PPO 最小示例")
    print("=" * 50)

    for step in range(5):
        loss, info = compute_ppo_loss(
            policy, value_model, ref_log_probs,
            states, actions, rewards,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step}: loss={loss.item():.4f}, "
              f"policy={info['policy_loss']:.4f}, "
              f"value={info['value_loss']:.4f}, "
              f"kl={info['kl']:.4f}, "
              f"ratio={info['ratio_mean']:.4f}")


if __name__ == "__main__":
    demo()
