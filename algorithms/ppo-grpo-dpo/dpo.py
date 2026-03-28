"""
DPO (Direct Preference Optimization) — 最小示例

核心: 不需要 reward model，直接从偏好对 (chosen, rejected) 优化 policy
来源: Rafailov et al., 2023 — "Direct Preference Optimization"
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


def get_seq_log_prob(model, sequences):
    """计算序列的 log-probability。"""
    log_probs = model(sequences)                              # (B, S, V)
    action_lp = log_probs.gather(2, sequences.unsqueeze(2)).squeeze(2)  # (B, S)
    return action_lp.sum(dim=-1)                              # (B,)


def compute_dpo_loss(
    policy: PolicyModel,
    ref_policy: PolicyModel,        # 参考模型 (frozen SFT model)
    chosen: torch.Tensor,           # (B, S) — 人类偏好的回答
    rejected: torch.Tensor,         # (B, S) — 人类不偏好的回答
    beta: float = 0.1,
):
    """
    DPO 损失函数 — 整个 RLHF 的核心浓缩成一个公式:

    L_DPO = -log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

    即: 让 policy 相对 ref 更偏好 chosen，更不偏好 rejected。

    β 控制偏离 ref model 的程度:
      β 大 → 保守，不偏离太远
      β 小 → 激进，更强地偏向偏好数据
    """
    # 1. 当前 policy 的 log-prob
    pi_chosen = get_seq_log_prob(policy, chosen)      # (B,)
    pi_rejected = get_seq_log_prob(policy, rejected)  # (B,)

    # 2. 参考 policy 的 log-prob
    with torch.no_grad():
        ref_chosen = get_seq_log_prob(ref_policy, chosen)
        ref_rejected = get_seq_log_prob(ref_policy, rejected)

    # 3. DPO loss — 就这一行
    #    log_ratio_chosen  = log(π_θ(y_w) / π_ref(y_w))
    #    log_ratio_rejected = log(π_θ(y_l) / π_ref(y_l))
    chosen_logratios = pi_chosen - ref_chosen
    rejected_logratios = pi_rejected - ref_rejected

    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # 隐式 reward (DPO 的理论洞察)
    chosen_reward = beta * (pi_chosen - ref_chosen).detach()
    rejected_reward = beta * (pi_rejected - ref_rejected).detach()

    return loss, {
        'loss': loss.item(),
        'chosen_reward': chosen_reward.mean().item(),
        'rejected_reward': rejected_reward.mean().item(),
        'reward_margin': (chosen_reward - rejected_reward).mean().item(),
        'accuracy': (chosen_logratios > rejected_logratios).float().mean().item(),
    }


def demo():
    torch.manual_seed(42)
    V, B, S = 100, 8, 10

    policy = PolicyModel(V)
    ref_policy = PolicyModel(V)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 模拟偏好数据
    chosen = torch.randint(0, V, (B, S))
    rejected = torch.randint(0, V, (B, S))

    print("=" * 55)
    print("DPO 最小示例")
    print("=" * 55)

    for step in range(10):
        loss, info = compute_dpo_loss(
            policy, ref_policy, chosen, rejected, beta=0.1,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step:2d}: loss={info['loss']:.4f}, "
              f"chosen_r={info['chosen_reward']:+.3f}, "
              f"rejected_r={info['rejected_reward']:+.3f}, "
              f"margin={info['reward_margin']:+.3f}, "
              f"acc={info['accuracy']:.2f}")


if __name__ == "__main__":
    demo()
