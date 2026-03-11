"""
LLM 生成中的 Beam Search（束搜索）

问题：在自回归语言模型生成文本时，如何找到全局最优（联合概率最高）的序列？

核心矛盾：
  - Greedy（贪心）：每步选最高概率的 token → 快但容易陷入局部最优
  - Beam Search：每步保留 k 条最优前缀 → 在搜索空间和质量之间取得平衡

演示结果（toy LM）：
  Greedy:    "The small mouse"   P = 0.2808
  Beam k=2:  "A legendary hero"  P = 0.4039  ← 更优！
"""

import math
import heapq
from typing import Dict, List, Tuple, Optional

# ── Toy 语言模型 ──────────────────────────────────────────────
# 模拟 LLM 的条件概率分布：P(next_token | context)
# 设计目的：展示 greedy 在第一步选 "The"(0.52) 后陷入局部最优，
#           而 beam search 发现了 "A"(0.48) → "legendary"(0.85) → "hero"(0.99) 的最优路径

TOY_LM: Dict[str, Dict[str, float]] = {
    '<BOS>':           {'The': 0.52, 'A': 0.48},
    'The':             {'small': 0.60, 'big': 0.40},
    'A':               {'quick': 0.15, 'legendary': 0.85},
    'The small':       {'mouse': 0.90, 'cat': 0.10},
    'The big':         {'house': 0.80, 'dog': 0.20},
    'A quick':         {'fox': 0.95, 'bird': 0.05},
    'A legendary':     {'hero': 0.99, 'fail': 0.01},
    # 终止状态
    'The small mouse': {'<EOS>': 1.0},
    'The small cat':   {'<EOS>': 1.0},
    'The big house':   {'<EOS>': 1.0},
    'The big dog':     {'<EOS>': 1.0},
    'A quick fox':     {'<EOS>': 1.0},
    'A quick bird':    {'<EOS>': 1.0},
    'A legendary hero':{'<EOS>': 1.0},
    'A legendary fail':{'<EOS>': 1.0},
}

EOS = '<EOS>'
BOS = '<BOS>'


def get_next_probs(lm: Dict, seq: List[str]) -> Dict[str, float]:
    """查询语言模型：给定已生成序列，返回下一个 token 的概率分布"""
    ctx = ' '.join(seq) if seq else BOS
    return lm.get(ctx, {EOS: 1.0})


# ── 算法一：Greedy Search（贪心搜索）─────────────────────────
def greedy_search(lm: Dict, max_steps: int = 20) -> Tuple[List[str], float]:
    """
    每步选条件概率最高的 token。

    时间复杂度：O(V × T)，V=词表大小，T=序列长度
    空间复杂度：O(T)

    缺点：第一步选了局部最优 "The"，之后再无法切换到更好的 "A" 路径。
    """
    seq = []
    log_prob = 0.0

    for _ in range(max_steps):
        probs = get_next_probs(lm, seq)
        best_token = max(probs, key=probs.get)
        log_prob += math.log(probs[best_token])

        if best_token == EOS:
            break
        seq.append(best_token)

    return seq, log_prob


# ── 算法二：Beam Search（束搜索）────────────────────────────
def beam_search(
    lm: Dict,
    beam_width: int = 2,
    max_steps: int = 20
) -> List[Tuple[List[str], float]]:
    """
    维护 beam_width 条当前最优前缀，每步展开所有可能 token 后取 top-k。

    数据结构：
        beams = [(log_prob, seq), ...]  # 当前活跃的 beam
        finished = [(log_prob, seq), ...]  # 已到达 EOS 的完整序列

    时间复杂度：O(k × V × T)
    空间复杂度：O(k × T)

    关键思想：用 log 概率相加代替概率相乘，避免数值下溢。
    """
    # 初始：空序列，log_prob=0
    beams: List[Tuple[float, List[str]]] = [(0.0, [])]
    finished: List[Tuple[float, List[str]]] = []

    for step in range(max_steps):
        if not beams:
            break

        # 展开：每条 beam × 每个可能的下一个 token
        candidates: List[Tuple[float, List[str]]] = []

        for log_p, seq in beams:
            next_probs = get_next_probs(lm, seq)

            for token, prob in next_probs.items():
                new_log_p = log_p + math.log(prob)
                new_seq = seq + [token] if token != EOS else seq

                if token == EOS:
                    finished.append((new_log_p, seq))
                else:
                    candidates.append((new_log_p, new_seq))

        # 剪枝：只保留 top-k
        beams = sorted(candidates, key=lambda x: -x[0])[:beam_width]

    # 未结束的 beam 也算入候选
    finished.extend(beams)

    # 返回所有已完成序列，按分数降序
    return sorted(finished, key=lambda x: -x[0])


# ── 算法三：Best-First Search（最佳优先，供对比）──────────────
def best_first_search(lm: Dict, max_nodes: int = 50) -> Tuple[List[str], float]:
    """
    用优先队列（最大堆）实现真正的最优搜索。
    等价于 beam_width=∞ 的 beam search，但更节省内存。
    保证找到全局最优，代价是内存和时间不可控。
    """
    # 堆：(-log_prob, seq)，用负号实现最大堆
    heap = [(0.0, [])]
    visited = 0

    while heap and visited < max_nodes:
        neg_log_p, seq = heapq.heappop(heap)
        log_p = -neg_log_p
        visited += 1

        next_probs = get_next_probs(lm, seq)

        for token, prob in next_probs.items():
            new_log_p = log_p + math.log(prob)

            if token == EOS:
                return seq, new_log_p  # 第一个到达 EOS 的就是最优

            heapq.heappush(heap, (-new_log_p, seq + [token]))

    return [], float('-inf')


# ── 辅助：列出所有可能序列（穷举，仅用于 toy 场景对比）────────
def enumerate_all(lm: Dict, max_depth: int = 10) -> List[Tuple[List[str], float]]:
    """穷举所有完整路径（仅适用于 toy LM）"""
    results = []

    def dfs(seq, log_p):
        ctx = ' '.join(seq) if seq else BOS
        probs = lm.get(ctx, {EOS: 1.0})
        for token, p in probs.items():
            new_lp = log_p + math.log(p)
            if token == EOS or len(seq) >= max_depth:
                results.append((seq[:], new_lp))
            else:
                seq.append(token)
                dfs(seq, new_lp)
                seq.pop()

    dfs([], 0.0)
    return sorted(results, key=lambda x: -x[1])


# ── 主程序 ───────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('LLM Beam Search 演示')
    print('=' * 60)
    print()

    # 1. Greedy
    g_seq, g_score = greedy_search(TOY_LM)
    print(f'[Greedy]      "{" ".join(g_seq)}"')
    print(f'              log P = {g_score:.4f},  P = {math.exp(g_score):.4f}')
    print()

    # 2. Beam Search (多个 k)
    for k in [2, 3, 4]:
        results = beam_search(TOY_LM, beam_width=k)
        best_score, best_seq = results[0]
        print(f'[Beam k={k}]    "{" ".join(best_seq)}"')
        print(f'              log P = {best_score:.4f},  P = {math.exp(best_score):.4f}')
        if k == 2:
            print(f'              Top-{k} results:')
            for score, seq in results[:k]:
                print(f'                → "{" ".join(seq)}"  P={math.exp(score):.4f}')
    print()

    # 3. Best-First（最优基准）
    bf_seq, bf_score = best_first_search(TOY_LM)
    print(f'[Best-First]  "{" ".join(bf_seq)}"')
    print(f'              log P = {bf_score:.4f},  P = {math.exp(bf_score):.4f}  ← 全局最优')
    print()

    # 4. 全部序列排名
    print('全部可能序列（穷举）：')
    print(f'  {"序列":<30} {"log P":>8}  {"P":>8}')
    print('  ' + '-' * 50)
    for seq, score in enumerate_all(TOY_LM):
        marker = ' ← greedy' if seq == g_seq else ''
        marker = ' ← 全局最优' if seq == bf_seq else marker
        print(f'  {" ".join(seq):<30} {score:>8.4f}  {math.exp(score):>8.4f}{marker}')

    print()
    print('=' * 60)
    print('结论：')
    print(f'  Greedy 找到 P={math.exp(g_score):.4f}，比全局最优低'
          f' {(math.exp(bf_score)-math.exp(g_score))/math.exp(bf_score)*100:.1f}%')
    print(f'  Beam k=2 即可找到全局最优 P={math.exp(bf_score):.4f}')
    print('=' * 60)
