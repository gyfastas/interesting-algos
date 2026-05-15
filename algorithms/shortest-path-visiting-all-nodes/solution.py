"""
LeetCode 847: 访问所有节点的最短路径
======================================

给定一个无向连通图，n 个节点（编号 0~n-1），找到一条最短路径，
使得路径访问所有节点至少一次。可以重复访问节点和边。

核心解法：状态压缩 BFS
  状态: (mask, u)，mask 表示已访问节点集合，u 是当前节点
  转移: 从 u 走到邻居 v，新状态 (mask | (1<<v), v)
  目标: mask == (1<<n) - 1

对比方法:
  1. 状态压缩 BFS（推荐）  O(2^n * n^2)
  2. Floyd-Warshall + TSP DP  O(n^3 + 2^n * n^2)
"""

from collections import deque
import random


# =============================================================================
# 核心解法：状态压缩 BFS
# =============================================================================

def shortest_path_length_bfs(graph: list[list[int]]) -> int:
    """
    状态压缩 BFS。

    同时从所有节点出发，记录每个 (mask, node) 状态的最短距离。
    当任意状态 mask == (1<<n) - 1 时，即为答案。

    参数:
        graph: 邻接表，graph[i] 表示节点 i 的所有邻居
    返回:
        访问所有节点的最短路径长度（边数）
    """
    n = len(graph)
    if n == 1:
        return 0

    target = (1 << n) - 1  # 所有节点都被访问

    # BFS 初始化：从每个节点同时出发
    # dist[mask][u] = 从某个起点走到 u，已访问 mask 的最短距离
    # 用二维数组代替字典，更快
    dist = [[-1] * n for _ in range(1 << n)]
    q = deque()

    for u in range(n):
        mask = 1 << u
        dist[mask][u] = 0
        q.append((mask, u))

    while q:
        mask, u = q.popleft()
        d = dist[mask][u]

        # 访问所有节点？
        if mask == target:
            return d

        for v in graph[u]:
            new_mask = mask | (1 << v)
            if dist[new_mask][v] == -1:  # 未访问过
                dist[new_mask][v] = d + 1
                q.append((new_mask, v))

    return -1  # 不可能（题目保证连通）


# =============================================================================
# 暴力解法（仅用于小图验证）

# =============================================================================

def shortest_path_length_brute(graph: list[list[int]]) -> int:
    """
    暴力 BFS（小图验证用，和核心解法逻辑相同但用 set 代替数组）。
    仅适用于 n <= 8 的小图验证。
    """
    n = len(graph)
    if n == 1:
        return 0

    target = (1 << n) - 1
    visited = set()
    q = deque()

    for u in range(n):
        mask = 1 << u
        visited.add((mask, u))
        q.append((mask, u, 0))

    while q:
        mask, u, d = q.popleft()
        if mask == target:
            return d
        for v in graph[u]:
            new_mask = mask | (1 << v)
            if (new_mask, v) not in visited:
                visited.add((new_mask, v))
                q.append((new_mask, v, d + 1))

    return -1


# =============================================================================
# 辅助：生成随机连通图
# =============================================================================

def generate_connected_graph(n: int, edge_prob: float = 0.3) -> list[list[int]]:
    """生成 n 个节点的随机连通无向图（邻接表）。"""
    # 先保证连通：生成一棵树
    graph = [[] for _ in range(n)]
    for i in range(1, n):
        j = random.randint(0, i - 1)
        graph[i].append(j)
        graph[j].append(i)

    # 再随机加边
    for i in range(n):
        for j in range(i + 1, n):
            if j not in graph[i] and random.random() < edge_prob:
                graph[i].append(j)
                graph[j].append(i)

    return graph


# =============================================================================
# 变种：哈密顿路径（每个节点仅允许访问一次）
# =============================================================================

def hamiltonian_path_dp(graph: list[list[int]]) -> int:
    """
    哈密顿路径：找到一条经过每个节点恰好一次的路径，求最短长度。

    与 LeetCode 847 的区别：
      - 原题允许重复访问节点（BFS 即可）
      - 本题每个节点只能访问一次（状态压缩 DP）

    状态：dp[mask][i] = 访问了 mask 中的节点，当前在 i 的最短路径长度
    转移：dp[mask|2^j][j] = min(dp[mask][i] + 1)  for i in mask, j not in mask, edge(i,j)
    初始：dp[2^i][i] = 0
    答案：min(dp[2^n - 1][i])

    时间复杂度: O(2^n * n^2)
    空间复杂度: O(2^n * n)
    """
    n = len(graph)
    if n == 1:
        return 0

    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]

    # 初始状态：从每个节点出发
    for i in range(n):
        dp[1 << i][i] = 0

    # 按 mask 从小到大遍历
    for mask in range(1 << n):
        for i in range(n):
            if dp[mask][i] == INF:
                continue
            if not (mask & (1 << i)):
                continue  # i 不在 mask 中，非法状态
            for j in graph[i]:
                if mask & (1 << j):
                    continue  # j 已经访问过，不能重复访问
                new_mask = mask | (1 << j)
                dp[new_mask][j] = min(dp[new_mask][j], dp[mask][i] + 1)

    target = (1 << n) - 1
    ans = min(dp[target][i] for i in range(n))
    return ans if ans != INF else -1


def hamiltonian_path_brute(graph: list[list[int]]) -> int:
    """
    暴力枚举所有排列，仅用于小图验证（n <= 8）。
    """
    n = len(graph)
    if n == 1:
        return 0

    # 建立邻接矩阵
    adj = [[False] * n for _ in range(n)]
    for u in range(n):
        for v in graph[u]:
            adj[u][v] = True

    from itertools import permutations
    best = float('inf')

    for perm in permutations(range(n)):
        valid = True
        dist = 0
        for i in range(n - 1):
            if not adj[perm[i]][perm[i + 1]]:
                valid = False
                break
            dist += 1
        if valid:
            best = min(best, dist)

    return best if best != float('inf') else -1


# =============================================================================
# 验证与测试
# =============================================================================

def demo():
    print("=" * 70)
    print("LeetCode 847: 访问所有节点的最短路径")
    print("=" * 70)
    print()

    # LeetCode 官方示例
    cases = [
        # 示例 1: [[1,2,3],[0],[0],[0]] -> 4
        ([[1, 2, 3], [0], [0], [0]], 4),
        # 示例 2: [[1],[0,2,4],[1,3,4],[2],[1,2]] -> 4
        ([[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]], 4),
        # 示例 3: 单节点
        ([[]], 0),
        # 示例 4: 链状 0-1-2-3
        ([[1], [0, 2], [1, 3], [2]], 3),
        # 示例 5: 环状 0-1-2-0
        ([[1, 2], [0, 2], [0, 1]], 2),
        # 示例 6: 星形 0 为中心，5 个节点
        ([[1, 2, 3, 4], [0], [0], [0], [0]], 6),
    ]

    print("[1] 官方示例测试（允许重复访问）")
    print("-" * 60)
    for graph, expected in cases:
        result = shortest_path_length_bfs(graph)
        status = "✓" if result == expected else "✗"
        print(f"  graph={graph}")
        print(f"  结果: {result}, 期望: {expected}  {status}")
        print()

    # 小图暴力验证
    print("[2] Monte Carlo 随机验证（n≤8，BFS vs 暴力）")
    print("-" * 60)
    random.seed(42)
    passed = 0
    total = 200
    for _ in range(total):
        n = random.randint(1, 8)
        graph = generate_connected_graph(n, edge_prob=0.4)
        fast = shortest_path_length_bfs(graph)
        if n <= 8:
            slow = shortest_path_length_brute(graph)
            if fast == slow:
                passed += 1

    print(f"  测试结果：通过 {passed}/{total} 轮")
    print()

    # 性能测试
    print("[3] 性能测试（状态压缩 BFS）")
    print("-" * 60)
    print(f"{'n':>4} | {'边数':>6} | {'状态数':>10} | {'时间(ms)':>10}")
    print("-" * 45)

    import time
    for n in [4, 6, 8, 10, 12]:
        graph = generate_connected_graph(n, edge_prob=0.3)
        edges = sum(len(nei) for nei in graph) // 2

        t0 = time.perf_counter()
        shortest_path_length_bfs(graph)
        t = (time.perf_counter() - t0) * 1000

        states = (1 << n) * n
        print(f"{n:>4} | {edges:>6} | {states:>10,} | {t:>10.3f}")

    print("-" * 45)

    # 复杂度分析
    print()
    print("[4] 复杂度分析（允许重复访问）")
    print("-" * 60)
    print("  状态数: O(2^n · n)")
    print("  总时间: O(2^n · n^2)")
    print("  总空间: O(2^n · n)")
    print()

    # ========== 变种：哈密顿路径 ==========
    print("=" * 70)
    print("变种：哈密顿路径（每个节点仅允许访问一次）")
    print("=" * 70)
    print()

    # 哈密顿路径示例
    hp_cases = [
        # 链状：有哈密顿路径 0-1-2-3
        ([[1], [0, 2], [1, 3], [2]], 3),
        # 环状：有哈密顿路径 0-1-2
        ([[1, 2], [0, 2], [0, 1]], 2),
        # 星形（5节点）：没有哈密顿路径！因为从叶子出发只能到中心，然后被困
        ([[1, 2, 3, 4], [0], [0], [0], [0]], -1),
        # 完全图 K4：哈密顿路径长度 = 3
        ([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], 3),
    ]

    print("[5] 哈密顿路径示例测试")
    print("-" * 60)
    for graph, expected in hp_cases:
        result = hamiltonian_path_dp(graph)
        status = "✓" if result == expected else "✗"
        note = "（不存在哈密顿路径）" if expected == -1 else ""
        print(f"  graph={graph}")
        print(f"  结果: {result}, 期望: {expected}  {status} {note}")
        print()

    # 哈密顿路径暴力验证
    print("[6] 哈密顿路径随机验证（n≤8，DP vs 暴力排列）")
    print("-" * 60)
    passed = 0
    total = 100
    for _ in range(total):
        n = random.randint(1, 8)
        graph = generate_connected_graph(n, edge_prob=0.4)
        fast = hamiltonian_path_dp(graph)
        slow = hamiltonian_path_brute(graph)
        if fast == slow:
            passed += 1

    print(f"  测试结果：通过 {passed}/{total} 轮")
    print()

    # 对比：允许重复 vs 不允许重复
    print("[7] 对比：允许重复访问 vs 仅允许访问一次")
    print("-" * 60)
    print(f"{'图类型':>12} | {'允许重复':>10} | {'仅一次':>8} | {'说明':>20}")
    print("-" * 60)

    chain = [[1], [0, 2], [1, 3], [2]]
    star = [[1, 2, 3, 4], [0], [0], [0], [0]]
    complete = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    for name, g in [("链状", chain), ("星形", star), ("完全图 K4", complete)]:
        r1 = shortest_path_length_bfs(g)
        r2 = hamiltonian_path_dp(g)
        note = "相同" if r1 == r2 else ("哈密顿不存在" if r2 == -1 else "重复可缩短")
        print(f"{name:>12} | {r1:>10} | {r2:>8} | {note:>20}")

    print("-" * 60)
    print()
    print("  关键发现：")
    print("  • 链状/环状：两种问题的答案相同（天然不需要重复）")
    print("  • 星形图：允许重复时答案=6，不允许时答案=-1（不存在哈密顿路径）")
    print("  • 完全图：两种问题的答案相同（任意排列都是哈密顿路径）")


if __name__ == "__main__":
    demo()
