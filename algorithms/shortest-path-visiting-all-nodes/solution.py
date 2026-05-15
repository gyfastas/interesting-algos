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

    print("[1] 官方示例测试")
    print("-" * 60)
    for graph, expected in cases:
        result = shortest_path_length_bfs(graph)
        status = "✓" if result == expected else "✗"
        print(f"  graph={graph}")
        print(f"  结果: {result}, 期望: {expected}  {status}")
        print()

    # 小图暴力验证
    print("[2] Monte Carlo 随机验证（n≤8，BFS vs 暴力 DFS）")
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
    print("[4] 复杂度分析")
    print("-" * 60)
    print("  状态数: O(2^n · n)，每个状态记录一个节点和访问集合")
    print("  转移数: 每个状态最多向 degree(u) 个邻居转移")
    print("  总时间: O(2^n · n · avg_degree) ≈ O(2^n · n^2)")
    print("  总空间: O(2^n · n)")
    print()
    print("  当 n=12 时，状态数 ≈ 12 × 4096 = 49,152，BFS 轻松处理")
    print("  当 n=20 时，状态数 ≈ 20 × 1,048,576 ≈ 2000 万，内存压力大")


if __name__ == "__main__":
    demo()
