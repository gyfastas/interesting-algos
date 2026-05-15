"""
LeetCode 329: 矩阵中的最长递增路径
====================================

给定 m × n 的整数矩阵，找到最长的递增路径。
每个单元格可以向上/下/左/右移动，且只能移动到值更大的相邻单元格。

两种写法对比:
  1. @lru_cache() 装饰器版 — 简洁优雅，但有函数调用开销
  2. 手写 memo 版 — 用二维数组，速度更快，更可控
"""

import functools
import time
import random


# =============================================================================
# 写法一: @lru_cache() 装饰器版
# =============================================================================

def longest_increasing_path_lru(matrix: list[list[int]]) -> int:
    """
    使用 functools.lru_cache 自动做记忆化。

    优点: 代码极短，自动处理缓存
    缺点: 需要将 matrix 转为 tuple 才能哈希，有额外开销
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    # lru_cache 要求参数可哈希，所以 matrix 要转成 tuple
    mt = tuple(tuple(row) for row in matrix)

    @functools.lru_cache(maxsize=None)
    def dfs(i: int, j: int) -> int:
        """从 (i,j) 出发的最长递增路径长度。"""
        best = 1
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and mt[ni][nj] > mt[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        return best

    ans = 0
    for i in range(m):
        for j in range(n):
            ans = max(ans, dfs(i, j))

    dfs.cache_clear()  # 清理缓存，避免测试间干扰
    return ans


# =============================================================================
# 写法二: 手写 memo 版
# =============================================================================

def longest_increasing_path_memo(matrix: list[list[int]]) -> int:
    """
    手写记忆化，用二维数组 dp[i][j] 存储从 (i,j) 出发的最长路径。

    优点: 无装饰器开销，无哈希计算，纯数组访问极快
    缺点: 代码稍长，需手动管理 memo
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(i: int, j: int) -> int:
        """从 (i,j) 出发的最长递增路径长度。"""
        if dp[i][j] != 0:
            return dp[i][j]

        best = 1
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))

        dp[i][j] = best
        return best

    ans = 0
    for i in range(m):
        for j in range(n):
            ans = max(ans, dfs(i, j))

    return ans


# =============================================================================
# 写法三: 拓扑排序 (BFS 逐层剥离) — 另一种思路
# =============================================================================

def longest_increasing_path_topo(matrix: list[list[int]]) -> int:
    """
    拓扑排序法：从所有"局部最小值"（出度 > 0，入度 = 0）开始 BFS，
    逐层剥离，层数即为最长路径长度。

    时间: O(mn)，空间: O(mn)
    特点: 不需要递归，适合极大矩阵（避免递归深度限制）
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 计算每个点的出度（可以向几个更大的邻居走）
    outdegree = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                    outdegree[i][j] += 1

    # 初始化队列：所有出度为 0 的点（局部最大值，路径终点）
    from collections import deque
    q = deque()
    for i in range(m):
        for j in range(n):
            if outdegree[i][j] == 0:
                q.append((i, j))

    # 逆向 BFS：从大到小逐层剥离
    layers = 0
    while q:
        layers += 1
        for _ in range(len(q)):
            i, j = q.popleft()
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] < matrix[i][j]:
                    outdegree[ni][nj] -= 1
                    if outdegree[ni][nj] == 0:
                        q.append((ni, nj))

    return layers


# =============================================================================
# 验证与测试
# =============================================================================

def demo():
    print("=" * 70)
    print("LeetCode 329: 矩阵中的最长递增路径")
    print("=" * 70)
    print()

    # LeetCode 官方示例
    cases = [
        # 示例 1
        ([[9, 9, 4], [6, 6, 8], [2, 1, 1]], 4),
        # 示例 2
        ([[3, 4, 5], [3, 2, 6], [2, 2, 1]], 4),
        # 示例 3: 单元素
        ([[1]], 1),
        # 示例 4: 递增行
        ([[1, 2, 3, 4, 5]], 5),
        # 示例 5: 递减列
        ([[5], [4], [3], [2], [1]], 5),
        # 示例 6: 全部相同
        ([[7, 7, 7], [7, 7, 7], [7, 7, 7]], 1),
        # 示例 7: 蛇形
        ([[1, 2, 3], [6, 5, 4], [7, 8, 9]], 9),
    ]

    print("[1] 官方示例测试（三种方法对比）")
    print("-" * 70)
    print(f"{'矩阵':>20} | {'lru':>4} | {'memo':>4} | {'topo':>4} | {'期望':>4}")
    print("-" * 70)

    for matrix, expected in cases:
        r1 = longest_increasing_path_lru(matrix)
        r2 = longest_increasing_path_memo(matrix)
        r3 = longest_increasing_path_topo(matrix)
        ok = "✓" if r1 == r2 == r3 == expected else "✗"
        desc = str(matrix[0]) if len(matrix) <= 3 else f"{len(matrix)}×{len(matrix[0])} matrix"
        print(f"{desc:>20} | {r1:>4} | {r2:>4} | {r3:>4} | {expected:>4} {ok}")

    print("-" * 70)
    print()

    # 性能对比
    print("[2] 性能对比: lru_cache vs 手写 memo vs 拓扑排序")
    print("-" * 70)

    def make_matrix(rows, cols, max_val=1000):
        return [[random.randint(1, max_val) for _ in range(cols)] for _ in range(rows)]

    sizes = [(10, 10), (20, 20), (50, 50), (100, 100), (200, 200)]
    print(f"{'尺寸':>10} | {'lru_cache':>12} | {'手写 memo':>12} | {'拓扑排序':>12}")
    print("-" * 60)

    for rows, cols in sizes:
        mat = make_matrix(rows, cols)

        t0 = time.perf_counter()
        longest_increasing_path_lru(mat)
        t1 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        longest_increasing_path_memo(mat)
        t2 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        longest_increasing_path_topo(mat)
        t3 = (time.perf_counter() - t0) * 1000

        print(f"{rows}×{cols:>3} | {t1:>11.2f} | {t2:>11.2f} | {t3:>11.2f}")

    print("-" * 60)
    print()

    # 复杂度分析
    print("[3] 复杂度分析")
    print("-" * 60)
    print("  每个单元格只被计算一次（记忆化）")
    print("  时间: O(m × n)，每个点向 4 个方向 DFS")
    print("  空间: O(m × n)，memo 数组 + 递归栈")
    print()
    print("  lru_cache vs 手写 memo 的差异:")
    print("  • lru_cache: 需要将 matrix 转 tuple，参数哈希有开销")
    print("  • 手写 memo: 纯数组访问，无装饰器/哈希开销，通常快 2~5 倍")
    print("  • 拓扑排序: 无递归，避免 Python 递归深度限制，适合超大矩阵")
    print()

    # 代码行数对比
    print("[4] 代码简洁度对比")
    print("-" * 60)
    print("  lru_cache 版: 约 15 行核心代码（含装饰器）")
    print("  手写 memo 版: 约 25 行核心代码（手动管理 dp 数组）")
    print("  拓扑排序版: 约 35 行核心代码（BFS + 出度计算）")
    print()
    print("  结论: lru_cache 最简洁，手写 memo 最快，拓扑排序最稳")


if __name__ == "__main__":
    random.seed(42)
    demo()
