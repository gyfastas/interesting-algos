"""
马与4×4棋盘 — 马尔可夫链求解

问题：4×4棋盘上一匹马从任意角落出发，遵循"马走日"规则，
      求再次踏上任意角落所需的期望步数。

答案：6步
"""

import random

# ── 棋盘与走法定义 ──────────────────────────────────────────
MOVES = [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)]
CORNERS = {(0,0),(0,3),(3,0),(3,3)}

def valid_moves(r, c):
    return [(r+dr, c+dc) for dr,dc in MOVES if 0 <= r+dr <= 3 and 0 <= c+dc <= 3]

# ── 方法一：马尔可夫链线性方程组 ────────────────────────────
def solve_markov():
    """
    设 E[r,c] = 从位置 (r,c) 出发，首次到达任意角落的期望步数

    边界条件：E[corner] = 0

    递推关系（非角落格）：
        E[r,c] = 1 + (1/|neighbors|) * Σ E[neighbor]

    整理成线性方程组 AE = b 求解。
    """
    non_corners = [(r,c) for r in range(4) for c in range(4) if (r,c) not in CORNERS]
    idx = {pos: i for i, pos in enumerate(non_corners)}
    n = len(non_corners)

    # 高斯消元（不用 numpy 的纯 Python 版本）
    A = [[0.0]*n for _ in range(n)]
    b = [1.0]*n

    for i, pos in enumerate(non_corners):
        r, c = pos
        ns = valid_moves(r, c)
        deg = len(ns)
        A[i][i] = 1.0
        for nr, nc in ns:
            if (nr, nc) not in CORNERS:
                j = idx[(nr, nc)]
                A[i][j] -= 1.0 / deg

    # 高斯消元
    for col in range(n):
        # 找主元
        pivot = max(range(col, n), key=lambda row: abs(A[row][col]))
        A[col], A[pivot] = A[pivot], A[col]
        b[col], b[pivot] = b[pivot], b[col]

        for row in range(n):
            if row != col and A[row][col] != 0:
                factor = A[row][col] / A[col][col]
                for k in range(n):
                    A[row][k] -= factor * A[col][k]
                b[row] -= factor * b[col]

    E = {pos: b[i] / A[i][i] for i, pos in enumerate(non_corners)}
    for corner in CORNERS:
        E[corner] = 0.0
    return E

# ── 方法二：利用对称性手动推导（验证用）────────────────────
def solve_by_symmetry():
    """
    4×4棋盘有 D4 对称性，16个格子分为3类：

      C (corners)  : (0,0) (0,3) (3,0) (3,3)      — E_C = 0
      B (边非角落) : (0,1) (0,2) (1,0) (2,0) ...   — E_B = ?
      I (内部)     : (1,1) (1,2) (2,1) (2,2)        — E_I = ?

    从 B 代表格 (0,1) 出发：
      有效邻居：(1,3)B, (2,2)I, (2,0)B → 2个B + 1个I
      E_B = 1 + (2/3)E_B + (1/3)E_I

    从 I 代表格 (1,1) 出发：
      有效邻居：(2,3)B, (0,3)C, (3,2)B, (3,0)C → 2个B + 2个C
      E_I = 1 + (2/4)E_B + (2/4)·0 = 1 + E_B/2

    代入解方程：
      E_B = 1 + (2/3)E_B + (1/3)(1 + E_B/2)
      E_B = 4/3 + (5/6)E_B
      E_B/6 = 4/3  →  E_B = 8
      E_I = 1 + 4 = 5

    从角落 (0,0) 出发：
      有效邻居：(1,2)I, (2,1)I → 全是 I 类
      E_from_corner = 1 + E_I = 1 + 5 = 6
    """
    E_B = 8.0
    E_I = 5.0
    E_from_corner = 1 + E_I  # = 6
    return E_B, E_I, E_from_corner

# ── 方法三：Monte Carlo 模拟验证 ─────────────────────────────
def monte_carlo(start=(0,0), n_trials=1_000_000, seed=42):
    """随机模拟：从角落出发，统计首次回到任意角落的平均步数"""
    random.seed(seed)
    total_steps = 0

    for _ in range(n_trials):
        r, c = start
        steps = 0
        while True:
            ns = valid_moves(r, c)
            r, c = random.choice(ns)
            steps += 1
            if (r, c) in CORNERS:
                break
        total_steps += steps

    return total_steps / n_trials

# ── 主程序 ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("马与4×4棋盘 — 期望步数求解")
    print("=" * 50)

    # 方法一：线性方程组
    E = solve_markov()
    print("\n[方法一] 马尔可夫链线性方程组求解：")
    print("各格子到角落的期望步数：")
    for r in range(4):
        row = []
        for c in range(4):
            if (r,c) in CORNERS:
                row.append("  C ")
            else:
                row.append(f"{E[(r,c)]:4.1f}")
        print("  " + "  ".join(row))

    corner_neighbors = valid_moves(0, 0)
    e_from_corner = 1 + sum(E[n] for n in corner_neighbors) / len(corner_neighbors)
    print(f"\n从角落 (0,0) 出发的期望步数：{e_from_corner:.4f}")

    # 方法二：对称性推导
    E_B, E_I, e_sym = solve_by_symmetry()
    print(f"\n[方法二] 对称性推导：")
    print(f"  E_B（边非角格）= {E_B}")
    print(f"  E_I（内部格）  = {E_I}")
    print(f"  E_from_corner  = 1 + E_I = {e_sym}")

    # 方法三：Monte Carlo
    print("\n[方法三] Monte Carlo 模拟（100万次）：")
    mc_result = monte_carlo()
    print(f"  模拟结果 = {mc_result:.4f}")

    print("\n" + "=" * 50)
    print(f"✅ 答案：期望步数 = 6")
    print("=" * 50)
