"""
最大矩形 (Maximal Rectangle)
=============================

LeetCode 85: 给定一个只包含 '0' 和 '1' 的二维矩阵，
找出只包含 '1' 的最大矩形，返回其面积。

核心思路: 逐行转化为柱状图 + 单调栈求最大矩形。
"""

import random
import time


# =============================================================================
# 子问题: LeetCode 84 柱状图中最大的矩形
# =============================================================================

def largest_rectangle_area(heights):
    """
    单调栈解法: 给定 n 个柱子的高度，求能构成的最大矩形面积。

    原理:
      维护一个高度单调递增的栈，栈中存索引。
      当遇到比栈顶矮的柱子时，弹出栈顶计算面积:
        - 高度 = 弹出的柱子高度
        - 宽度 = 当前索引 - 新栈顶索引 - 1
        (新栈顶是第一个比弹出柱子矮的柱子)
      末尾补 0，确保所有柱子都被弹出处理。
    """
    stack = []
    max_area = 0
    n = len(heights)

    for i in range(n + 1):
        h = heights[i] if i < n else 0  # 末尾补 0
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            # 宽度: 如果栈空，说明左边没有更矮的，宽度为 i
            #       否则宽度为 i - stack[-1] - 1
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area


# =============================================================================
# 主问题: LeetCode 85 最大矩形
# =============================================================================

def maximal_rectangle(matrix):
    """
    逐行转化为柱状图，对每行调用 largest_rectangle_area。

    heights[j] = 当前行第 j 列向上连续的 '1' 的个数。
    如果 matrix[i][j] == '0'，则 heights[j] = 0 (柱子断裂)。
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        area = largest_rectangle_area(heights)
        max_area = max(max_area, area)

    return max_area


# =============================================================================
# DP 解法 (另一种思路)
# =============================================================================

def maximal_rectangle_dp(matrix):
    """
    DP 解法: 逐行维护每个位置的高度、左边界、右边界。

    对于每个位置 (i,j):
      height[j]  = 向上连续 '1' 的个数
      left[j]    = 当前行中，以 j 为右边界的最左位置
      right[j]   = 当前行中，以 j 为左边界的最右位置

    当前行以 (i,j) 为右下角的最大矩形面积 = height[j] * (right[j] - left[j])
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    height = [0] * n
    left = [0] * n
    right = [n] * n
    max_area = 0

    for i in range(m):
        cur_left = 0
        cur_right = n

        # 更新 height
        for j in range(n):
            height[j] = height[j] + 1 if matrix[i][j] == '1' else 0

        # 更新 left (从左往右)
        for j in range(n):
            if matrix[i][j] == '1':
                left[j] = max(left[j], cur_left)
            else:
                left[j] = 0
                cur_left = j + 1

        # 更新 right (从右往左)
        for j in range(n - 1, -1, -1):
            if matrix[i][j] == '1':
                right[j] = min(right[j], cur_right)
            else:
                right[j] = n
                cur_right = j

        # 计算面积
        for j in range(n):
            if matrix[i][j] == '1':
                area = height[j] * (right[j] - left[j])
                max_area = max(max_area, area)

    return max_area


# =============================================================================
# 暴力解法 (用于小数据验证)
# =============================================================================

def maximal_rectangle_brute(matrix):
    """O(m^2 * n^2) 暴力枚举所有子矩阵。"""
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])

    # 二维前缀和
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            prefix[i + 1][j + 1] = (prefix[i][j + 1] + prefix[i + 1][j]
                                    - prefix[i][j] + int(matrix[i][j]))

    def get_sum(r1, c1, r2, c2):
        return (prefix[r2 + 1][c2 + 1] - prefix[r1][c2 + 1]
                - prefix[r2 + 1][c1] + prefix[r1][c1])

    max_area = 0
    for r1 in range(m):
        for c1 in range(n):
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if area <= max_area:
                        continue
                    if get_sum(r1, c1, r2, c2) == area:
                        max_area = area

    return max_area


# =============================================================================
# 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("📐 最大矩形 — LeetCode 85")
    print("=" * 70)
    print()

    # 经典测试用例
    test_cases = [
        [['1', '0', '1', '0', '0'],
         ['1', '0', '1', '1', '1'],
         ['1', '1', '1', '1', '1'],
         ['1', '0', '0', '1', '0']],
        [['0']],
        [['1']],
        [['1', '0'], ['1', '0']],
        [['1', '1'], ['1', '1']],
        [['0', '0', '0'],
         ['0', '1', '0'],
         ['0', '0', '0']],
    ]

    print("【经典测试用例】")
    print(f"{'测试':>4} | {'矩阵大小':>10} | {'单调栈':>8} | {'DP':>8} | {'暴力':>8}")
    print("-" * 50)
    for idx, matrix in enumerate(test_cases):
        m, n = len(matrix), len(matrix[0])
        ans1 = maximal_rectangle(matrix)
        ans2 = maximal_rectangle_dp(matrix)
        brute = maximal_rectangle_brute(matrix) if m * n <= 25 else '--'
        ok = '✓' if ans1 == ans2 == brute or brute == '--' else '✗'
        print(f"{idx + 1:>4} | {m}x{n:>7} | {ans1:>8} | {ans2:>8} | {brute:>8} {ok}")
    print()

    # 随机小矩阵验证
    print("【随机小矩阵验证】")
    for _ in range(20):
        m, n = random.randint(2, 5), random.randint(2, 5)
        matrix = [[str(random.randint(0, 1)) for _ in range(n)] for _ in range(m)]
        ans1 = maximal_rectangle(matrix)
        ans2 = maximal_rectangle_dp(matrix)
        brute = maximal_rectangle_brute(matrix)
        if ans1 != brute or ans2 != brute:
            print(f"  ✗ 失败: {m}x{n}")
            for row in matrix:
                print(f"    {row}")
            break
    else:
        print("  ✓ 20 组随机测试全部通过")
    print()

    # 性能测试
    print("【性能测试】")
    sizes = [(50, 50), (100, 100), (200, 200), (300, 300)]
    print(f"{'矩阵大小':>10} | {'单调栈(ms)':>12} | {'DP(ms)':>10}")
    print("-" * 40)
    for m, n in sizes:
        matrix = [[str(random.randint(0, 1)) for _ in range(n)] for _ in range(m)]

        t0 = time.time()
        a1 = maximal_rectangle(matrix)
        t1 = time.time()

        t2 = time.time()
        a2 = maximal_rectangle_dp(matrix)
        t3 = time.time()

        print(f"{m}x{n:>7} | {(t1-t0)*1000:>10.2f} | {(t3-t2)*1000:>8.2f}")
    print()

    # 可视化一个例子
    print("【可视化示例】")
    matrix = [['1', '0', '1', '0', '0'],
              ['1', '0', '1', '1', '1'],
              ['1', '1', '1', '1', '1'],
              ['1', '0', '0', '1', '0']]
    m, n = len(matrix), len(matrix[0])
    print(f"  矩阵 ({m}x{n}):")
    for row in matrix:
        print(f"    {' '.join(row)}")
    print()

    heights = [0] * n
    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        print(f"  第{i + 1}行 heights: {heights}")
        area = largest_rectangle_area(heights)
        print(f"    该行最大矩形面积: {area}")
    print(f"  全局最大矩形面积: {maximal_rectangle(matrix)}")


if __name__ == "__main__":
    demo()
