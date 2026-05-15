"""
LeetCode 4: 寻找两个有序数组的中位数
========================================

Part A: 两个有序数组 —— 二分查找分割点（Partition）
Part B: K 个有序数组 —— 值域二分 & 最小堆

将数组在某处"切开"，使得左半部分的所有元素 <= 右半部分的所有元素。
对于 K 个数组，核心思想相同：找到一条全局分割线。
"""

import random
import time
import heapq
from bisect import bisect_right


# =============================================================================
# PART A: 两个有序数组
# =============================================================================

def find_median(nums1: list[int], nums2: list[int]) -> float:
    """
    二分查找法求两个有序数组的中位数。
    时间复杂度: O(log(min(m,n)))
    空间复杂度: O(1)
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    total_left = (m + n + 1) // 2

    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = total_left - i

        nums1_left_max = float('-inf') if i == 0 else nums1[i - 1]
        nums1_right_min = float('inf') if i == m else nums1[i]
        nums2_left_max = float('-inf') if j == 0 else nums2[j - 1]
        nums2_right_min = float('inf') if j == n else nums2[j]

        if nums1_left_max <= nums2_right_min and nums2_left_max <= nums1_right_min:
            if (m + n) % 2 == 1:
                return float(max(nums1_left_max, nums2_left_max))
            else:
                return (max(nums1_left_max, nums2_left_max) +
                        min(nums1_right_min, nums2_right_min)) / 2.0
        elif nums1_left_max > nums2_right_min:
            hi = i - 1
        else:
            lo = i + 1

    raise ValueError("输入数组未排序或格式错误")


def brute_force_median(nums1: list[int], nums2: list[int]) -> float:
    """
    暴力解法一：合并后排序求中位数。
    时间复杂度: O((m+n) log(m+n))
    空间复杂度: O(m+n)
    """
    merged = sorted(nums1 + nums2)
    L = len(merged)
    if L % 2 == 1:
        return float(merged[L // 2])
    else:
        return (merged[L // 2 - 1] + merged[L // 2]) / 2.0


def linear_merge_median(nums1: list[int], nums2: list[int]) -> float:
    """
    暴力解法二：双指针线性合并，只遍历到中位数位置。
    时间复杂度: O(m+n)
    空间复杂度: O(1)
    """
    m, n = len(nums1), len(nums2)
    total = m + n
    target = total // 2
    i = j = 0
    prev = curr = 0

    for _ in range(target + 1):
        prev = curr
        if i < m and (j >= n or nums1[i] <= nums2[j]):
            curr = nums1[i]
            i += 1
        else:
            curr = nums2[j]
            j += 1

    if total % 2 == 1:
        return float(curr)
    else:
        return (prev + curr) / 2.0


# =============================================================================
# PART B: K 个有序数组
# =============================================================================

def k_median_brute(arrays: list[list[int]]) -> float:
    """
    K 个有序数组的暴力解法：全部合并后排序。
    时间复杂度: O(N log N)，N 为总元素数
    空间复杂度: O(N)
    """
    merged = []
    for arr in arrays:
        merged.extend(arr)
    merged.sort()
    L = len(merged)
    if L % 2 == 1:
        return float(merged[L // 2])
    else:
        return (merged[L // 2 - 1] + merged[L // 2]) / 2.0


def k_median_heap(arrays: list[list[int]]) -> float:
    """
    K 个有序数组的最小堆解法：K 路归并，走到中位数位置停止。
    时间复杂度: O((N/2) log K) ≈ O(N log K)，N 为总元素数
    空间复杂度: O(K)
    """
    total = sum(len(arr) for arr in arrays)
    if total == 0:
        raise ValueError("所有数组为空")

    target = total // 2
    # 堆元素: (值, 数组索引, 元素索引)
    heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))

    prev = curr = 0
    for _ in range(target + 1):
        prev = curr
        curr, arr_idx, elem_idx = heapq.heappop(heap)
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))

    if total % 2 == 1:
        return float(curr)
    else:
        return (prev + curr) / 2.0


def k_median_binary_search(arrays: list[list[int]]) -> float:
    """
    K 个有序数组的值域二分解法。

    核心思想：
    中位数是第 (total//2 + 1) 小的元素（奇数）
    或第 total//2 和 total//2+1 小的元素的平均（偶数）。

    我们在值域 [min, max] 上二分猜测一个值 mid，
    对每个有序数组用二分查找统计 <= mid 的元素个数。
    如果总数 >= k，说明第 k 小的元素 <= mid，向左收缩；否则向右扩张。

    时间复杂度: O(K * log M * log W)
      - K: 数组个数
      - M: 单个数组平均长度
      - W: 值域范围 (max - min)
    空间复杂度: O(1)
    """
    # 过滤空数组
    arrays = [arr for arr in arrays if arr]
    if not arrays:
        raise ValueError("所有数组为空")

    total = sum(len(arr) for arr in arrays)
    low = min(arr[0] for arr in arrays)
    high = max(arr[-1] for arr in arrays)

    def find_kth(k: int) -> int:
        """找到第 k 小的元素（1-indexed）。"""
        lo, hi = low, high
        while lo < hi:
            mid = (lo + hi) // 2
            # 统计所有数组中 <= mid 的元素个数
            cnt = sum(bisect_right(arr, mid) for arr in arrays)
            if cnt < k:
                lo = mid + 1
            else:
                hi = mid
        return lo

    if total % 2 == 1:
        return float(find_kth(total // 2 + 1))
    else:
        left = find_kth(total // 2)
        right = find_kth(total // 2 + 1)
        return (left + right) / 2.0


# =============================================================================
# 验证与测试
# =============================================================================

def random_test_k(rounds: int = 3000, max_k: int = 8, max_len: int = 30, max_val: int = 200):
    """
    Monte Carlo 随机测试 K 个有序数组的三种解法。
    """
    print(f"\n开始 K 数组 Monte Carlo 随机测试：{rounds} 轮")
    print(f"参数: K=2~{max_k}, 每个数组长度 0~{max_len}")
    passed = 0
    failed_cases = []

    for r in range(rounds):
        k = random.randint(2, max_k)
        arrays = []
        for _ in range(k):
            length = random.randint(0, max_len)
            arr = sorted([random.randint(-max_val, max_val) for _ in range(length)])
            arrays.append(arr)

        # 确保非空
        if all(len(arr) == 0 for arr in arrays):
            arrays[0] = [random.randint(-max_val, max_val)]

        try:
            val_brute = k_median_brute(arrays)
            val_heap = k_median_heap(arrays)
            val_bs = k_median_binary_search(arrays)
            if abs(val_brute - val_heap) < 1e-9 and abs(val_brute - val_bs) < 1e-9:
                passed += 1
            else:
                failed_cases.append((arrays, val_brute, val_heap, val_bs))
        except Exception as e:
            failed_cases.append((arrays, str(e), "N/A", "N/A"))

    print(f"测试结果：通过 {passed}/{rounds} 轮")
    if failed_cases:
        print(f"失败样例数：{len(failed_cases)}")
        for case in failed_cases[:3]:
            print(f"  arrays={case[0]}")
            print(f"    暴力={case[1]}, 堆={case[2]}, 值域二分={case[3]}")
    else:
        print("全部通过！")
    return len(failed_cases) == 0


def benchmark_k():
    """
    K 个有序数组性能对比。
    """
    print("\n" + "=" * 80)
    print("K 个有序数组性能对比（单位：毫秒）")
    print("=" * 80)
    print(f"{'K':>3} | {'每组长':>6} | {'总元素':>8} | {'暴力排序':>10} | {'最小堆':>10} | {'值域二分':>10}")
    print("-" * 80)

    configs = [
        (3, 500),
        (5, 500),
        (10, 500),
        (20, 500),
        (50, 200),
        (100, 100),
    ]

    for k, each_len in configs:
        arrays = []
        for _ in range(k):
            arr = sorted([random.randint(-10**6, 10**6) for _ in range(each_len)])
            arrays.append(arr)
        total = k * each_len

        # 暴力
        t0 = time.perf_counter()
        k_median_brute(arrays)
        t_brute = (time.perf_counter() - t0) * 1000

        # 最小堆
        t0 = time.perf_counter()
        k_median_heap(arrays)
        t_heap = (time.perf_counter() - t0) * 1000

        # 值域二分
        t0 = time.perf_counter()
        k_median_binary_search(arrays)
        t_bs = (time.perf_counter() - t0) * 1000

        print(f"{k:>3} | {each_len:>6,} | {total:>8,} | {t_brute:>10.3f} | {t_heap:>10.3f} | {t_bs:>10.3f}")

    print("-" * 80)


def benchmark_two():
    """
    两个有序数组性能对比（三种方法）。
    """
    print("\n" + "=" * 70)
    print("两个有序数组性能对比（单位：毫秒）")
    print("=" * 70)
    print(f"{'规模':>10} | {'暴力合并排序':>14} | {'双指针线性':>12} | {'二分最优':>10}")
    print("-" * 70)

    for size in [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]:
        arr1 = sorted([random.randint(-10**9, 10**9) for _ in range(size)])
        arr2 = sorted([random.randint(-10**9, 10**9) for _ in range(size)])

        t0 = time.perf_counter()
        brute_force_median(arr1, arr2)
        t_brute = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        linear_merge_median(arr1, arr2)
        t_linear = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        find_median(arr1, arr2)
        t_fast = (time.perf_counter() - t0) * 1000

        print(f"{size:>10,} | {t_brute:>14.3f} | {t_linear:>12.3f} | {t_fast:>10.3f}")

    print("-" * 70)


def demo():
    """打印典型样例。"""
    print("=" * 60)
    print("Part A: 两个有序数组")
    print("=" * 60)
    cases = [
        ([1, 3], [2]),
        ([1, 2], [3, 4]),
        ([], [1]),
        ([2], []),
        ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12]),
    ]
    for nums1, nums2 in cases:
        result = find_median(nums1, nums2)
        expected = brute_force_median(nums1, nums2)
        status = "✓" if abs(result - expected) < 1e-9 else "✗"
        print(f"\nnums1 = {nums1}")
        print(f"nums2 = {nums2}")
        print(f"  中位数 = {result}  {status}")

    print("\n" + "=" * 60)
    print("Part B: K 个有序数组 (K=3)")
    print("=" * 60)
    k_cases = [
        [[1, 5, 9], [2, 6, 10], [3, 7, 11]],
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [[-5, 3, 6, 12], [0, 4, 10], [-2, 8]],
    ]
    for arrays in k_cases:
        brute = k_median_brute(arrays)
        heap = k_median_heap(arrays)
        bs = k_median_binary_search(arrays)
        ok = abs(brute - heap) < 1e-9 and abs(brute - bs) < 1e-9
        print(f"\narrays = {arrays}")
        print(f"  中位数 = {bs}  {'✓' if ok else '✗'}")


if __name__ == "__main__":
    demo()

    print("\n" + "=" * 60)
    print("随机测试")
    print("=" * 60)

    # 两个数组的随机测试
    print("\n[两个数组] 5000 轮随机测试...")
    passed_two = 0
    for _ in range(5000):
        len1 = random.randint(0, 50)
        len2 = random.randint(0, 50)
        if len1 == 0 and len2 == 0:
            len1 = 1
        arr1 = sorted([random.randint(-200, 200) for _ in range(len1)])
        arr2 = sorted([random.randint(-200, 200) for _ in range(len2)])
        if abs(find_median(arr1, arr2) - brute_force_median(arr1, arr2)) < 1e-9:
            passed_two += 1
    print(f"通过 {passed_two}/5000 轮")

    # K 个数组的随机测试
    success_k = random_test_k(rounds=3000, max_k=8, max_len=30, max_val=200)

    # 性能对比
    benchmark_two()
    benchmark_k()

    exit(0 if (passed_two == 5000 and success_k) else 1)
