"""
快速排序 — 递归 & 非递归实现

包含: Lomuto partition / Hoare partition / 三路快排 / 非递归 / 尾递归优化
"""

import random


# ============================================================
# 1. Lomuto Partition
# ============================================================
def lomuto_partition(arr: list, lo: int, hi: int) -> int:
    """
    Lomuto partition: 以 arr[hi] 为 pivot

    维护不变量:
      arr[lo..i-1] < pivot
      arr[i..j-1] >= pivot
      arr[hi] = pivot

    最后把 pivot 放到 i 的位置
    """
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i


# ============================================================
# 2. Hoare Partition
# ============================================================
def hoare_partition(arr: list, lo: int, hi: int) -> int:
    """
    Hoare partition: 以中间元素为 pivot，两指针从两端夹逼

    比 Lomuto 更高效:
    - 交换次数更少
    - 重复元素表现更好

    注意: 返回的 j 是分界点，不是 pivot 的最终位置
    左半 arr[lo..j], 右半 arr[j+1..hi]
    """
    pivot = arr[(lo + hi) // 2]
    i, j = lo - 1, hi + 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]


# ============================================================
# 3. 递归快排 (Lomuto)
# ============================================================
def quicksort_recursive(arr: list, lo: int = 0, hi: int | None = None) -> None:
    """
    经典递归快排。

    每次:
    1. partition 把数组分成两部分
    2. 递归排左半
    3. 递归排右半

    递归深度: 期望 O(log n), 最差 O(n)
    """
    if hi is None:
        hi = len(arr) - 1
    if lo < hi:
        p = lomuto_partition(arr, lo, hi)
        quicksort_recursive(arr, lo, p - 1)
        quicksort_recursive(arr, p + 1, hi)


# ============================================================
# 4. 递归快排 (Hoare)
# ============================================================
def quicksort_hoare(arr: list, lo: int = 0, hi: int | None = None) -> None:
    """用 Hoare partition 的递归快排。"""
    if hi is None:
        hi = len(arr) - 1
    if lo < hi:
        p = hoare_partition(arr, lo, hi)
        quicksort_hoare(arr, lo, p)      # 注意: Hoare 返回 j, 左半包含 j
        quicksort_hoare(arr, p + 1, hi)


# ============================================================
# 5. 非递归快排 (显式栈)
# ============================================================
def quicksort_iterative(arr: list) -> None:
    """
    非递归快排: 用显式栈模拟递归调用栈。

    优势:
    - 不会栈溢出 (Python 默认递归限制 1000)
    - 大数据量安全
    - 可以控制处理顺序

    栈里存的就是递归函数的参数: (lo, hi)
    """
    if len(arr) <= 1:
        return

    stack = [(0, len(arr) - 1)]

    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

        # Partition
        p = lomuto_partition(arr, lo, hi)

        # 把两个子问题压栈 (相当于两次递归调用)
        # 先压大的后压小的 → 先处理小区间 → 栈深度 O(log n)
        if p - 1 - lo > hi - p - 1:
            stack.append((lo, p - 1))     # 大区间先压 (后处理)
            stack.append((p + 1, hi))     # 小区间后压 (先处理)
        else:
            stack.append((p + 1, hi))
            stack.append((lo, p - 1))


# ============================================================
# 6. 尾递归优化
# ============================================================
def quicksort_tail_optimized(arr: list, lo: int = 0, hi: int | None = None) -> None:
    """
    尾递归优化: 只递归较小的一半，循环处理较大的一半。

    保证: 栈深度严格 O(log n)，即使最差情况也不超过
    """
    if hi is None:
        hi = len(arr) - 1

    while lo < hi:
        p = lomuto_partition(arr, lo, hi)

        # 递归较小的一半，循环较大的一半
        if p - lo < hi - p:
            quicksort_tail_optimized(arr, lo, p - 1)
            lo = p + 1       # 尾递归变循环
        else:
            quicksort_tail_optimized(arr, p + 1, hi)
            hi = p - 1       # 尾递归变循环


# ============================================================
# 7. 三路快排 (Dutch National Flag)
# ============================================================
def quicksort_3way(arr: list, lo: int = 0, hi: int | None = None) -> None:
    """
    三路快排: 把等于 pivot 的元素聚在中间，不再递归。

    适合: 大量重复元素的场景

    维护三个指针:
      arr[lo..lt-1]  < pivot
      arr[lt..i-1]   == pivot
      arr[gt+1..hi]  > pivot
      arr[i..gt]     未扫描
    """
    if hi is None:
        hi = len(arr) - 1
    if lo >= hi:
        return

    pivot = arr[lo]
    lt, i, gt = lo, lo + 1, hi

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
            # 不 i += 1, 因为换来的元素还没检查
        else:
            i += 1

    # arr[lt..gt] 全是 pivot, 不用管了
    quicksort_3way(arr, lo, lt - 1)
    quicksort_3way(arr, gt + 1, hi)


# ============================================================
# 8. 带随机化的实用版快排
# ============================================================
def quicksort(arr: list) -> None:
    """
    实用版: 随机 pivot + Lomuto + 小数组切 insertion sort

    - 随机 pivot 避免最差情况
    - 小数组 (< 16) 用插入排序，减少递归开销
    """
    _qsort(arr, 0, len(arr) - 1)


def _qsort(arr: list, lo: int, hi: int) -> None:
    # 小数组用插入排序
    if hi - lo < 16:
        for i in range(lo + 1, hi + 1):
            key = arr[i]
            j = i - 1
            while j >= lo and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return

    # 随机选 pivot, 放到末尾
    rand_idx = random.randint(lo, hi)
    arr[rand_idx], arr[hi] = arr[hi], arr[rand_idx]

    p = lomuto_partition(arr, lo, hi)
    _qsort(arr, lo, p - 1)
    _qsort(arr, p + 1, hi)


# ============================================================
# 演示
# ============================================================
def demo():
    print("=" * 60)
    print("快速排序 — 递归 vs 非递归 对比演示")
    print("=" * 60)

    # 测试数据
    tests = [
        ("随机数组", [3, 6, 8, 10, 1, 2, 1]),
        ("已排序", list(range(10))),
        ("逆序", list(range(10, 0, -1))),
        ("全相同", [5, 5, 5, 5, 5]),
        ("大量重复", [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]),
    ]

    methods = [
        ("递归 (Lomuto)", quicksort_recursive),
        ("递归 (Hoare)", quicksort_hoare),
        ("非递归 (栈)", quicksort_iterative),
        ("尾递归优化", quicksort_tail_optimized),
        ("三路快排", quicksort_3way),
        ("实用版", quicksort),
    ]

    for name, data in tests:
        print(f"\n--- {name}: {data} ---")
        for mname, method in methods:
            arr = data.copy()
            method(arr)
            print(f"  {mname:>16}: {arr}")

    # Partition 对比
    print(f"\n{'=' * 60}")
    print("Partition 过程演示")
    print("=" * 60)

    arr = [3, 7, 8, 5, 2, 1, 9, 5, 4]
    print(f"\n原始: {arr}")

    a1 = arr.copy()
    p1 = lomuto_partition(a1, 0, len(a1) - 1)
    print(f"Lomuto (pivot=arr[-1]={arr[-1]}): {a1}, pivot_idx={p1}")

    a2 = arr.copy()
    p2 = hoare_partition(a2, 0, len(a2) - 1)
    print(f"Hoare  (pivot=arr[mid]={arr[len(arr)//2]}): {a2}, split_at={p2}")

    # 性能对比
    print(f"\n{'=' * 60}")
    print("非递归 vs 递归 — 大数组测试")
    print("=" * 60)

    import time
    for n in [1000, 10000, 50000]:
        data = [random.randint(0, n) for _ in range(n)]

        arr1 = data.copy()
        t0 = time.perf_counter()
        quicksort_iterative(arr1)
        t1 = time.perf_counter()

        arr2 = data.copy()
        t2 = time.perf_counter()
        quicksort(arr2)
        t3 = time.perf_counter()

        assert arr1 == arr2 == sorted(data)
        print(f"  n={n:>6}: 非递归={t1-t0:.4f}s  实用版={t3-t2:.4f}s")


if __name__ == "__main__":
    demo()
