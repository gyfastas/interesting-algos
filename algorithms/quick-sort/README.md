# 快速排序：递归与非递归实现

## 问题描述

**手写快速排序，分别给出递归和非递归（迭代）的实现。分析时间复杂度和 partition 策略。**

快速排序是实际工程中最常用的排序算法。C 标准库的 `qsort`、Python 的 `list.sort`（Timsort 混合）、Java 的 `Arrays.sort` 都有快排的身影。

## 直觉分析

### 核心思想：分治

```
选一个 pivot（基准元素）
把数组分成三部分：< pivot | == pivot | > pivot
递归排左边和右边
```

类比：整理一堆试卷
1. 拿出一张（pivot），分数 = 75
2. 比 75 低的放左边，比 75 高的放右边
3. 左边和右边各自再这样分
4. 最终每堆只剩一张 → 排好了

### 为什么快？

- 每次 partition 把问题规模砍半（期望情况）
- 不需要额外的合并步骤（不像归并排序）
- 原地排序，cache 友好

## Partition 策略

### Lomuto Partition（简单版）

```
用最后一个元素做 pivot
i 指针记录 "< pivot 区域" 的边界
从左到右扫描，遇到 < pivot 的就换到左边

[  < pivot  |  >= pivot  | 未扫描 | pivot ]
             ↑i           ↑j
```

```python
def lomuto_partition(arr, lo, hi):
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i
```

### Hoare Partition（高效版）

```
用中间元素做 pivot
两个指针从两端向中间扫
左指针找 >= pivot 的，右指针找 <= pivot 的
交换，直到指针交叉

[ <= pivot | 未扫描 | >= pivot ]
           ↑i     ↑j
```

```python
def hoare_partition(arr, lo, hi):
    pivot = arr[(lo + hi) // 2]
    i, j = lo - 1, hi + 1
    while True:
        i += 1
        while arr[i] < pivot: i += 1
        j -= 1
        while arr[j] > pivot: j -= 1
        if i >= j: return j
        arr[i], arr[j] = arr[j], arr[i]
```

### 对比

| | Lomuto | Hoare |
|---|---|---|
| 交换次数 | 多（每个 < pivot 都要交换） | 少（只在两边不对时交换） |
| 处理重复元素 | 退化为 O(n²) | 仍然较好 |
| 实现难度 | 简单 | 边界稍复杂 |
| 实际性能 | 约慢 3x | 更快 |

## 递归实现

```python
def quicksort_recursive(arr, lo=0, hi=None):
    if hi is None: hi = len(arr) - 1
    if lo < hi:
        p = partition(arr, lo, hi)
        quicksort_recursive(arr, lo, p - 1)  # 左半
        quicksort_recursive(arr, p + 1, hi)   # 右半
```

递归深度：期望 $O(\log n)$，最差 $O(n)$（已排序数组 + 取首/末元素做 pivot）。

## 非递归实现

用一个显式栈模拟递归调用栈：

```python
def quicksort_iterative(arr):
    stack = [(0, len(arr) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo < hi:
            p = partition(arr, lo, hi)
            stack.append((lo, p - 1))
            stack.append((p + 1, hi))
```

**为什么需要非递归版？**
- 避免栈溢出（Python 默认递归限制 1000）
- 大数据量时更安全
- 可以控制处理顺序（先处理小区间 → 限制栈深度为 $O(\log n)$）

### ��递归优化

```python
def quicksort_tail_optimized(arr, lo, hi):
    while lo < hi:
        p = partition(arr, lo, hi)
        # 先递归较小的一半，循环处理较大的一半
        if p - lo < hi - p:
            quicksort_tail_optimized(arr, lo, p - 1)
            lo = p + 1  # 尾递归变循环
        else:
            quicksort_tail_optimized(arr, p + 1, hi)
            hi = p - 1
```

这保证栈深度最多 $O(\log n)$。

## 三路快排（处理大量重复元素）

```
分成三部分: [< pivot | == pivot | > pivot]
== pivot 部分不再递归

[  < pivot  |  == pivot  |  ���扫描  |  > pivot  ]
             ↑lt          ↑i        ↑gt
```

```python
def quicksort_3way(arr, lo, hi):
    if lo >= hi: return
    pivot = arr[lo]
    lt, i, gt = lo, lo + 1, hi
    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1; i += 1
        elif arr[i] > pivot:
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
        else:
            i += 1
    quicksort_3way(arr, lo, lt - 1)
    quicksort_3way(arr, gt + 1, hi)
```

## 复杂度分析

| | 最好 | 平均 | 最差 | 空间 |
|---|---|---|---|---|
| 时间 | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | — |
| 空间 | $O(\log n)$ | $O(\log n)$ | $O(n)$ | 递归栈 |

**最差情况**：每次 pivot 恰好是最大/最小 → 只切掉一个元素 → $T(n) = T(n-1) + O(n) = O(n^2)$

**平均情况**：pivot 期望把数组切成大致相等的两半 → $T(n) = 2T(n/2) + O(n) = O(n \log n)$

### 如何避免最差情况

| 策略 | 效果 |
|------|------|
| **随机选 pivot** | 期望 $O(n \log n)$，概率极低退化 |
| **三数取中 (median of 3)** | 取首、中、末三个数的中位数 |
| **Introsort** | 递归太深就切换到 heapsort（C++ STL 的做法） |

## 动画演示

> 打开 `animation.html` 查看交互动画，逐步展示 partition 过程和递归/非递归对比。

## 答案与总结

| 要点 | 结论 |
|------|------|
| 核心 | 选 pivot → partition → 递归两半 |
| 递归版 | 简洁，但可能栈溢出 |
| 非递归版 | 用显式栈模拟，大数据更安全 |
| Partition | Hoare 比 Lomuto 快约 3x |
| 三路快排 | 重复元素多时效果好 |
| 最差退化 | 随机 pivot 或三数取中避免 |

**一句话总结**：快排 = "选 pivot 分两堆，递归排两边"——递归版简洁直观，非递归版用栈模拟避免溢出，Hoare partition + 随机 pivot 是最实用的组合。
