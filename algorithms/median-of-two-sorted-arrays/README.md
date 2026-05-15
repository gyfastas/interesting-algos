# 寻找两个有序数组的中位数

> LeetCode 4 · Hard · 二分查找

## 问题描述

给定两个长度分别为 $m$ 和 $n$ 的已排序数组 `nums1` 和 `nums2`，找出它们合并后的中位数。

要求算法的时间复杂度为 $O(\log(m+n))$。

> **示例**
> - `nums1 = [1, 3], nums2 = [2]` → 中位数 `2.0`
> - `nums1 = [1, 2], nums2 = [3, 4]` → 中位数 `2.5`

## 暴力解法

在思考最优解之前，先看两种直观的暴力方法：

### 方法一：合并后排序

直接把两个数组合并成一个数组，然后排序，取中位数：

```python
def brute_force_median(nums1, nums2):
    merged = sorted(nums1 + nums2)
    L = len(merged)
    if L % 2 == 1:
        return float(merged[L // 2])
    else:
        return (merged[L // 2 - 1] + merged[L // 2]) / 2.0
```

**复杂度**：
- 时间：合并是 $O(m+n)$，排序是 $O((m+n)\log(m+n))$，总复杂度 $O((m+n)\log(m+n))$
- 空间：新建合并数组，$O(m+n)$

### 方法二：双指针线性合并

利用两个数组已经有序的特点，用双指针按顺序"归并"，只需遍历到中位数位置即可停止：

```python
def linear_merge_median(nums1, nums2):
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
```

**复杂度**：
- 时间：最多遍历到中位数位置，$O(m+n)$
- 空间：只用常数额外变量，$O(1)$

### 复杂度对比

| 方法 | 时间复杂度 | 空间复杂度 | 核心思想 |
|------|-----------|-----------|---------|
| 合并排序 | $O((m+n)\log(m+n))$ | $O(m+n)$ | 完全忽略有序性，通用但低效 |
| 双指针线性 | $O(m+n)$ | $O(1)$ | 利用有序性归并，但遍历全部 |
| **二分分割（最优）** | $O(\log(\min(m,n)))$ | $O(1)$ | 直接定位分割线，跳过无关元素 |

题目要求 $O(\log(m+n))$，这意味着暴力线性方法不够。我们需要一种**直接"跳到"正确位置**的方法——这就是二分查找。

## 直觉分析

从暴力到最优的跨越在于：我们不再需要逐个查看元素，而是直接利用有序性"跳到"正确位置。这自然让人想到**二分查找**。

但二分查找什么？两个数组各有自己的"中点"，互不相关。关键洞察是：

> **中位数的本质是"分割线"——它把合并后的数组切成左右两半，左边所有元素 <= 右边所有元素。**

如果我们能在两个原始数组上各找一个切分位置，使得切出来的左半部分恰好是合并后数组的左半部分，问题就解决了。

想象两排学生按身高站好，你要在两排之间画一条"分界线"，让分界线左边的人全都不高于右边的人，且左右人数相等（或左边多一个）。这就是我们要做的。

## 数学建模

### 变量定义

- `nums1`: 长度 $m$，在位置 $i$ 处切分（$0 \le i \le m$），左边有 $i$ 个元素
- `nums2`: 长度 $n$，在位置 $j$ 处切分（$0 \le j \le n$），左边有 $j$ 个元素

### 约束条件

合并后总长度为 $m+n$。中位数分割线要求左半部分包含的元素个数为：

$$
\text{left\_count} = \left\lfloor \frac{m+n+1}{2} \right\rfloor
$$

因此：

$$
i + j = \left\lfloor \frac{m+n+1}{2} \right\rfloor
$$

### 正确分割的判定

设：
- `nums1_left_max` = `nums1[i-1]`（若 $i=0$ 则为 $-\infty$）
- `nums1_right_min` = `nums1[i]`（若 $i=m$ 则为 $+\infty$）
- `nums2_left_max` = `nums2[j-1]`（若 $j=0$ 则为 $-\infty$）
- `nums2_right_min` = `nums2[j]`（若 $j=n$ 则为 $+\infty$）

当且仅当满足以下条件时，分割正确：

$$
\begin{cases}
\text{nums1\_left\_max} \le \text{nums2\_right\_min} \\
\text{nums2\_left\_max} \le \text{nums1\_right\_min}
\end{cases}
$$

### 中位数计算

- 若 $m+n$ 为奇数：
  $$
  \text{median} = \max(\text{nums1\_left\_max}, \text{nums2\_left\_max})
  $$

- 若 $m+n$ 为偶数：
  $$
  \text{median} = \frac{\max(\text{nums1\_left\_max}, \text{nums2\_left\_max}) + \min(\text{nums1\_right\_min}, \text{nums2\_right\_min})}{2}
  $$

## 求解过程

### 二分查找策略

由于 $i + j = \text{常数}$，我们只需在较短的数组（假设是 `nums1`）上对 $i$ 做二分查找：

1. 设 `lo = 0, hi = m`
2. 取 `i = (lo + hi) // 2`，则 `j = left_count - i`
3. 检查分割条件：
   - 若 `nums1_left_max > nums2_right_min` → `nums1` 切得太多，$i$ 太大，`hi = i - 1`
   - 若 `nums2_left_max > nums1_right_min` → `nums1` 切得太少，$i$ 太小，`lo = i + 1`
   - 否则，找到正确分割，计算中位数

### 时间复杂度

二分查找的范围是 `nums1` 的长度 $m$（已保证 $m \le n$），每轮 $O(1)$ 判定，总时间复杂度 $O(\log m) = O(\log(\min(m,n)))$，满足 $O(\log(m+n))$ 的要求。

### 空间复杂度

仅使用常数个变量，$O(1)$。

## 代码实现

```python
def find_median(nums1, nums2):
    # 保证 nums1 是较短的数组
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

        # 找到正确分割
        if nums1_left_max <= nums2_right_min and nums2_left_max <= nums1_right_min:
            if (m + n) % 2 == 1:
                return float(max(nums1_left_max, nums2_left_max))
            else:
                return (max(nums1_left_max, nums2_left_max) +
                        min(nums1_right_min, nums2_right_min)) / 2.0

        # nums1 切得太多，向左收缩
        elif nums1_left_max > nums2_right_min:
            hi = i - 1
        # nums1 切得太少，向右扩张
        else:
            lo = i + 1
```

## 动画演示

动画展示了二分查找分割点的完整过程：

- **数组可视化**：两个有序数组以条形图形式展示，高度代表数值大小
- **分割动态**：红色虚线实时显示当前 `i` 和 `j` 的切分位置
- **条件检查**：标注四个关键值（左最大、右最小），用颜色直观表示条件是否满足
- **Monte Carlo**：点击"随机测试"按钮，实时生成随机数组并执行算法，对比暴力法结果
- **交互控制**：支持逐步演示（Step）、自动播放（Play）、速度调节和重置

> 打开 `animation.html` 查看交互动画

## 扩展：K 个有序数组的中位数

如果把问题从 2 个数组扩展到 **K 个有序数组**，分割线的思路依然适用，但直接在两两之间做分割变得复杂。我们需要新的策略。

### 方法三选

| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| 暴力合并排序 | $O(N \log N)$ | $O(N)$ | $N$ 较小，代码最短 |
| **最小堆 K 路归并** | $O(N \log K)$ | $O(K)$ | $K$ 不大，值域极广 |
| **值域二分** | $O(K \log M \log W)$ | $O(1)$ | $K$ 很大，值域有限 |

> $N$ = 总元素数，$K$ = 数组个数，$M$ = 单个数组平均长度，$W$ = 值域宽度

### 方法一：最小堆 K 路归并

维护一个大小为 $K$ 的最小堆，每个数组当前最小的元素入堆。每次弹出堆顶（全局最小），并将该数组的下一个元素入堆。只需走到中位数位置即可停止。

```python
import heapq

def k_median_heap(arrays):
    total = sum(len(arr) for arr in arrays)
    target = total // 2
    heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))

    prev = curr = 0
    for _ in range(target + 1):
        prev = curr
        curr, arr_idx, elem_idx = heapq.heappop(heap)
        if elem_idx + 1 < len(arrays[arr_idx]):
            heapq.heappush(heap, (arrays[arr_idx][elem_idx + 1], arr_idx, elem_idx + 1))

    if total % 2 == 1:
        return float(curr)
    else:
        return (prev + curr) / 2.0
```

**分析**：每次堆操作 $O(\log K)$，要走大约 $N/2$ 步，总时间 $O(N \log K)$。当 $K$ 固定时，这接近线性。

### 方法二：值域二分（推荐）

这是把「在数组索引上二分」升级为「在值域上二分」：

1. **确定值域**：所有数组的最小值 $L$ 和最大值 $R$
2. **二分猜测**：取 $mid = (L+R)//2$
3. **统计验证**：对每个有序数组，用二分查找统计 $\le mid$ 的元素个数。如果总数 $\ge k$，说明第 $k$ 小的元素 $\le mid$，向左收缩；否则向右扩张
4. **收敛**：最终 $L=R$ 即为第 $k$ 小的元素

```python
from bisect import bisect_right

def k_median_binary_search(arrays):
    arrays = [arr for arr in arrays if arr]
    total = sum(len(arr) for arr in arrays)
    low = min(arr[0] for arr in arrays)
    high = max(arr[-1] for arr in arrays)

    def find_kth(k):
        lo, hi = low, high
        while lo < hi:
            mid = (lo + hi) // 2
            cnt = sum(bisect_right(arr, mid) for arr in arrays)
            if cnt < k:
                lo = mid + 1
            else:
                hi = mid
        return lo

    if total % 2 == 1:
        return float(find_kth(total // 2 + 1))
    else:
        return (find_kth(total // 2) + find_kth(total // 2 + 1)) / 2.0
```

**分析**：
- 外层二分：值域范围 $W$，需 $O(\log W)$ 轮
- 内层统计：$K$ 个数组各做一次二分查找，每轮 $O(K \log M)$
- 总时间：$O(K \log M \log W)$
- 空间：$O(1)$

**为什么值域二分比堆更好？**
- 当 $K$ 很大（如 $K=10^5$）但每个数组很短时，值域二分几乎不随 $K$ 线性增长（$\log M$ 很小），而堆的 $O(N \log K)$ 会显著变慢
- 值域二分不依赖总元素数 $N$，只依赖值域宽度和数组个数

### K 数组的复杂度全景

```
K 个有序数组找中位数
│
├─ 暴力合并排序    O(N log N)      O(N)      完全忽略有序性
├─ 最小堆归并      O(N log K)      O(K)      利用有序性顺序遍历
└─ 值域二分        O(K logM logW)  O(1)      利用有序性直接定位
```

## 答案与总结

**核心 Insight**：把"找中位数"转化为"找正确分割线"。在两个有序数组上各切一刀，只要保证左边所有元素 <= 右边所有元素，并且左右元素数量平衡，中位数就能直接从边界值算出来。

**关键点**：
1. 在较短的数组上做二分，降低时间复杂度
2. 用 $-\infty$ 和 $+\infty$ 优雅处理数组边界（$i=0$ 或 $i=m$）
3. 判定条件 `nums1_left_max <= nums2_right_min` 和 `nums2_left_max <= nums1_right_min` 确保了两半有序

**复杂度**：
- 时间：$O(\log(\min(m,n)))$
- 空间：$O(1)$
