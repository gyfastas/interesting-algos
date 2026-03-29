# K-Means 聚类算法

## 问题描述

**实现 K-Means 聚类算法。给定 N 个数据点和 K 个簇，找到每个点所属的簇。**

K-Means 是最经典的无监督学习算法，思想简单但应用广泛：图像分割、特征量化、数据预处理、向量检索（IVF）。

## 直觉分析

### 核心思想

K-Means 就是反复做两件事：
1. **分配**：每个点归入离它最近的中心
2. **更新**：每个中心移到它的成员的平均位置

```
初始: 随机放 K 个中心点
重复:
  Step 1 (Assign):  每个数据点 → 最近的中心
  Step 2 (Update):  每个中心 → 其成员的均值
直到中心不再移动 (收敛)
```

### 为什么会收敛？

K-Means 优化的目标函数是 **组内平方和 (Within-Cluster Sum of Squares, WCSS)**：

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

- **Assign 步**：固定中心 $\mu_k$，最优分配是把每个点分给最近的中心 → $J$ 不增
- **Update 步**：固定分配，最优中心是成员均值 → $J$ 不增

每步 $J$ 单调不增 + $J ≥ 0$ → 一定收敛。但可能收敛到局部最优。

## 算法细节

### Lloyd's Algorithm (标准 K-Means)

```python
def kmeans(X, K, max_iter=100):
    # 1. 随机初始化 K 个中心
    centers = random_select(X, K)

    for _ in range(max_iter):
        # 2. Assign: 每个点分配到最近的中心
        labels = [argmin_j ||x_i - centers[j]|| for x_i in X]

        # 3. Update: 每个中心 = 其成员均值
        for k in range(K):
            centers[k] = mean(X[labels == k])

        # 4. 收敛检查
        if centers_not_changed:
            break

    return labels, centers
```

### 时间复杂度

每次迭代：$O(N \cdot K \cdot D)$

- $N$：数据点数
- $K$：簇数
- $D$：维度

### 初始化策略

随机初始化可能导致糟糕的局部最优。K-Means++ 是更好的选择：

```
K-Means++ 初始化:
1. 随机选第一个中心 c_1
2. 对每个数据点 x，计算 D(x) = 到最近已选中心的距离
3. 以概率 D(x)² / Σ D(x)² 选择下一个中心
4. 重复 2-3 直到选了 K 个

直觉: 让初始中心彼此尽量远离
```

K-Means++ 保证期望的 WCSS 不超过最优的 $O(\log K)$ 倍。

### K 的选择

| 方法 | 思路 |
|------|------|
| **肘部法 (Elbow)** | 画 K vs WCSS 曲线，找"拐点" |
| **轮廓系数 (Silhouette)** | 衡量类内紧密度 vs 类间分离度 |
| **Gap Statistic** | 比较真实数据和随机数据的 WCSS 差 |

## 核心代码

```python
import numpy as np

class KMeans:
    def __init__(self, k, max_iter=100, init='kmeans++'):
        self.k = k
        self.max_iter = max_iter
        self.init = init

    def fit(self, X):
        N, D = X.shape

        # 初始化
        if self.init == 'kmeans++':
            self.centers = self._kmeans_pp(X)
        else:
            idx = np.random.choice(N, self.k, replace=False)
            self.centers = X[idx].copy()

        for _ in range(self.max_iter):
            # Assign
            dists = np.linalg.norm(X[:, None] - self.centers[None], axis=2)
            self.labels = np.argmin(dists, axis=1)

            # Update
            new_centers = np.array([
                X[self.labels == k].mean(axis=0) if (self.labels == k).any()
                else self.centers[k]
                for k in range(self.k)
            ])

            if np.allclose(new_centers, self.centers):
                break
            self.centers = new_centers

        return self.labels

    def _kmeans_pp(self, X):
        centers = [X[np.random.randint(len(X))]]
        for _ in range(1, self.k):
            dists = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
            probs = dists / dists.sum()
            centers.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centers)
```

## K-Means 的局限

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 只能找凸形簇 | 基于距离的分配 | DBSCAN, Spectral Clustering |
| 需要预设 K | 算法假设已知 | Elbow / Silhouette 选 K |
| 对初始化敏感 | 非凸优化 | K-Means++, 多次运行取最优 |
| 对离群点敏感 | 均值被拉偏 | K-Medoids (用中位数) |
| 各簇大小差异大 | 均匀分配偏好 | GMM (高斯混合模型) |

## K-Means 在 LLM / CV 中的应用

| 场景 | 用法 |
|------|------|
| **向量检索 (IVF)** | 先 K-Means 聚类，查询时只搜索最近的几个簇 |
| **VQ-VAE** | 用 K-Means 思想做向量量化，离散化 latent space |
| **图像分割** | 对像素颜色做 K-Means，得到分割结果 |
| **Product Quantization** | 分段 K-Means，用于高维向量压缩 |
| **Token 聚类** | 对 token embedding 聚类，分析模型表示 |

## 动画演示

> 打开 `animation.html` 查看交互动画，包含 2D 聚类可视化和 3D 旋转可视化。

## 答案与总结

| 要点 | 结论 |
|------|------|
| 核心 | Assign (分配到最近中心) + Update (中心 = 成员均值) |
| 收敛性 | WCSS 单调不增，一定收敛，但可能到局部最优 |
| 初始化 | K-Means++ 远好于随机，$O(\log K)$ 近似比 |
| 复杂度 | 每轮 $O(NKD)$，通常 10-50 轮收敛 |
| 局限 | 只能凸形簇、需要预设 K、对离群点敏感 |

**一句话总结**：K-Means = 反复"分配到最近中心 + 中心移到均值"，简单高效但只能处理凸形簇，用 K-Means++ 初始化效果好得多。
