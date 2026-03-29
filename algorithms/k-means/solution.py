"""
K-Means 聚类算法 — 手写实现

包含: 标准 K-Means (Lloyd's) + K-Means++ 初始化 + Elbow Method
"""

import numpy as np
from collections import defaultdict


# ============================================================
# 1. K-Means 核心实现
# ============================================================
class KMeans:
    """
    K-Means 聚类:
    1. 初始化 K 个中心 (随机 or K-Means++)
    2. Assign: 每个点分配到最近中心
    3. Update: 每个中心 = 成员均值
    4. 重复直到收敛
    """

    def __init__(self, k: int, max_iter: int = 100, init: str = 'kmeans++', tol: float = 1e-6):
        self.k = k
        self.max_iter = max_iter
        self.init = init
        self.tol = tol
        self.centers = None
        self.labels = None
        self.inertia = None  # WCSS
        self.n_iter = 0

    def fit(self, X: np.ndarray) -> 'KMeans':
        N, D = X.shape

        # Step 1: 初始化中心
        if self.init == 'kmeans++':
            self.centers = self._kmeans_pp_init(X)
        else:
            idx = np.random.choice(N, self.k, replace=False)
            self.centers = X[idx].copy()

        for it in range(self.max_iter):
            # Step 2: Assign — 每个点分配到最近中心
            # dists[i, k] = ||x_i - center_k||^2
            dists = self._compute_distances(X, self.centers)
            self.labels = np.argmin(dists, axis=1)

            # Step 3: Update — 每个中心 = 成员均值
            new_centers = np.empty_like(self.centers)
            for k in range(self.k):
                members = X[self.labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                else:
                    # 空簇: 重新随机选一个点
                    new_centers[k] = X[np.random.randint(N)]

            # Step 4: 收敛检查
            shift = np.linalg.norm(new_centers - self.centers)
            self.centers = new_centers
            self.n_iter = it + 1

            if shift < self.tol:
                break

        # 计算最终 WCSS
        self.inertia = self._compute_wcss(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """对新数据点预测簇标签。"""
        dists = self._compute_distances(X, self.centers)
        return np.argmin(dists, axis=1)

    def _kmeans_pp_init(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++ 初始化:
        1. 随机选第一个中心
        2. 后续中心按 D(x)² 概率选择 (离已选中心越远概率越大)

        保证: E[WCSS] ≤ O(log K) × 最优 WCSS
        """
        N = len(X)
        centers = [X[np.random.randint(N)].copy()]

        for _ in range(1, self.k):
            # D(x)² = 到最近已选中心的距离平方
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
            min_dists = dists.min(axis=0)  # 每个点到最近中心的距离

            # 按距离平方作为概率采样
            probs = min_dists / min_dists.sum()
            next_idx = np.random.choice(N, p=probs)
            centers.append(X[next_idx].copy())

        return np.array(centers)

    @staticmethod
    def _compute_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """计算每个点到每个中心的距离平方。(N, K)"""
        # ||x - c||² = ||x||² + ||c||² - 2 x·c
        # 利用矩阵运算避免显式循环
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)       # (N, 1)
        C_sq = np.sum(centers ** 2, axis=1, keepdims=True)  # (K, 1)
        cross = X @ centers.T                                # (N, K)
        return X_sq + C_sq.T - 2 * cross                    # (N, K)

    def _compute_wcss(self, X: np.ndarray) -> float:
        """计算 Within-Cluster Sum of Squares。"""
        wcss = 0.0
        for k in range(self.k):
            members = X[self.labels == k]
            if len(members) > 0:
                wcss += np.sum((members - self.centers[k]) ** 2)
        return wcss


# ============================================================
# 2. Elbow Method — 选择最优 K
# ============================================================
def elbow_method(X: np.ndarray, k_range: range = range(1, 11), n_runs: int = 3):
    """
    肘部法: 对每个 K 运行 K-Means，记录 WCSS。
    画 K vs WCSS 曲线，拐点处的 K 通常是最佳选择。

    n_runs: 每个 K 运行多次，取最优 (解决初始化随机性)
    """
    results = []
    for k in k_range:
        best_wcss = float('inf')
        for _ in range(n_runs):
            km = KMeans(k=k)
            km.fit(X)
            best_wcss = min(best_wcss, km.inertia)
        results.append((k, best_wcss))
    return results


# ============================================================
# 3. Silhouette Score — 聚类质量评估
# ============================================================
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    轮廓系数: 衡量聚类质量。

    对每个点 i:
      a(i) = 到同簇其他点的平均距离 (类内紧密度)
      b(i) = 到最近其他簇所有点的平均距离 (类间分离度)
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    s ∈ [-1, 1]:
      +1: 完美分离
       0: 在边界上
      -1: 分错了
    """
    N = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) <= 1:
        return 0.0

    scores = np.zeros(N)
    for i in range(N):
        # a(i): 同簇平均距离
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a_i = 0

        # b(i): 最近其他簇平均距离
        b_i = float('inf')
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = X[labels == label]
            avg_dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
            b_i = min(b_i, avg_dist)

        if max(a_i, b_i) > 0:
            scores[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(scores)


# ============================================================
# 演示
# ============================================================
def demo():
    np.random.seed(42)

    # 生成 3 个高斯簇的 2D 数据
    cluster1 = np.random.randn(50, 2) + [2, 2]
    cluster2 = np.random.randn(50, 2) + [-2, -2]
    cluster3 = np.random.randn(50, 2) + [2, -2]
    X = np.vstack([cluster1, cluster2, cluster3])

    print("=" * 60)
    print("K-Means 聚类演示 (150 个 2D 点, 3 个真实簇)")
    print("=" * 60)

    # K-Means++ 初始化
    km = KMeans(k=3, init='kmeans++')
    km.fit(X)

    print(f"\n收敛轮数: {km.n_iter}")
    print(f"WCSS: {km.inertia:.2f}")
    print(f"簇中心:")
    for i, c in enumerate(km.centers):
        n = np.sum(km.labels == i)
        print(f"  Cluster {i}: center=({c[0]:.2f}, {c[1]:.2f}), {n} 个点")

    # 轮廓系数
    sil = silhouette_score(X, km.labels)
    print(f"\n轮廓系数: {sil:.3f} (越接近 1 越好)")

    # Elbow method
    print(f"\n{'=' * 60}")
    print("Elbow Method — K vs WCSS")
    print("=" * 60)
    results = elbow_method(X, k_range=range(1, 8))
    for k, wcss in results:
        bar = '█' * int(wcss / 20)
        print(f"  K={k}: WCSS={wcss:>8.1f}  {bar}")
    print("  → 拐点在 K=3 (之后 WCSS 下降变缓)")

    # K-Means++ vs Random 对比
    print(f"\n{'=' * 60}")
    print("K-Means++ vs 随机初始化 (各运行 10 次)")
    print("=" * 60)
    pp_results, rand_results = [], []
    for _ in range(10):
        km_pp = KMeans(k=3, init='kmeans++')
        km_pp.fit(X)
        pp_results.append(km_pp.inertia)

        km_rand = KMeans(k=3, init='random')
        km_rand.fit(X)
        rand_results.append(km_rand.inertia)

    print(f"  K-Means++: mean WCSS = {np.mean(pp_results):.1f} ± {np.std(pp_results):.1f}")
    print(f"  Random:    mean WCSS = {np.mean(rand_results):.1f} ± {np.std(rand_results):.1f}")
    print(f"  → K-Means++ 更稳定，方差更小")


if __name__ == "__main__":
    demo()
