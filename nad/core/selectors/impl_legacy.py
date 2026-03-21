"""
完全复制旧版NAD的选择器实现，用于验证差异原因
"""
from __future__ import annotations
import numpy as np
from .base import Selector

class LegacyKNNMedoidSelector(Selector):
    """
    完全复制旧版NAD的knn-medoid实现
    使用相似度而不是距离，计算top-k相似度的均值
    """
    def __init__(self, k: int = 3):
        self.k = k

    def select(self, D: np.ndarray, run_stats):
        """在距离矩阵上实现 kNN-Jaccard (用相似度=1-D)，返回获选索引"""
        R = D.shape[0]
        if R <= 1:
            return 0

        k = max(1, min(self.k, R-1))

        # 关键差异1: 转换为相似度矩阵
        S = 1.0 - D

        scores = np.zeros(R, dtype=np.float32)

        for i in range(R):
            # 关键差异2: 删除自己（不包括自己）
            sims = np.delete(S[i], i)

            # 关键差异3: 取前k个最高相似度（而不是最低距离）
            # 使用负数partition来获取最大的k个
            topk = np.partition(sims, -k)[-k:]

            # 关键差异4: 计算均值（而不是和）
            scores[i] = float(topk.mean())

        # 关键差异5: 选择分数最高的（而不是最低的）
        best = int(np.argmax(scores))

        # 关键差异6: tie-breaking使用np.isclose和medoid
        ties = np.where(np.isclose(scores, scores[best]))[0]
        if ties.size > 1:
            meanD = D.mean(axis=1)
            best = int(ties[np.argmin(meanD[ties])])

        return best


class LegacyMedoidSelector(Selector):
    """
    旧版medoid实现：平均距离最小
    """
    def select(self, D: np.ndarray, run_stats):
        if D.shape[0] <= 1:
            return 0
        meanD = D.mean(axis=1)
        return int(np.argmin(meanD))


class LegacyDBSCANMedoidSelector(Selector):
    """
    完全复制旧版NAD的DBSCAN-medoid实现
    """
    def __init__(self, eps = None, min_samples: int = 3, use_legacy_ja: bool = False):
        # eps: None or 'auto' -> adaptive per problem (30% quantile)
        if eps is None or (isinstance(eps, str) and str(eps).lower()=='auto'):
            self.eps = None
        else:
            self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.use_legacy_ja = bool(use_legacy_ja)

    def select(self, D: np.ndarray, run_stats):
        """基于距离阈值的DBSCAN，选择最大簇的medoid（复刻旧版行为）"""
        n = D.shape[0]
        if n <= 1:
            return 0

        # 如果启用legacy-JA，重新计算距离矩阵（使用去重的keys）
        if self.use_legacy_ja and "views" in run_stats:
            D = self._legacy_ja_matrix(run_stats["views"])

        # 旧版NAD在_main_cluster_medoid中应用了max(2, min_samples)约束
        actual_min_samples = int(max(2, self.min_samples))

        # 构建DBSCAN的labels（精确复刻旧版算法）
        visited = np.zeros(n, dtype=bool)
        labels = np.full(n, -1, dtype=np.int32)
        # 计算实际 eps（支持自适应）
        if self.eps is None:
            n = D.shape[0]
            if n <= 1:
                eps_val = 0.0
            else:
                tri = D[np.triu_indices(n, k=1)]
                eps_val = float(np.quantile(tri, 0.30)) if tri.size else 0.0
        else:
            eps_val = float(self.eps)
        Nbrs = (D <= eps_val)  # 邻接矩阵（含自身）
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = np.where(Nbrs[i])[0]

            # 如果邻居数量不足min_samples，标记为噪声
            if neighbors.size < actual_min_samples:
                labels[i] = -1
                continue

            # 开始新的簇
            labels[i] = cluster_id
            seeds = list(neighbors.tolist())  # seeds包含i的所有邻居（包括i自己）
            seeds_set = set(seeds)  # 用集合加速成员检查
            j = 0

            while j < len(seeds):
                p = seeds[j]

                if not visited[p]:
                    visited[p] = True
                    p_neighbors = np.where(Nbrs[p])[0]

                    # 如果p是核心点，将其邻居加入seeds
                    if p_neighbors.size >= actual_min_samples:
                        for u in p_neighbors:
                            # 旧版使用 int(u) 转换，确保是Python int
                            u_int = int(u)
                            if u_int not in seeds_set:
                                seeds.append(u_int)
                                seeds_set.add(u_int)

                # 无论p是否核心点，只要当前标签为负（未标注/噪声）就标注为当前簇
                if labels[p] < 0:
                    labels[p] = cluster_id

                j += 1

            cluster_id += 1

        # 找最大的簇
        if cluster_id == 0:
            # 无簇，退化为medoid
            return LegacyMedoidSelector().select(D, run_stats)

        # 与旧版一致：用 unique 统计 label 与计数
        cl_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        max_size = counts.max()
        best_cands = cl_ids[counts == max_size]

        if best_cands.size == 1:
            main = int(best_cands[0])
        else:
            # 多个并列的最大簇，按簇内平均相似度决胜
            S = 1.0 - D  # 转换为相似度矩阵
            best = None
            best_score = -np.inf
            for c in best_cands:
                idx = np.where(labels == c)[0]
                if idx.size <= 1:
                    score = 0.0
                else:
                    sub = S[np.ix_(idx, idx)]
                    score = float((sub.sum() - np.trace(sub)) / (idx.size * (idx.size - 1)))
                if score > best_score:
                    best_score = score
                    best = c
            main = int(best)

        # 在选中的簇内选medoid
        cluster_mask = (labels == main)
        cluster_indices = np.where(cluster_mask)[0]

        D_cluster = D[cluster_mask][:, cluster_mask]
        meanD_cluster = D_cluster.mean(axis=1)
        local_best = int(np.argmin(meanD_cluster))

        return int(cluster_indices[local_best])

    def _legacy_ja_matrix(self, views):
        """
        重新计算距离矩阵，使用去重的keys（复刻旧版NAD的JA计算）
        旧版NAD在计算Jaccard距离前会对keys做np.unique()
        """
        n = len(views)
        D = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                # 对每个view的keys进行去重
                keys_i = np.unique(views[i].keys)
                keys_j = np.unique(views[j].keys)

                # 计算Jaccard距离
                dist = self._jaccard_pair_legacy(keys_i, keys_j)
                D[i, j] = dist
                D[j, i] = dist

        return D

    @staticmethod
    def _jaccard_pair_legacy(keys_i, keys_j):
        """
        计算两个去重后的key数组的Jaccard距离
        Jaccard距离 = 1 - (交集大小 / 并集大小)
        """
        # 转换为集合进行交并集操作
        set_i = set(keys_i.tolist() if hasattr(keys_i, 'tolist') else keys_i)
        set_j = set(keys_j.tolist() if hasattr(keys_j, 'tolist') else keys_j)

        # 计算交集和并集
        intersection = len(set_i & set_j)
        union = len(set_i | set_j)

        # 避免除零错误
        if union == 0:
            return 0.0

        # Jaccard距离 = 1 - Jaccard相似度
        return 1.0 - (intersection / union)


class LegacyConsensusMinSelector(Selector):
    """
    旧版consensus-min实现
    """
    def __init__(self, k: int = 3, eps: float = 0.3, min_samples: int = 3):
        self.k = k
        self.eps = eps
        self.min_samples = min_samples

    def select(self, D: np.ndarray, run_stats):
        # 收集三个选择器的候选
        candidates = set()

        # KNN-medoid
        knn_choice = LegacyKNNMedoidSelector(self.k).select(D, run_stats)
        candidates.add(knn_choice)

        # Medoid
        medoid_choice = LegacyMedoidSelector().select(D, run_stats)
        candidates.add(medoid_choice)

        # DBSCAN-medoid
        dbscan_choice = LegacyDBSCANMedoidSelector(self.eps, self.min_samples).select(D, run_stats)
        candidates.add(dbscan_choice)

        # 在候选中选择激活数最少的
        candidates = list(candidates)
        if len(candidates) == 1:
            return candidates[0]

        lengths = run_stats["lengths"][candidates]
        min_idx = np.argmin(lengths)

        return candidates[min_idx]


class LegacyConsensusMaxSelector(Selector):
    """
    旧版consensus-max实现
    """
    def __init__(self, k: int = 3, eps: float = 0.3, min_samples: int = 3):
        self.k = k
        self.eps = eps
        self.min_samples = min_samples

    def select(self, D: np.ndarray, run_stats):
        # 收集三个选择器的候选
        candidates = set()

        # KNN-medoid
        knn_choice = LegacyKNNMedoidSelector(self.k).select(D, run_stats)
        candidates.add(knn_choice)

        # Medoid
        medoid_choice = LegacyMedoidSelector().select(D, run_stats)
        candidates.add(medoid_choice)

        # DBSCAN-medoid
        dbscan_choice = LegacyDBSCANMedoidSelector(self.eps, self.min_samples).select(D, run_stats)
        candidates.add(dbscan_choice)

        # 在候选中选择激活数最多的
        candidates = list(candidates)
        if len(candidates) == 1:
            return candidates[0]

        lengths = run_stats["lengths"][candidates]
        max_idx = np.argmax(lengths)

        return candidates[max_idx]
