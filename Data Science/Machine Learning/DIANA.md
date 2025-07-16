
DIANA（Divisive Analysis）是一种经典的**分裂式层次聚类算法**，它与 AGNES（凝聚式层次聚类）相反，从“所有样本属于一个簇”开始，递归地将簇分裂为更小的簇，直到满足终止条件。以下是 DIANA 算法的详细解析：

### 一、核心思想与流程

#### 1. 算法逻辑
- **自顶向下**：从一个包含所有样本的根簇开始，每次选择“最不相似”的簇进行分裂，直到每个样本自成一簇或达到预设的簇数。
- **分裂依据**：通过计算簇内样本间的相似度（如距离），将簇划分为两个子簇，使得子簇内的相似度尽可能高，子簇间的相似度尽可能低。

#### 2. 关键步骤
1. **初始化**：所有样本属于同一个簇 $C_1$。
2. **分裂选择**：在当前所有簇中，选择**直径最大**的簇 $C_j$（直径定义为簇内任意两点间的最大距离）。
3. **簇分裂**  
   - 找出簇 $C_j$ 中与其他点平均距离最大的样本 $p$，将其作为**种子点（Seed）**。  
   - 将种子点 $p$ 移至新簇 $C_{\text{new}}$。  
   - 对原簇 $C_j$ 中的每个剩余样本 $q$：  
     - 计算 $q$ 到原簇 $C_j$ 的平均距离 $d_{\text{old}}$。  
     - 计算 $q$ 到新簇 $C_{\text{new}}$ 的平均距离 $d_{\text{new}}$。  
     - 若 $d_{\text{new}} < d_{\text{old}}$，则将 $q$ 移入新簇 $C_{\text{new}}$。  
   - 重复上述步骤，直到没有样本需要移动。
4. **迭代终止**：重复步骤 2–3，直到所有簇的直径小于阈值，或达到预设的簇数，或每个簇只包含一个样本。

---

### 二、数学形式化

#### 1. 簇的直径
对于簇 $C_j$，其直径 $D(C_j)$ 定义为  
$$D(C_j) = \max_{p, q \in C_j} \text{distance}(p, q)$$  
其中 $\text{distance}(p, q)$ 通常是欧氏距离或曼哈顿距离。

#### 2. 分裂标准
在分裂步骤中，样本 $q$ 的移动条件为  
$$\text{平均距离}(q, C_{\text{new}}) < \text{平均距离}(q, C_j)$$  
其中 $\text{平均距离}(q, C)$ 是样本 $q$ 到簇 $C$ 内所有样本的平均距离。

---

### 三、算法示例

假设有 5 个样本 $\{A, B, C, D, E\}$，初始时都属于同一个簇。DIANA 的分裂过程如下：

1. **第一次分裂**  
   - 计算所有样本间的距离，找出直径最大的簇（当前只有一个簇）。  
   - 假设样本 $C$ 与其他样本的平均距离最大，将 $C$ 作为种子点，创建新簇 $\{C\}$。  
   - 对剩余样本 $\{A, B, D, E\}$，计算每个样本到原簇和新簇的平均距离，将距离新簇更近的样本移入新簇。例如，最终分裂为 $\{A, B, D\}$ 和 $\{C, E\}$。

2. **第二次分裂**  
   - 选择直径较大的簇（如 $\{A, B, D\}$），重复上述步骤，可能分裂为 $\{A, B\}$ 和 $\{D\}$。

3. **迭代直至终止**  
   - 继续分裂，直到满足终止条件（如每个簇只含一个样本）。

---

### 四、优缺点

#### 优点
1. **全局视角**：初始时考虑所有样本的关系，避免凝聚式算法（如 AGNES）的“合并错误无法回溯”问题。  
2. **适合小数据集**：在样本数较少时，能生成更有层次结构的聚类结果。  
3. **任意形状簇**：理论上可处理非凸形状的簇（取决于距离度量）。

#### 缺点
1. **计算复杂度高**：每次分裂需计算所有样本间的距离，时间复杂度为 $O(n^2)$，对大样本（如 $n > 10^4$）不适用。  
2. **内存需求大**：需存储完整的距离矩阵。  
3. **对噪声敏感**：无明确的噪声处理机制，异常值可能影响分裂结果。  
4. **终止条件模糊**：需人工指定簇数或直径阈值，缺乏自动确定最佳簇数的方法。

---

### 五、与 AGNES 的对比

| 特性 | DIANA（分裂式） | AGNES（凝聚式） |
|------|-----------------|-----------------|
| 方向 | 自顶向下（从 1 到 n 个簇） | 自底向上（从 n 到 1 个簇） |
| 计算复杂度 | $O(n^2)$（每次分裂需全量计算） | $O(n^3)$（需维护距离矩阵） |
| 合并/分裂决策 | 基于簇内相似度（直径） | 基于簇间相似度（单/全/平均链接） |
| 对噪声的鲁棒性 | 较差（异常值影响分裂点选择） | 较好（单链接可缓解） |
| 适用场景 | 小数据集、需全局结构 | 中等规模数据、需灵活相似度定义 |

---

### 六、应用场景

- **生物学分类**：如物种进化树构建（从大类逐步细分）。  
- **文档聚类**：如将文档集合按主题层级划分（从通用到具体）。  
- **地理区域划分**：如将国家逐级划分为省、市、县等。

---

### 七、Python 实现示例

Scikit-learn 未直接提供 DIANA 实现，但可通过递归方式手动实现：

```python
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def diana(X, max_clusters=None):
    """
    DIANA 分裂式层次聚类实现
    参数:
        X: 输入数据，形状为 (n_samples, n_features)
        max_clusters: 最大簇数，若为 None 则分裂到每个样本自成一簇
    返回:
        clusters: 最终的簇划分，列表形式，每个元素为样本索引的集合
    """
    n_samples = X.shape[0]
    clusters = [np.arange(n_samples)]  # 初始时所有样本属于一个簇
    while True:
        if max_clusters is not None and len(clusters) >= max_clusters:
            break
        if all(len(c) == 1 for c in clusters):
            break
        diameters = [cluster_diameter(X, c) for c in clusters]
        max_idx = np.argmax(diameters)
        if diameters[max_idx] == 0:
            break
        selected_cluster = clusters.pop(max_idx)
        sub_clusters = split_cluster(X, selected_cluster)
        clusters.extend(sub_clusters)
    return clusters

def cluster_diameter(X, indices):
    """计算簇的直径（簇内任意两点的最大距离）"""
    if len(indices) <= 1:
        return 0
    dists = euclidean_distances(X[indices])
    return np.max(dists)

def split_cluster(X, indices):
    """分裂簇，返回两个子簇"""
    n_points = len(indices)
    if n_points <= 1:
        return [indices]
    dists = euclidean_distances(X[indices])
    avg_dists = np.mean(dists, axis=1)
    seed_idx = np.argmax(avg_dists)
    cluster1 = [seed_idx]
    cluster2 = list(range(n_points))
    cluster2.remove(seed_idx)
    changed = True
    while changed:
        changed = False
        for i in cluster2.copy():
            dist_to_cluster1 = np.mean([dists[i, j] for j in cluster1])
            dist_to_cluster2 = np.mean([dists[i, j] for j in cluster2 if j != i])
            if dist_to_cluster1 < dist_to_cluster2:
                cluster2.remove(i)
                cluster1.append(i)
                changed = True
        for i in cluster1.copy():
            dist_to_cluster1 = np.mean([dists[i, j] for j in cluster1 if j != i])
            dist_to_cluster2 = np.mean([dists[i, j] for j in cluster2])
            if dist_to_cluster2 < dist_to_cluster1:
                cluster1.remove(i)
                cluster2.append(i)
                changed = True
    sub_cluster1 = indices[cluster1]
    sub_cluster2 = indices[cluster2]
    return [sub_cluster1, sub_cluster2] if len(sub_cluster2) > 0 else [sub_cluster1]
```

---

### 八、总结

DIANA 算法通过自顶向下的分裂策略，为层次聚类提供了一种与 AGNES 互补的实现方式。其优势在于全局视角和适合小数据集的层次结构挖掘，但计算开销较大，需谨慎用于大规模数据。实际应用中，需结合距离度量和终止条件的选择，平衡聚类质量与效率。
