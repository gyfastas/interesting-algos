# 036 单词接龙 (Word Ladder) — LeetCode 127

**标签:** `BFS` · `双向 BFS` · `图论` · `最短路径`

## 问题

给定两个单词 `beginWord` 和 `endWord`，以及一个字典 `wordList`。每次变换只能改变一个字母，且新单词必须在 `wordList` 中。求从 `beginWord` 到 `endWord` 的**最短变换序列长度**。

**示例:**
```
beginWord = "hit"
endWord   = "cog"
wordList  = ["hot","dot","dog","lot","log","cog"]

最短序列: hit → hot → dot → dog → cog
答案: 5（包含起点和终点）
```

## 核心思想

把每个单词看作图中的一个节点，如果两个单词只相差一个字母，则它们之间有边。
问题转化为求**无权图的最短路径** → **BFS**。

### 标准 BFS

从 `beginWord` 出发，逐层向外扩展。第一次到达 `endWord` 时的深度即为答案。

```
第 1 层: hit
第 2 层: hot
第 3 层: dot, lot
第 4 层: dog, log
第 5 层: cog  ✓
```

### 双向 BFS（面试考点 ⭐）

从 `beginWord` 和 `endWord` **两端同时搜索**，每次扩展较小的一侧。

**为什么更快？**
- 标准 BFS 需要遍历 kᵈ 个节点（d = 深度, k = 分支因子）
- 双向 BFS 只需 2 × k^(d/2)，**指数级减少搜索空间**

```
前向:  hit → hot → dot → dog
后向:  cog → dog

在 "dog" 相遇！路径长度 = 4 + 2 = 6
```

## 复杂度分析

| 指标 | 标准 BFS | 双向 BFS |
|------|----------|----------|
| 时间 | O(N × L × 26) | O(N × L × 26)，常数更小 |
| 空间 | O(N) | O(N) |

N = wordList 长度, L = 单词长度

## 关键代码

```python
# 标准 BFS
queue = deque([(beginWord, 1)])
visited = {beginWord}

while queue:
    word, steps = queue.popleft()
    if word == endWord:
        return steps

    for i in range(len(word)):
        for c in 'abcdefghijklmnopqrstuvwxyz':
            next_word = word[:i] + c + word[i+1:]
            if next_word in wordSet and next_word not in visited:
                visited.add(next_word)
                queue.append((next_word, steps + 1))

# 双向 BFS — 每次扩展较小的一侧
if len(begin_queue) > len(end_queue):
    begin_queue, end_queue = end_queue, begin_queue
    begin_visited, end_visited = end_visited, begin_visited
```

## 面试要点

1. **为什么用 BFS 而不是 DFS？** BFS 保证第一次到达终点时路径最短；DFS 需要遍历所有路径才能确定最短。
2. **双向 BFS 的终止条件？** 当扩展的节点被另一端访问过时，两端的距离之和即为答案。
3. **如何优化邻居生成？** 可以预先用通配符建立邻接表（`"h*t" → [hot, hit, hat]`），将查询从 O(26×L) 降到 O(1)。

## 文件

- `solution.py` — 标准 BFS + 双向 BFS + 可视化演示
- `animation.html` — 交互式 BFS 动画（零依赖）

## 运行

```bash
python algorithms/word-ladder/solution.py
```
