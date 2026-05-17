"""
单词接龙 (Word Ladder) — LeetCode 127
=========================================

核心考点：BFS 最短路径 + 双向 BFS 优化

问题：给定 beginWord、endWord 和 wordList，每次只能改变一个字母，
      求从 beginWord 到 endWord 的最短变换序列长度。

两种解法：
  1. 标准 BFS — O(N × L × 26)，N=wordList长度, L=单词长度
  2. 双向 BFS — 从两端同时搜索，大幅减少搜索空间

纯 Python 实现，含可视化打印。
"""

from collections import deque


# =============================================================================
# 解法 1: 标准 BFS
# =============================================================================

def ladder_length_bfs(beginWord: str, endWord: str, wordList: list) -> int:
    """
    标准 BFS：从 beginWord 一层一层向外扩展，第一次到达 endWord 即最短路径。

    思路：把每个单词看作图中的一个节点，如果两个单词只相差一个字母，则它们之间有边。
    问题转化为求图中两点之间的最短路径 → BFS。
    """
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0

    queue = deque([(beginWord, 1)])
    visited = {beginWord}

    while queue:
        word, steps = queue.popleft()

        if word == endWord:
            return steps

        # 尝试改变每一个位置的字母
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == word[i]:
                    continue
                next_word = word[:i] + c + word[i+1:]

                if next_word in wordSet and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, steps + 1))

    return 0


# =============================================================================
# 解法 2: 双向 BFS（面试考点）
# =============================================================================

def ladder_length_bibfs(beginWord: str, endWord: str, wordList: list) -> int:
    """
    双向 BFS：从 beginWord 和 endWord 同时开始搜索，每次扩展较小的一侧。

    为什么更快？
    - 标准 BFS 需要遍历 k^d 个节点（d=深度, k=分支因子）
    - 双向 BFS 只需 2 × k^(d/2)，指数级减少搜索空间
    """
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    if beginWord == endWord:
        return 1

    # 两端的 visited 字典：word -> 到起点的距离
    begin_visited = {beginWord: 1}
    end_visited = {endWord: 1}
    begin_queue = deque([beginWord])
    end_queue = deque([endWord])

    while begin_queue and end_queue:
        # 每次扩展队列较小的一侧，保证平衡
        if len(begin_queue) > len(end_queue):
            begin_queue, end_queue = end_queue, begin_queue
            begin_visited, end_visited = end_visited, begin_visited

        for _ in range(len(begin_queue)):
            word = begin_queue.popleft()
            steps = begin_visited[word]

            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    next_word = word[:i] + c + word[i+1:]

                    if next_word not in wordSet:
                        continue

                    # 如果 next_word 在另一端已被访问，说明找到了路径
                    if next_word in end_visited:
                        return steps + end_visited[next_word]

                    if next_word not in begin_visited:
                        begin_visited[next_word] = steps + 1
                        begin_queue.append(next_word)

    return 0


# =============================================================================
# 可视化 BFS 过程
# =============================================================================

def bfs_with_visualization(beginWord: str, endWord: str, wordList: list):
    """打印 BFS 的逐层扩展过程。"""
    wordSet = set(wordList)
    if endWord not in wordSet:
        print(f"{endWord} 不在 wordList 中，无法到达")
        return 0

    queue = deque([(beginWord, 1, [beginWord])])
    visited = {beginWord}
    layer = 1
    found = False

    print(f"开始 BFS: {beginWord} → {endWord}")
    print("=" * 50)

    while queue and not found:
        print(f"\n第 {layer} 层（当前队列大小: {len(queue)}）")
        for _ in range(len(queue)):
            word, steps, path = queue.popleft()

            if word == endWord:
                print(f"  ✓ 到达终点! 路径: {' → '.join(path)}")
                found = True
                return steps

            print(f"  从 '{word}' 扩展:")
            neighbors = []
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in wordSet and next_word not in visited:
                        visited.add(next_word)
                        queue.append((next_word, steps + 1, path + [next_word]))
                        neighbors.append(next_word)
            if neighbors:
                print(f"    → {neighbors}")

        layer += 1

    print(f"\n无法从 {beginWord} 到达 {endWord}")
    return 0


# =============================================================================
# 测试与验证
# =============================================================================

def run_tests():
    """运行 LeetCode 官方测试用例。"""
    tests = [
        # (beginWord, endWord, wordList, expected)
        ('hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog'], 5),
        ('hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log'], 0),
        ('a', 'c', ['a', 'b', 'c'], 2),
        ('hot', 'dog', ['hot', 'dog', 'dot'], 3),
        ('leet', 'code', ['lest', 'leet', 'lose', 'code', 'lode', 'robe', 'lost'], 6),
        ('talk', 'tail', ['talk', 'tons', 'fall', 'tail', 'gale', 'hall', 'neat'], 0),
    ]

    print("=" * 60)
    print("测试标准 BFS")
    print("=" * 60)
    for bw, ew, wl, expected in tests:
        result = ladder_length_bfs(bw, ew, wl)
        status = "✓" if result == expected else "✗"
        print(f"{status} {bw:6s} → {ew:6s}: {result:2d} (expected {expected})")

    print()
    print("=" * 60)
    print("测试双向 BFS")
    print("=" * 60)
    for bw, ew, wl, expected in tests:
        result = ladder_length_bibfs(bw, ew, wl)
        status = "✓" if result == expected else "✗"
        print(f"{status} {bw:6s} → {ew:6s}: {result:2d} (expected {expected})")


def demo_visualization():
    """可视化演示。"""
    print()
    bfs_with_visualization('hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog'])


def benchmark():
    """对比标准 BFS 和双向 BFS 的性能。"""
    import time

    # 构造一个较大的测试用例
    import random
    random.seed(42)

    # 构造一个链式 wordList: aaaa, aaab, aabb, abbb, bbbb
    chain = ['aaaa']
    for i in range(1, 26):
        prev = chain[-1]
        # 改变一个字母
        idx = (i - 1) % 4
        c = chr(ord('a') + i)
        chain.append(prev[:idx] + c + prev[idx+1:])

    begin = chain[0]
    end = chain[-1]

    # 加入一些干扰项
    noise = []
    for _ in range(500):
        w = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(4))
        noise.append(w)

    wordList = chain + noise

    print("\n" + "=" * 60)
    print("性能对比（链长 26 + 500 干扰项）")
    print("=" * 60)

    t0 = time.time()
    r1 = ladder_length_bfs(begin, end, wordList)
    t1 = time.time()

    t2 = time.time()
    r2 = ladder_length_bibfs(begin, end, wordList)
    t3 = time.time()

    print(f"标准 BFS:  结果={r1}, 耗时={(t1-t0)*1000:.2f} ms")
    print(f"双向 BFS:  结果={r2}, 耗时={(t3-t2)*1000:.2f} ms")


if __name__ == "__main__":
    run_tests()
    demo_visualization()
    benchmark()
