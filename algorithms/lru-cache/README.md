# LRU Cache（基础版 + 线程安全进阶版）

> LeetCode 146 · 数据结构 · 哈希表 · 双向链表 · ⭐⭐⭐

## 问题描述

设计和实现一个 **LRU (Least Recently Used) 缓存** 机制：
- `get(key)`：获取 key 对应的值，若不存在返回 -1。访问后该 key 变为「最近使用」
- `put(key, value)`：插入或更新 key-value。若容量已满，淘汰「最久未使用」的 key

要求 `get` 和 `put` 的时间复杂度都是 $O(1)$。

> **示例**（capacity = 2）
> - `put(1, 1)` → 缓存 `{1=1}`
> - `put(2, 2)` → 缓存 `{1=1, 2=2}`
> - `get(1)` → 返回 1，缓存变为 `{2=2, 1=1}`（1 变为最近使用）
> - `put(3, 3)` → 容量满，淘汰 2，缓存变为 `{1=1, 3=3}`
> - `get(2)` → 返回 -1（已被淘汰）

## 直觉分析

### 为什么需要双向链表？

LRU 的核心是维护**访问时间的全序关系**：谁最近被用了，谁最久没用了。

如果用数组或队列：
- 把刚访问的元素移到末尾 → $O(n)$ 的移动开销
- 删除最前面的元素 → $O(n)$

双向链表可以做到：
- 任意节点的删除 → $O(1)$（已知前驱和后继）
- 任意节点插到尾部 → $O(1)$
- 删除头部 → $O(1)$

### 为什么需要哈希表？

链表的问题在于**查找慢**：找 key 对应的节点需要 $O(n)$ 遍历。

哈希表 $key \to node$ 实现了 $O(1)$ 定位：
- `get`：哈希表找到节点 → 链表移到尾部
- `put`：哈希表判断是否存在 → 更新或新建节点 → 链表操作

> **核心 Insight**：哈希表负责"找"，双向链表负责"排顺序"，两者互补。

## 数学建模

### 数据结构

| 结构 | 作用 | 操作复杂度 |
|------|------|-----------|
| 哈希表 `dict` | `key → DLinkedNode` 映射 | 查找 $O(1)$，插入 $O(1)$，删除 $O(1)$ |
| 双向链表 | 按访问时间排序，头=最旧，尾=最新 | 头/尾删除 $O(1)$，任意节点移动 $O(1)$ |

### 链表操作

```
伪头部 ←→ 节点1 ←→ 节点2 ←→ ... ←→ 节点k ←→ 伪尾部
   ↑        最久未使用                  最近使用      ↑
   └────────────────────────────────────────────────┘
```

## 求解过程

### 基础版实现

```python
class DLinkedNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        self.size = 0

        # 伪头部和伪尾部，简化边界处理
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_tail(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._move_to_tail(node)
        else:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_to_tail(node)
            self.size += 1
            if self.size > self.capacity:
                removed = self._pop_head()
                del self.cache[removed.key]
                self.size -= 1
```

### 线程安全进阶版

在基础版上加 `threading.RLock`：

```python
class ThreadSafeLRUCache:
    def __init__(self, capacity: int):
        # ... 同基础版 ...
        self.lock = threading.RLock()

    def get(self, key: int) -> int:
        with self.lock:
            # ... 同基础版 get ...

    def put(self, key: int, value: int) -> None:
        with self.lock:
            # ... 同基础版 put ...
```

> 用 `RLock`（可重入锁）而非 `Lock`，防止同一线程内部方法互相调用时死锁。

### 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| `get` | $O(1)$ | $O(1)$ |
| `put` | $O(1)$ | $O(1)$ |
| 总空间 | — | $O(capacity)$ |

## 代码实现

完整代码见 `solution.py`，包含：
- 手写双向链表版 LRUCache
- 线程安全版 ThreadSafeLRUCache
- OrderedDict 版对比（Python 内置）
- 边界条件测试 + 并发测试 + 性能测试

## 动画演示

动画展示了 LRU 缓存的实时操作过程：

- **缓存面板**：可视化显示当前缓存中的 key-value，按访问时间从左到右排列（左=最旧，右=最新）
- **链表结构**：实时展示双向链表的节点连接关系，get/put 操作时的节点移动动画
- **操作日志**：每次 get/put 的详细记录，标注淘汰事件
- **并发模拟**：多个「线程」同时请求缓存，展示线程安全版的锁竞争和队列效果
- **性能对比**：实时对比手写链表版和 OrderedDict 版的操作延迟
- **交互控制**：调节容量、选择操作序列（预设或自定义）、单步/自动播放

> 打开 `animation.html` 查看交互动画

## 答案与总结

**核心 Insight**：LRU 的 $O(1)$ 实现依赖**哈希表 + 双向链表**的组合——哈希表负责 $O(1)$ 定位，链表负责 $O(1)$ 维护访问顺序。

**关键点**：
1. 伪头部和伪尾部节点**消除边界判断**，所有操作统一化
2. `get` 不只是返回值，还要**把节点移到尾部**（更新访问时间）
3. `put` 新节点时若容量满，**淘汰头部节点**（最久未使用）
4. 线程安全用 `RLock` 包裹公开方法，粒度为方法级

**复杂度**：
- 时间：`get` $O(1)$，`put` $O(1)$
- 空间：$O(capacity)$

**一句话总结**：哈希表让你找到它，双向链表让你记住谁最近被碰过——两者合体就是 LRU。
