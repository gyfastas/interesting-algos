"""
LRU Cache 实现（基础版 + 线程安全进阶版）
==========================================

基础版: 手写双向链表 + 哈希表，O(1) get/put
进阶版: 基础版 + threading.Lock，保证线程安全

核心思想:
  - 哈希表: key -> node，O(1) 定位
  - 双向链表: 维护访问顺序，越靠近尾部越近被访问
  - 淘汰: 容量满时移除链表头部（最久未使用）
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# 基础版: 手写双向链表 + 哈希表
# =============================================================================

class DLinkedNode:
    """双向链表节点。"""
    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    """
    手写 LRU Cache。

    使用双向链表 + 哈希表实现 O(1) 的 get 和 put：
    - get: 哈希表查找 → 移动到尾部
    - put: 已存在则更新 + 移尾；不存在则新建 + 移尾 + 可能淘汰头部
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> DLinkedNode
        self.size = 0

        # 伪头部和伪尾部节点，方便操作
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: DLinkedNode):
        """从链表中移除节点。"""
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def _add_to_tail(self, node: DLinkedNode):
        """将节点添加到链表尾部（最近使用）。"""
        prev = self.tail.prev
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node

    def _move_to_tail(self, node: DLinkedNode):
        """将已有节点移到尾部。"""
        self._remove(node)
        self._add_to_tail(node)

    def _pop_head(self) -> DLinkedNode:
        """移除并返回链表头部节点（最久未使用）。"""
        node = self.head.next
        self._remove(node)
        return node

    def get(self, key: int) -> int:
        """获取值，若不存在返回 -1。访问后移到尾部。"""
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_tail(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        """插入或更新值。"""
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

    def keys(self) -> list[int]:
        """返回当前缓存中的 key 列表（从旧到新）。"""
        result = []
        cur = self.head.next
        while cur != self.tail:
            result.append(cur.key)
            cur = cur.next
        return result


# =============================================================================
# 进阶版: 线程安全 LRU Cache
# =============================================================================

class ThreadSafeLRUCache:
    """
    线程安全版 LRU Cache。

    在基础版基础上，用 threading.RLock 保护所有公开方法。
    RLock 允许同一线程多次获取锁（防止内部方法互相调用时死锁）。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.size = 0
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.lock = threading.RLock()

    def _remove(self, node: DLinkedNode):
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def _add_to_tail(self, node: DLinkedNode):
        prev = self.tail.prev
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node

    def _move_to_tail(self, node: DLinkedNode):
        self._remove(node)
        self._add_to_tail(node)

    def _pop_head(self) -> DLinkedNode:
        node = self.head.next
        self._remove(node)
        return node

    def get(self, key: int) -> int:
        with self.lock:
            if key not in self.cache:
                return -1
            node = self.cache[key]
            self._move_to_tail(node)
            return node.val

    def put(self, key: int, value: int) -> None:
        with self.lock:
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

    def keys(self) -> list[int]:
        with self.lock:
            result = []
            cur = self.head.next
            while cur != self.tail:
                result.append(cur.key)
                cur = cur.next
            return result


# =============================================================================
# 测试与验证
# =============================================================================

def test_basic():
    """基础功能测试。"""
    print("=" * 60)
    print("基础版 LRU Cache 测试")
    print("=" * 60)

    cache = LRUCache(capacity=3)

    operations = [
        ("put", 1, 10),
        ("put", 2, 20),
        ("put", 3, 30),
        ("keys", "当前缓存", None),
        ("get", 1, None),  # 访问 1，移到尾部
        ("keys", "访问 1 后", None),
        ("put", 4, 40),    # 容量满，淘汰最久未使用的 2
        ("keys", "插入 4 后", None),
        ("get", 2, None),  # 已淘汰，返回 -1
        ("get", 3, None),  # 存在
        ("put", 5, 50),    # 淘汰 1
        ("keys", "插入 5 后", None),
        ("put", 3, 300),   # 更新 3
        ("keys", "更新 3 后", None),
    ]

    for op, k, v in operations:
        if op == "put":
            cache.put(k, v)
            print(f"  put({k}, {v}) -> keys={cache.keys()}")
        elif op == "get":
            result = cache.get(k)
            print(f"  get({k}) = {result} -> keys={cache.keys()}")
        elif op == "keys":
            print(f"  [{k}] keys={cache.keys()}")


def test_edge_cases():
    """边界条件测试。"""
    print(f"\n{'='*60}")
    print("边界条件测试")
    print(f"{'='*60}")

    # 容量为 0
    cache = LRUCache(0)
    cache.put(1, 1)
    assert cache.get(1) == -1, "容量为 0 时无法存入"
    print("  ✓ 容量为 0: 无法存入任何数据")

    # 容量为 1
    cache = LRUCache(1)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == -1, "1 应被淘汰"
    assert cache.get(2) == 2, "2 应存在"
    print("  ✓ 容量为 1: 正确淘汰旧数据")

    # 重复 put 同一 key
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(1, 10)
    assert cache.get(1) == 10, "值应更新为 10"
    print("  ✓ 重复 put: 值正确更新")

    # get 不存在的 key
    cache = LRUCache(2)
    assert cache.get(99) == -1, "不存在应返回 -1"
    print("  ✓ get 不存在: 返回 -1")

    # 大量操作
    cache = LRUCache(100)
    for i in range(1000):
        cache.put(i, i * 10)
    assert len(cache.keys()) == 100, "容量限制生效"
    assert cache.get(900) == 9000, "最近的数据应存在"
    assert cache.get(0) == -1, "最早的数据应被淘汰"
    print("  ✓ 大量操作: 容量限制和淘汰策略正确")


def test_thread_safety():
    """线程安全测试。"""
    print(f"\n{'='*60}")
    print("线程安全版测试")
    print(f"{'='*60}")

    cache = ThreadSafeLRUCache(capacity=100)
    errors = []

    def worker(worker_id):
        try:
            for i in range(200):
                key = (worker_id * 1000 + i) % 150  # 多个线程竞争少量 key
                cache.put(key, key * 10)
                val = cache.get(key)
                if val != key * 10:
                    errors.append(f"worker {worker_id}: get({key})={val}, expected {key*10}")
                time.sleep(0.0001)  # 增加竞争概率
        except Exception as e:
            errors.append(str(e))

    # 启动多个线程并发操作
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if errors:
        print(f"  ✗ 发现 {len(errors)} 个错误")
        for e in errors[:5]:
            print(f"    {e}")
    else:
        print("  ✓ 10 线程 × 200 次操作：无数据竞争错误")

    print(f"  最终缓存大小: {len(cache.keys())}（容量=100）")

    # 对比：非线程安全版本在并发下会出问题
    print(f"\n  对比：非线程安全版在并发下...")
    unsafe_cache = LRUCache(capacity=100)
    unsafe_errors = []

    def unsafe_worker(worker_id):
        for i in range(100):
            key = (worker_id * 100 + i) % 80
            unsafe_cache.put(key, key * 10)
            val = unsafe_cache.get(key)
            if val != key * 10:
                unsafe_errors.append(f"mismatch")

    threads = []
    for i in range(8):
        t = threading.Thread(target=unsafe_worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    print(f"  非线程安全版出现 {len(unsafe_errors)} 次数据不一致")


def test_performance():
    """性能对比测试。"""
    print(f"\n{'='*60}")
    print("性能对比: 手写链表 vs OrderedDict")
    print(f"{'='*60}")

    from collections import OrderedDict

    class OrderedDictLRU:
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = OrderedDict()

        def get(self, key):
            if key not in self.cache:
                return -1
            self.cache.move_to_end(key)
            return self.cache[key]

        def put(self, key, value):
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    N = 100000
    cap = 1000

    # 手写链表版
    t0 = time.perf_counter()
    cache1 = LRUCache(cap)
    for i in range(N):
        cache1.put(i, i)
        cache1.get(i - 500)
    t1 = time.perf_counter() - t0

    # OrderedDict 版
    t0 = time.perf_counter()
    cache2 = OrderedDictLRU(cap)
    for i in range(N):
        cache2.put(i, i)
        cache2.get(i - 500)
    t2 = time.perf_counter() - t0

    print(f"  手写链表版: {t1*1000:.2f} ms")
    print(f"  OrderedDict: {t2*1000:.2f} ms")
    print(f"  速度比: {t2/t1:.2f}x")


def demo():
    test_basic()
    test_edge_cases()
    test_thread_safety()
    test_performance()


if __name__ == "__main__":
    demo()
