"""
VLM DataLoader — 多进程生产者-消费者数据加载

包含:
- VLMDataset: 数据集抽象 (读图 + tokenize)
- worker_fn: 生产者进程
- VLMDataLoader: 完整的多进程加载器
- vlm_collate: 变长序列 batch 整理
"""

import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import torch
import torch.nn.functional as F


# ============================================================
# 1. VLM Dataset — 单条数据的处理逻辑
# ============================================================
class VLMDataset:
    """
    VLM 数据集: 每条数据 = 图像路径 + 对话文本

    __getitem__ 完成:
    1. 读取图像 → decode → resize → normalize → tensor
    2. 对话文本 → tokenize → input_ids + labels
    """

    def __init__(
        self,
        data_list: list[dict],
        image_size: int = 336,
        max_seq_len: int = 2048,
        tokenizer=None,
    ):
        """
        data_list: [{"image": "path.jpg", "conversations": [...]}, ...]
        """
        self.data_list = data_list
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        """
        处理一条数据，返回:
        {
            "image": tensor (3, H, W),
            "input_ids": tensor (seq_len,),
            "labels": tensor (seq_len,),
            "attention_mask": tensor (seq_len,),
        }
        """
        item = self.data_list[idx]

        # --- 1. 图像处理 ---
        image = self._load_image(item["image"])

        # --- 2. 文本处理 ---
        input_ids, labels = self._process_conversations(item["conversations"])

        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        读取并预处理图像:
        1. 读文件 (I/O 密集)
        2. Decode JPEG/PNG (CPU 密集)
        3. Resize 到统一尺寸
        4. Normalize (ImageNet 均值/标准差)
        """
        try:
            # 实际项目用 PIL 或 torchvision
            # from PIL import Image
            # img = Image.open(image_path).convert("RGB")
            # img = img.resize((self.image_size, self.image_size))
            # tensor = transforms.ToTensor()(img)
            # tensor = transforms.Normalize(mean, std)(tensor)

            # 这里用模拟数据 (不依赖 PIL)
            # 模拟读盘 + decode 耗时
            time.sleep(0.005)  # 模拟 5ms I/O + decode
            image = torch.randn(3, self.image_size, self.image_size)
            # Normalize
            image = (image - image.mean()) / (image.std() + 1e-6)
            return image

        except Exception as e:
            # 图像损坏时返回黑图，不要让一个坏样本杀死 worker
            print(f"[WARNING] 图像加载失败 {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

    def _process_conversations(self, conversations: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对话文本 → token ids + labels

        VLM 的 label 规则:
        - 用户部分 (user turn): label = -100 (不计算 loss)
        - 模型部分 (assistant turn): label = input_ids (计算 loss)
        """
        if self.tokenizer is not None:
            # 实际 tokenizer 处理
            all_ids = []
            all_labels = []
            for turn in conversations:
                text = turn["content"]
                ids = self.tokenizer.encode(text)
                if turn["role"] == "user":
                    all_ids.extend(ids)
                    all_labels.extend([-100] * len(ids))  # 用户部分不计算 loss
                else:
                    all_ids.extend(ids)
                    all_labels.extend(ids)  # 模型部分计算 loss

            # Truncate
            all_ids = all_ids[:self.max_seq_len]
            all_labels = all_labels[:self.max_seq_len]
            return torch.tensor(all_ids), torch.tensor(all_labels)
        else:
            # 模拟: 随机长度的 token ids
            seq_len = random.randint(64, 512)
            input_ids = torch.randint(0, 32000, (seq_len,))
            labels = input_ids.clone()
            # 前 1/3 是 user turn → label = -100
            labels[:seq_len // 3] = -100
            return input_ids, labels


# ============================================================
# 2. Worker 函数 (生产者)
# ============================================================
def worker_fn(
    worker_id: int,
    dataset: VLMDataset,
    index_queue: mp.Queue,
    output_queue: mp.Queue,
    base_seed: int = 42,
):
    """
    Worker 进程的主循环 (生产者):

    1. 从 index_queue 取数据索引
    2. 调用 dataset[idx] 处理数据
    3. 把结果放入 output_queue
    4. 收到 None (毒丸) → 退出

    每个 worker 独立的随机种子，保证 augmentation 不同。
    """
    # 每个 worker 设置不同的随机种子
    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)

    while True:
        try:
            # 从 index queue 取任务
            idx = index_queue.get()

            # 毒丸 (Poison Pill): 收到 None 就退出
            if idx is None:
                break

            # 处理数据
            try:
                sample = dataset[idx]
                sample["__idx__"] = idx  # 附带索引，方便调试
                output_queue.put(sample)
            except Exception as e:
                # 单条数据出错不应杀死 worker
                # 放一个错误标记，主进程可以跳过
                output_queue.put({"__error__": True, "__idx__": idx, "__msg__": str(e)})

        except Exception:
            # Queue 被关闭等异常
            break


# ============================================================
# 3. VLM DataLoader (消费者 + 调度器)
# ============================================================
class VLMDataLoader:
    """
    多进程 VLM DataLoader，生产者-消费者模式。

    架构:
                     ┌──────────┐
      index_queue →  │ Worker 0 │ ──┐
                     │ Worker 1 │ ──┤→ output_queue → 主进程 collate → GPU
                     │ Worker 2 │ ──┘
                     └──────────┘
    """

    def __init__(
        self,
        dataset: VLMDataset,
        batch_size: int = 8,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self._workers: list[mp.Process] = []
        self._index_queue: mp.Queue | None = None
        self._output_queue: mp.Queue | None = None

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        每个 epoch:
        1. 创建 Queue
        2. 启动 worker 进程
        3. 把打乱的索引放入 index_queue
        4. 从 output_queue 取数据，凑成 batch
        5. yield batch
        6. 发送毒丸，等 worker 退出
        """
        # --- 1. 创建队列 ---
        # output_queue 的 maxsize 控制预取量
        # 太大占内存，太小 worker 会阻塞
        self._index_queue = mp.Queue()
        self._output_queue = mp.Queue(
            maxsize=self.prefetch_factor * self.batch_size
        )

        # --- 2. 启动 workers ---
        self._workers = []
        for wid in range(self.num_workers):
            w = mp.Process(
                target=worker_fn,
                args=(wid, self.dataset, self._index_queue, self._output_queue, self.seed),
                daemon=True,  # 主进程退出时自动杀死 worker
            )
            w.start()
            self._workers.append(w)

        # --- 3. 生成索引并放入 queue ---
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        # Drop last incomplete batch
        total = len(self)
        n_samples = total * self.batch_size if self.drop_last else len(indices)
        indices = indices[:n_samples]

        # 把索引全部放入 index_queue
        for idx in indices:
            self._index_queue.put(idx)

        # --- 4. 从 output_queue 取数据，组 batch ---
        batch_buffer = []
        samples_received = 0

        while samples_received < n_samples:
            sample = self._output_queue.get()
            samples_received += 1

            # 跳过错误样本
            if isinstance(sample, dict) and sample.get("__error__"):
                continue

            batch_buffer.append(sample)

            if len(batch_buffer) == self.batch_size:
                yield vlm_collate(batch_buffer)
                batch_buffer = []

        # 处理最后不足一个 batch 的数据
        if batch_buffer and not self.drop_last:
            yield vlm_collate(batch_buffer)

        # --- 5. 优雅关闭 ---
        self._shutdown()

    def _shutdown(self):
        """发送毒丸，等待 worker 退出。"""
        # 给每个 worker 发一个 None (毒丸)
        for _ in self._workers:
            self._index_queue.put(None)

        # 等待所有 worker 结束 (设超时避免死锁)
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()

        self._workers = []


# ============================================================
# 4. Collate 函数 — 把多个样本整理成一个 batch
# ============================================================
def vlm_collate(samples: list[dict]) -> dict:
    """
    VLM batch collate:
    - 图像: 直接 stack (所有图已 resize 到相同大小)
    - 文本: pad 到 batch 内最长长度
    - labels: pad 到最长，padding 部分 = -100 (不计算 loss)
    - attention_mask: 有效位置 = 1, padding = 0

    这是 VLM 训练最关键的 collate 逻辑:
    图像固定大小可以 stack，但文本长度不同必须 pad。
    """
    batch_size = len(samples)

    # 图像: (B, 3, H, W)
    images = torch.stack([s["image"] for s in samples])

    # 找 batch 内最长序列
    max_len = max(len(s["input_ids"]) for s in samples)

    # 初始化 padded tensors
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 = ignore
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, s in enumerate(samples):
        seq_len = len(s["input_ids"])
        input_ids[i, :seq_len] = s["input_ids"]
        labels[i, :seq_len] = s["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "images": images,               # (B, 3, H, W)
        "input_ids": input_ids,          # (B, max_len)
        "labels": labels,                # (B, max_len), padding=-100
        "attention_mask": attention_mask, # (B, max_len)
    }


# ============================================================
# 5. Prefetch Loader — 后台线程预取 + 异步 H2D 拷贝
# ============================================================
import threading
import queue as thread_queue


class PrefetchLoader:
    """
    在任意 DataLoader 外面包一层 prefetch:
    后台线程提前 collate + 搬到 GPU，主线程拿到的 batch 已经在 GPU 上了。

    原理:
      普通流程: [从 queue 取样本] → [collate] → [.to(device)] → [GPU train]
                 ↑ 这段时间 GPU 在等

      Prefetch: 后台线程不断做 [取样本 + collate + .to(device)]
                主线程直接从 prefetch buffer 拿已就绪的 GPU batch
                → GPU 训练和数据准备完全重叠

    关键实现细节:
    1. 用 threading.Thread 而非 mp.Process
       — .to(device) 底层是 CUDA API 调用，会释放 GIL，线程就够
    2. 用 CUDA Stream 做异步拷贝
       — non_blocking=True 不等拷贝完成就返回
       — wait_stream 确保主 stream 用 batch 前拷贝已完成
    3. prefetch_count 控制预取深度
       — 太大: 浪费 GPU 显存 (多个 batch 同时在显存)
       — 太小: 可能来不及准备 → GPU 等
       — 经验值: 2-3 个 batch
    """

    def __init__(self, loader, device: str = 'cuda', prefetch_count: int = 2):
        self.loader = loader
        self.device = device
        self.prefetch_count = prefetch_count

    def __iter__(self):
        # 预取队列: 存放已经在 GPU 上的 batch
        batch_queue = thread_queue.Queue(maxsize=self.prefetch_count)

        # 是否有 CUDA (CPU-only 环境下退化为普通迭代)
        use_cuda = self.device != 'cpu' and torch.cuda.is_available()

        # 创建独立的 CUDA stream 用于异步 H2D 拷贝
        stream = torch.cuda.Stream() if use_cuda else None

        def _prefetch_thread():
            """后台线程: 不断从 loader 取 batch → 搬到 GPU → 放入队列"""
            try:
                for batch in self.loader:
                    if use_cuda:
                        # 在独立 stream 上做异步 H2D 拷贝
                        with torch.cuda.stream(stream):
                            batch = {
                                k: v.to(self.device, non_blocking=True)
                                if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()
                            }
                    else:
                        batch = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }

                    batch_queue.put((batch, stream.record_event() if use_cuda else None))
            finally:
                batch_queue.put(None)  # sentinel: 告诉主线程结束

        # 启动后台预取线程
        thread = threading.Thread(target=_prefetch_thread, daemon=True)
        thread.start()

        # 主线程: 从预取队列消费
        while True:
            item = batch_queue.get()
            if item is None:
                break

            batch, event = item

            if use_cuda and event is not None:
                # 等待这个 batch 的 H2D 拷贝完成
                # (大部分情况下早就完成了，因为 prefetch 提前了)
                torch.cuda.current_stream().wait_event(event)

            yield batch

        thread.join()

    def __len__(self):
        return len(self.loader)


class PerWorkerPrefetchDataLoader:
    """
    PyTorch 风格的 per-worker prefetch:

    不是用一个大 output_queue.maxsize 来控制缓冲，
    而是控制每个 worker "手上" 未完成的任务数 = prefetch_factor。

    核心区别:
      简单 Queue maxsize:
        index_queue 里一次性塞所有索引 → workers 可能处理顺序不可控
        output_queue 满了 worker 才阻塞

      Per-worker prefetch:
        主进程按需给每个 worker 发索引 (每个 worker 最多 prefetch_factor 个未完成)
        → 更精准地控制内存使用
        → 保证各 worker 负载均衡

    这是 PyTorch DataLoader 的实际实现方式。
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 8,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        if self.drop_last:
            indices = indices[:len(indices) // self.batch_size * self.batch_size]

        # 每个 worker 有自己的 index queue (而不是一个共享的)
        worker_index_queues = [mp.Queue() for _ in range(self.num_workers)]
        output_queue = mp.Queue()

        # Worker 函数: 从自己的 index queue 取任务
        def _worker(wid, idx_q, out_q):
            random.seed(42 + wid)
            torch.manual_seed(42 + wid)
            while True:
                idx = idx_q.get()
                if idx is None:
                    break
                try:
                    sample = self.dataset[idx]
                    sample["__idx__"] = idx
                    sample["__wid__"] = wid
                    out_q.put(sample)
                except Exception as e:
                    out_q.put({"__error__": True, "__idx__": idx})

        # 启动 workers
        workers = []
        for wid in range(self.num_workers):
            w = mp.Process(target=_worker, args=(wid, worker_index_queues[wid], output_queue), daemon=True)
            w.start()
            workers.append(w)

        # === Per-worker prefetch 调度 ===
        # 每个 worker 的 "in-flight" 任务数
        worker_inflight = [0] * self.num_workers
        next_idx = 0  # 下一个要发的索引位置
        received = 0
        total = len(indices)

        # 初始填充: 每个 worker 发 prefetch_factor 个任务
        for wid in range(self.num_workers):
            for _ in range(self.prefetch_factor):
                if next_idx < total:
                    worker_index_queues[wid].put(indices[next_idx])
                    worker_inflight[wid] += 1
                    next_idx += 1

        # 主循环: 取一个结果 → 给那个 worker 补一个新任务
        batch_buffer = []
        while received < total:
            sample = output_queue.get()
            received += 1

            if isinstance(sample, dict) and sample.get("__error__"):
                continue

            # 给完成任务的 worker 补一个新索引 (保持 inflight = prefetch_factor)
            wid = sample.get("__wid__", received % self.num_workers)
            worker_inflight[wid] -= 1
            if next_idx < total and worker_inflight[wid] < self.prefetch_factor:
                worker_index_queues[wid].put(indices[next_idx])
                worker_inflight[wid] += 1
                next_idx += 1

            batch_buffer.append(sample)
            if len(batch_buffer) == self.batch_size:
                yield vlm_collate(batch_buffer)
                batch_buffer = []

        if batch_buffer and not self.drop_last:
            yield vlm_collate(batch_buffer)

        # 关闭
        for wid in range(self.num_workers):
            worker_index_queues[wid].put(None)
        for w in workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()


# ============================================================
# 6. 性能对比: 串行 vs 多进程
# ============================================================
class SimpleDataLoader:
    """串行 DataLoader (对比用): 主进程一个一个加载。"""

    def __init__(self, dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield vlm_collate(batch)
                batch = []


# ============================================================
# 演示
# ============================================================
def demo():
    # 模拟 VLM 训练数据
    N = 64
    data_list = [
        {
            "image": f"images/{i:06d}.jpg",
            "conversations": [
                {"role": "user", "content": f"Describe this image {i}"},
                {"role": "assistant", "content": f"This is a sample response for image {i}"},
            ],
        }
        for i in range(N)
    ]

    dataset = VLMDataset(data_list, image_size=224)

    print("=" * 65)
    print("VLM DataLoader — 多进程 vs 串行 性能对比")
    print("=" * 65)
    print(f"数据集: {N} 条, batch_size=8, image_size=224")

    # --- 串行 ---
    print(f"\n--- 串行 DataLoader ---")
    loader_serial = SimpleDataLoader(dataset, batch_size=8)
    t0 = time.perf_counter()
    batches_serial = 0
    for batch in loader_serial:
        batches_serial += 1
    t1 = time.perf_counter()
    print(f"  {batches_serial} batches, 耗时: {t1-t0:.3f}s")

    # --- 多进程 ---
    for nw in [2, 4]:
        print(f"\n--- 多进程 DataLoader (num_workers={nw}) ---")
        loader_mp = VLMDataLoader(dataset, batch_size=8, num_workers=nw, prefetch_factor=2)
        t0 = time.perf_counter()
        batches_mp = 0
        for batch in loader_mp:
            batches_mp += 1
            # 模拟 GPU 计算
            time.sleep(0.01)
        t1 = time.perf_counter()
        print(f"  {batches_mp} batches, 耗时: {t1-t0:.3f}s")

    # --- 展示 batch 结构 ---
    print(f"\n--- Batch 结构 ---")
    loader = VLMDataLoader(dataset, batch_size=4, num_workers=2)
    for batch in loader:
        print(f"  images:         {batch['images'].shape}")
        print(f"  input_ids:      {batch['input_ids'].shape}")
        print(f"  labels:         {batch['labels'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels 中 -100 比例: {(batch['labels'] == -100).float().mean():.1%} (user turn + padding)")
        break

    # --- 生产者消费者模型图 ---
    print(f"\n--- 架构 ---")
    print("""
    主进程 (调度 + 消费):
      ┌─ index_queue ──→ [0, 3, 7, 1, ...] ──→ workers
      │
      └─ output_queue ←─ [{img, ids}, ...] ←── workers
            ↓
         collate(batch)
            ↓
         GPU training
    """)


if __name__ == "__main__":
    # multiprocessing 在 macOS 上需要 spawn
    mp.set_start_method("spawn", force=True)
    demo()
