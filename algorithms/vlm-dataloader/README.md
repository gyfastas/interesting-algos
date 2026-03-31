# VLM DataLoader：多进程生产者-消费者数据加载

## 问题描述

**在 VLM (Vision-Language Model) 训练中，写一个多进程 DataLoader，用生产者-消费者队列并行加载图像和文本数据，保证 GPU 不等数据。**

VLM 训练的数据管线瓶颈往往在 I/O 和图像解码。一条数据包含图像（需要读磁盘 + decode + resize）和文本（需要 tokenize），如果串行处理，GPU 大部分时间在等数据。

## 直觉分析

### 为什么需要多进程？

```
串行 DataLoader:
  GPU: [空闲等数据][训练][空闲等数据][训练]...
  CPU: [读图+decode][          ][读图+decode]...
  → GPU 利用率 < 50%

多进程 DataLoader:
  GPU:     [训练][训练][训练][训练]...
  Worker0: [读图A][读图D][读图G]...
  Worker1: [读图B][读图E][读图H]...
  Worker2: [读图C][读图F][读图I]...
  Queue:   [A,B,C] → GPU 随时有数据
  → GPU 利用率 ≈ 100%
```

### 为什么用多进程而不是多线程？

Python 的 GIL (Global Interpreter Lock) 让多线程无法真正并行执行 CPU 密集任务。图像 decode 和 resize 是 CPU 密集型 → 必须用多进程。

| | 多线程 | 多进程 |
|---|---|---|
| GIL | 受限，同一时刻只有一个线程执行 Python | 不受限，每个进程有自己的 GIL |
| I/O 密集 | 有效（等 I/O 时释放 GIL） | 有效 |
| CPU 密集 | **无效**（图像 decode 被 GIL 卡住） | **有效** |
| 内存 | 共享 | 独立（需要 IPC） |

### 生产者-消费者模型

```
                ┌──────────┐
  数据索引 ───→ │ Worker 0 │ ──┐
  (index_queue) │ Worker 1 │ ──┤──→ [output_queue] ──→ 主进程 collate → GPU
                │ Worker 2 │ ──┘     (prefetch_factor × batch_size)
                └──────────┘
                 生产者 (多个)           消费者 (1个)
```

- **Index Queue**：主进程把数据索引 (index) 发给 workers
- **Workers (生产者)**：从 index queue 取索引 → 读图/decode/tokenize → 放入 output queue
- **Output Queue**：预取缓冲区，存放处理好的样本
- **主进程 (消费者)**：从 output queue 取数据 → collate 成 batch → 送 GPU

## VLM 数据的特殊性

一条 VLM 训练数据：
```python
{
    "image": "path/to/image.jpg",           # 需要: 读磁盘 → decode → resize → normalize
    "conversations": [                       # 需要: tokenize → padding
        {"role": "user", "content": "描述这张图"},
        {"role": "assistant", "content": "图中是一只猫..."},
    ]
}
```

处理耗时分析：
| 步骤 | 耗时 | 类型 |
|------|------|------|
| 读磁盘 | 1-10ms | I/O |
| JPEG decode | 5-20ms | CPU |
| Resize + augment | 2-10ms | CPU |
| Tokenize | 0.5-2ms | CPU |
| **总计** | ~10-40ms/样本 | |
| GPU forward+backward | ~50-200ms/batch | GPU |

如果 batch_size=8，串行加载 8 张图 = 80-320ms → 可能超过 GPU 计算时间。4 个 worker 并行 → 20-80ms → GPU 不等待。

## 核心代码结构

```python
class VLMDataLoader:
    def __init__(self, dataset, batch_size, num_workers, prefetch_factor=2):
        self.index_queue = mp.Queue()       # 主进程 → workers
        self.output_queue = mp.Queue(        # workers → 主进程
            maxsize=prefetch_factor * batch_size
        )
        self.workers = [
            mp.Process(target=worker_fn,
                       args=(dataset, self.index_queue, self.output_queue))
            for _ in range(num_workers)
        ]

    def __iter__(self):
        # 1. 把所有索引打乱后放入 index_queue
        # 2. 启动 workers
        # 3. 从 output_queue 取数据，凑成 batch
        # 4. yield batch
```

## Collate 函数

VLM 的 batch 需要处理变长序列：

```python
def vlm_collate(samples):
    # 图像: stack 成 (B, C, H, W)
    images = torch.stack([s['image'] for s in samples])

    # 文本: pad 到同一长度
    max_len = max(len(s['input_ids']) for s in samples)
    input_ids = torch.full((B, max_len), pad_token_id)
    for i, s in enumerate(samples):
        input_ids[i, :len(s['input_ids'])] = s['input_ids']

    return {'images': images, 'input_ids': input_ids, ...}
```

## 关键设计决策

### 1. Queue 的 maxsize

```python
output_queue = mp.Queue(maxsize=prefetch_factor * batch_size)
```

- 太小：worker 产出快但 queue 满了 → worker 阻塞等待
- 太大：内存占用高（每个样本含图像 tensor）
- 经验值：`prefetch_factor=2`，即预取 2 个 batch

### 2. 优雅关闭 (Poison Pill)

```python
# 主进程发送 "毒丸" 告诉 worker 退出
for _ in workers:
    index_queue.put(None)  # None = 退出信号

# Worker 里
while True:
    idx = index_queue.get()
    if idx is None:  # 收到毒丸
        break
    ...
```

### 3. 异常处理

Worker 里图像可能损坏、路径不存在。不能让一个坏样本杀死整个 worker：

```python
try:
    sample = process(idx)
    output_queue.put(sample)
except Exception as e:
    output_queue.put(ErrorSample(idx, str(e)))
    # 或者直接跳过，取下一个索引
```

### 4. Seed 管理

每个 worker 进程需要不同的随机种子（否则 augmentation 完全一样）：

```python
def worker_fn(worker_id, ...):
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

## Prefetch：GPU 训练和数据加载完全重叠

### 问题

即使用了多进程 worker，主进程仍然是同步的：

```
主进程: [从 queue 取数据 + collate] → [GPU.train(batch)] → [从 queue 取数据 + collate] → ...
                                       ↑ 这段时间 queue 没人取 → workers 可能被 queue 满阻塞
```

`collate` + `to(device)` 也需要时间（尤其 pin_memory 拷贝）。如果这段时间 GPU 在等 → 利用率下降。

### Prefetch 的思想

**用一个后台线程提前把下一个 batch 准备好**，GPU 训练当前 batch 时，下一个 batch 已经在 GPU 显存里等着了。

```
无 prefetch:
  主线程:  [collate B0] [GPU train B0] [collate B1] [GPU train B1]
  GPU:     [  等待   ] [  train B0  ] [  等待   ] [  train B1  ]

有 prefetch (prefetch_factor=2):
  prefetch线程: [collate B0] [collate B1] [collate B2] ...
  主线程:                    [GPU train B0] [GPU train B1] [GPU train B2]
  GPU:                       [  train B0  ] [  train B1  ] [  train B2  ]
                              ↑ prefetch 线程已经在准备 B2 了
```

### prefetch_factor 的含义

`prefetch_factor` = **每个 worker 提前准备的样本数**。

PyTorch 的 `DataLoader(prefetch_factor=2, num_workers=4)`:
- 每个 worker 提前准备 2 个样本
- 总共 buffer 中有 `2 × 4 = 8` 个样本待消费
- 如果 `batch_size=4`，就是 2 个 batch 的预取量

```python
# PyTorch DataLoader 的 prefetch 实际上是通过控制
# 发给每个 worker 的"未完成任务数"来实现的:
#
# 每个 worker 的 index_queue 中最多有 prefetch_factor 个索引
# worker 完成一个 → 主进程再发一个 → 始终保持 prefetch_factor 个在飞
#
# 而不是简单的 output_queue.maxsize
```

### 实现：Prefetch Thread

```python
class PrefetchLoader:
    """在 DataLoader 外层包一层 prefetch thread + pin_memory。"""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        # 后台线程: 提前把 batch 搬到 GPU
        stream = torch.cuda.Stream()
        batch_queue = queue.Queue(maxsize=2)  # 预取 2 个 batch

        def prefetch_worker():
            for batch in self.loader:
                with torch.cuda.stream(stream):
                    batch = {k: v.to(self.device, non_blocking=True)
                             for k, v in batch.items()}
                batch_queue.put(batch)
            batch_queue.put(None)  # sentinel

        thread = threading.Thread(target=prefetch_worker, daemon=True)
        thread.start()

        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            torch.cuda.current_stream().wait_stream(stream)
            yield batch
```

关键细节：
- 用 **CUDA Stream** 做异步 H2D 拷贝，不阻塞 GPU 计算
- `non_blocking=True` 让 `.to(device)` 不等拷��完成就返回
- `wait_stream` 确保主 stream 用 batch 前拷贝已完成
- 这是一个**线程**不是进程——因为 H2D 拷贝释放 GIL，线程就够了

### prefetch 的三层含义

| 层级 | 谁做的 | 做什么 |
|------|--------|--------|
| **Worker prefetch** | `output_queue.maxsize` | Workers 提前处理好的样本缓冲在 queue 里 |
| **Collate prefetch** | prefetch thread | 提前从 queue 取样本 + collate 成 batch |
| **Device prefetch** | CUDA stream | 提前把 batch 从 CPU 拷贝到 GPU |

PyTorch 的 `prefetch_factor` 参数主要控制第一层。完整的 prefetch 需要三层都做。

## 对比 PyTorch DataLoader

我们的实现就是 `torch.utils.data.DataLoader` 的简化版。PyTorch 的实现更复杂：

| 功能 | 我们的实现 | PyTorch DataLoader |
|------|-----------|-------------------|
| 多进程 | `mp.Process` + `mp.Queue` | `mp.Process` + 共享内存 |
| IPC | Queue (pickle 序列化) | 共享内存 (零拷贝) |
| Pin memory | 未实现 | `pin_memory=True` → CUDA 传输更快 |
| Sampler | 简单 shuffle | 支持自定义 Sampler |
| 异常传播 | 简单处理 | worker 异常传回主进程 re-raise |

PyTorch 用**共享内存**而非 Queue 传数据，避免了 pickle 序列化的开销——这是主要的性能差异。

## 动画演示

> 打开 `animation.html` 查看多进程数据加载的动态流程可视化。

## 答案与总结

| 要点 | 结论 |
|------|------|
| 为什么多进程 | GIL 阻止多线程并行 CPU 密集任务 (图像 decode) |
| 核心模型 | 生产者-消费者：workers 生产样本 → Queue → 主进程消费 |
| Queue maxsize | `prefetch_factor × batch_size`，预取 2 个 batch |
| 优雅关闭 | Poison pill (None) 通知 worker 退出 |
| VLM 特殊性 | 图像 decode 重、文本变长需 pad、多模态 collate |
| vs PyTorch | PyTorch 用共享内存零拷贝，我们用 Queue 序列化 |

**一句话总结**：VLM DataLoader = 多个 worker 进程并行读图 + decode + tokenize，通过 Queue 传给主进程 collate 成 batch，核心是生产者-消费者模式保证 GPU 永远有数据吃。
