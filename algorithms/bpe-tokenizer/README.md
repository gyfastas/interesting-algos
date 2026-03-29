# BPE Tokenizer：原理、实现与 BBPE 对比

## 问题描述

**BPE 是怎么工作的？手写一个 BPE tokenizer。BBPE 和 BPE 有什么区别？**

BPE (Byte Pair Encoding) 是现代 LLM 最主流的 tokenization 算法。GPT、Llama、DeepSeek 的 tokenizer 都基于 BPE 的变体。

## 直觉分析

### 为什么不直接用字符或单词？

| 方案 | 问题 |
|------|------|
| **字符级** | 序列太长（"attention" = 9 个 token），模型难学远距离依赖 |
| **单词级** | vocab 爆炸（英语 50 万+ 词），OOV 严重（"unhappiness" 没见过就 `<UNK>`） |
| **BPE** | 折中：高频词是一个 token，低频词被拆成子词（subword） |

BPE 的效果：
```
"tokenization" → ["token", "ization"]     ← 高频子词组合
"unhappiness"  → ["un", "happiness"]      ← 词根 + 词缀
"低频罕见词xyz" → ["低", "频", "罕", "见", "词", "x", "y", "z"]  ← 回退到字符
```

## BPE 算法

### 训练（学习 merge 规则）

```
输入: 训练语料
输出: 一个有序的 merge 列表 [(pair, new_token), ...]

Step 1: 预处理
  - ���空格分词
  - 每个词拆成字符序列，末尾加 </w> 标记词边界
  - 统计词频

  例: "low low lower newest" →
    ('l','o','w','</w>')     : 2
    ('l','o','w','e','r','</w>') : 1
    ('n','e','w','e','s','t','</w>') : 1

Step 2: 统计所有相邻 pair 的频率

Step 3: 合并频率最高的 pair → 产生新 token
  例: ('e','s') 出现最多 → 合并为 'es'

Step 4: 重复 Step 2-3，直到达到目标 vocab 大小
```

### 具体例子

语料：`low low low low lower lower newest newest newest newest`

```
初始:
  l o w </w>           × 4
  l o w e r </w>       × 2
  n e w e s t </w>     × 4

Merge #1: ('e','s') → 'es'  (freq=4, 来自 newest)
  l o w </w>           × 4
  l o w e r </w>       × 2
  n e w es t </w>      × 4

Merge #2: ('es','t') → 'est' (freq=4)
  l o w </w>           × 4
  l o w e r </w>       × 2
  n e w est </w>       × 4

Merge #3: ('est','</w>') → 'est</w>' (freq=4)
  l o w </w>           × 4
  l o w e r </w>       × 2
  n e w est</w>        × 4

Merge #4: ('l','o') → 'lo' (freq=6)
  lo w </w>            × 4
  lo w e r </w>        × 2
  n e w est</w>        × 4

Merge #5: ('lo','w') → 'low' (freq=6)
  low </w>             × 4
  low e r </w>         × 2
  n e w est</w>        × 4

...以此类推
```

### 编码

```python
def encode(text, merges):
    # 每个词从字符级开始
    symbols = list("newest") + ["</w>"]
    # ['n', 'e', 'w', 'e', 's', 't', '</w>']

    # 按 merge 学习顺序依次合并
    for pair, new_token in merges:
        apply_merge(symbols, pair, new_token)
    # → ['n', 'e', 'w', 'est</w>']

    return [vocab[s] for s in symbols]
```

### 解码

```python
def decode(ids):
    tokens = [id_to_token[i] for i in ids]
    text = ''.join(tokens).replace('</w>', ' ')
    return text.strip()
```

## 核心代码

```python
class BPETokenizer:
    def train(self, text, num_merges):
        # 1. 分词 + 字符拆分 + 词频统计
        word_freqs = Counter(text.split())
        splits = {tuple(list(w) + ['</w>']): f for w, f in word_freqs.items()}

        for i in range(num_merges):
            # 2. 统计相邻 pair 频率
            pair_freqs = Counter()
            for symbols, freq in splits.items():
                for j in range(len(symbols) - 1):
                    pair_freqs[(symbols[j], symbols[j+1])] += freq

            # 3. 合并最高频 pair
            best_pair = pair_freqs.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]
            self.merges.append((best_pair, new_token))

            # 4. 在所有词中执行合并
            splits = {merge(s, best_pair, new_token): f for s, f in splits.items()}

    def encode(self, text):
        for word in text.split():
            symbols = list(word) + ['</w>']
            for pair, new_token in self.merges:
                symbols = apply_merge(symbols, pair, new_token)
            yield from [self.vocab[s] for s in symbols]
```

## BBPE (Byte-level BPE) vs BPE

### BPE 的问题

经典 BPE 在**字符**上操作。但什么是"字符"？

- ASCII 英文：26 个字母，没问题
- Unicode 全集：14 万+ 字符（中文、日文、emoji、数学符号...）
- 如果训练语料没见过某个字符 → **OOV**，只能用 `<UNK>`

### BBPE 的解决方案

**不在字符上做 BPE，在字节 (byte) 上做**。

任何文本先编码为 UTF-8 字节序列（0~255），然后在字节上执行 BPE。

```
BPE:   "猫" → 如果没见过 → <UNK>
BBPE:  "猫" → UTF-8: [0xE7, 0x8C, 0xAB] → 3 个字节 token → 永远不会 OOV
```

### 核心区别

| | BPE | BBPE |
|---|---|---|
| **初始 vocab** | 训练语料中的所有 Unicode 字符 | 固定 256 个字节 (0x00~0xFF) |
| **输入单位** | 字符 | UTF-8 字节 |
| **OOV** | 有风险（没见过的字符） | **不可能**（所有字节都在 vocab 里） |
| **初始 vocab 大小** | 不确定（可能上千） | 固定 256 |
| **中文/Emoji** | 需要在训练语料中出现 | 天然支持，UTF-8 编码即可 |
| **代表** | 原始 BPE (Sennrich 2016) | GPT-2, GPT-4, Llama, tiktoken |

### BBPE 处理中文

```
"你好" 的 UTF-8 编码:
  "你" → [0xE4, 0xBD, 0xA0]  (3 字节)
  "好" → [0xE5, 0xA5, 0xBD]  (3 字节)

BBPE 初始: 6 个字节 token
训练后: 如果 "你好" 高频，会被合并成 1-2 个 token
```

中文每个字至少 3 个 UTF-8 字节，所以 BBPE 天然对中文不太"经济"——这也是为什么中文 LLM 通常需要更大的 vocab（Qwen 用 15 万+ token）。

### GPT-2 的 BBPE 实现细节

GPT-2 做了一个额外的 trick：用一个 **字节到 Unicode 的映射表**把 256 个字节映射成可打印的 Unicode 字符，这样可以用字符串操作来做 BPE，而不是裸字节操作。

```python
# GPT-2 的 byte_encoder: 字节 → 可打印 Unicode
# 0x41 ('A') → 'A'  (本身就可打印)
# 0x00 (null) → 'Ā'  (映射到一个可打印字符)
```

这是纯工程 trick，不影响算法本质。

## 现代 Tokenizer 全景

| Tokenizer | 算法 | 用于 |
|-----------|------|------|
| BPE | 字符级 BPE | 原始论文 |
| BBPE | 字节级 BPE | GPT-2/3/4, tiktoken |
| SentencePiece BPE | Unigram 或 BPE | Llama, T5 |
| WordPiece | 类 BPE（用似然而非频率） | BERT |
| tiktoken | BBPE + regex 预分词 | OpenAI 系列 |

现在主流就两种：**BBPE**（tiktoken 风格）和 **SentencePiece BPE**。

## 动画演示

> 打开 `animation.html` 查看交互动画，逐步展示 BPE 的 merge 过程。

## 答案与总结

| 要点 | 结论 |
|------|------|
| BPE 核心 | 反复合并最高频相邻 pair，从字符到子词 |
| 训练复杂度 | 每次 merge 需要遍历所有词，共 M 次 merge → O(M·N) |
| 编码 | 按 merge 顺序依次合并，贪心 |
| BBPE vs BPE | BBPE 在字节上做，初始 vocab=256，零 OOV |
| 为什么用 BBPE | 多语言天然支持，不需要预定义字符集 |

**一句话总结**：BPE 就是"反复合并最频繁的邻居"——在字符上做是经典 BPE，在字节上做是 BBPE，后者零 OOV 所以现在人人都用。
