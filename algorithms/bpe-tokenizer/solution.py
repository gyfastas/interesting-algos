"""
BPE (Byte Pair Encoding) Tokenizer — 手写实现

包含: 训练 (learn merges) + 编码 (encode) + 解码 (decode)
"""

from collections import Counter


# ============================================================
# 1. BPE Tokenizer（字符级 BPE）
# ============================================================
class BPETokenizer:
    """
    经典 BPE 流程:
    1. 训练: 从字符级 vocab 出发，反复合并最高频的相邻 pair
    2. 编码: 按学到的 merge 顺序，贪心合并输入文本
    3. 解码: 把 token id 映射回字符串
    """

    def __init__(self):
        self.merges = []          # [(pair, merged_token), ...] 按学习顺序
        self.vocab = {}           # token_str -> id
        self.id_to_token = {}    # id -> token_str

    # ---------- 训练 ----------

    def train(self, text: str, num_merges: int = 50, verbose: bool = False):
        """
        从语料中学习 BPE merge 规则。

        Step 1: 把文本分成词（按空格），每个词拆成字符序列，末尾加 </w> 标记词边界
        Step 2: 统计所有相邻字符对的频率
        Step 3: 合并最高频的 pair → 产生新 token
        Step 4: 重复 Step 2-3 直��达到 num_merges 次
        """
        # Step 1: 预处理 — 词频统计，每个词拆成字符 tuple
        words = text.split()
        word_freqs = Counter(words)

        # 每个词 → 字符列表（加 </w> 词尾标记）
        # 例如 "low" → ('l', 'o', 'w', '</w>')
        splits = {}
        for word, freq in word_freqs.items():
            symbols = list(word) + ['</w>']
            splits[tuple(symbols)] = freq

        # 初始化字符级 vocab
        all_chars = set()
        for symbols in splits:
            all_chars.update(symbols)

        if verbose:
            print(f"初始 vocab ({len(all_chars)} 个字符): {sorted(all_chars)}")

        # Step 2-4: 反复合并
        for i in range(num_merges):
            # 统计所有相邻 pair 的频率
            pair_freqs = Counter()
            for symbols, freq in splits.items():
                for j in range(len(symbols) - 1):
                    pair_freqs[(symbols[j], symbols[j + 1])] += freq

            if not pair_freqs:
                break

            # 找最高频 pair
            best_pair = pair_freqs.most_common(1)[0]
            pair, count = best_pair
            new_token = pair[0] + pair[1]

            if verbose:
                print(f"Merge #{i+1}: '{pair[0]}' + '{pair[1]}' → '{new_token}' (freq={count})")

            self.merges.append((pair, new_token))

            # 在所有词中执行这次合并
            new_splits = {}
            for symbols, freq in splits.items():
                new_symbols = self._apply_merge(symbols, pair, new_token)
                new_splits[new_symbols] = freq
            splits = new_splits

        # 构建最终 vocab
        all_tokens = set()
        for symbols in splits:
            all_tokens.update(symbols)
        # 也加上所有 merge 中间产物（确保 encode 时能用）
        for _, token in self.merges:
            all_tokens.add(token)
        for symbols in splits:
            all_tokens.update(symbols)

        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

        if verbose:
            print(f"\n最终 vocab ({len(self.vocab)} tokens): {sorted(self.vocab.keys())}")

    @staticmethod
    def _apply_merge(symbols: tuple, pair: tuple, new_token: str) -> tuple:
        """在一个词的 symbol 序列中执行一次 merge。"""
        result = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        return tuple(result)

    # ---------- 编码 ----------

    def encode(self, text: str) -> list[int]:
        """
        编码文本为 token id 序列。

        对每个词，���字符级开始，按 merge 的学习顺序依次合并。
        """
        tokens = []
        for word in text.split():
            symbols = list(word) + ['</w>']

            # 按 merge 的学习顺序依次合并
            for pair, new_token in self.merges:
                symbols = list(self._apply_merge(tuple(symbols), pair, new_token))

            for s in symbols:
                if s in self.vocab:
                    tokens.append(self.vocab[s])
                else:
                    # OOV: 退回字符级
                    for ch in s:
                        if ch in self.vocab:
                            tokens.append(self.vocab[ch])
        return tokens

    # ---------- 解码 ----------

    def decode(self, ids: list[int]) -> str:
        """解码 token id 序列为文本。"""
        tokens = [self.id_to_token[i] for i in ids if i in self.id_to_token]
        text = ''.join(tokens)
        # 把 </w> 替换回空格
        text = text.replace('</w>', ' ')
        return text.strip()


# ============================================================
# 2. Byte-level BPE (BBPE) — GPT-2 / tiktoken 风格
# ============================================================
class ByteLevelBPE:
    """
    BBPE 和 BPE 的核心区别:
    - BPE:  初始 vocab = Unicode 字符集（可能很大或遇到 OOV 字符）
    - BBPE: 初始 vocab = 256 个字节（完全覆盖，零 OOV）

    所有文本先转成 UTF-8 字节序列，然后在字节上做 BPE。
    """

    def __init__(self):
        self.merges = []
        self.vocab = {bytes([i]): i for i in range(256)}  # 初始: 256 个单字节
        self.id_to_token = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, num_merges: int = 50, verbose: bool = False):
        """在 UTF-8 字节序列上训练 BPE���"""
        raw_bytes = text.encode('utf-8')

        # 把字节序列按空格对应的字节 (0x20) 分词
        # 简化处理: 直接对整个字节序列操作
        data = list(raw_bytes)

        if verbose:
            print(f"原始文本: {len(text)} 字符 → {len(data)} 字节")
            print(f"初始 vocab: 256 个字节 (0x00 ~ 0xFF)")

        for i in range(num_merges):
            # 统计相邻字节对频率
            pair_freqs = Counter()
            for j in range(len(data) - 1):
                pair_freqs[(data[j], data[j + 1])] += 1

            if not pair_freqs:
                break

            best_pair = pair_freqs.most_common(1)[0]
            pair, count = best_pair
            new_id = 256 + i

            # 新 token = 两个子 token 拼接
            new_token = self.id_to_token.get(pair[0], bytes([pair[0]])) + \
                        self.id_to_token.get(pair[1], bytes([pair[1]]))

            if verbose:
                p0 = self.id_to_token.get(pair[0], bytes([pair[0]]))
                p1 = self.id_to_token.get(pair[1], bytes([pair[1]]))
                try:
                    desc = new_token.decode('utf-8', errors='replace')
                except:
                    desc = repr(new_token)
                print(f"Merge #{i+1}: {repr(p0)} + {repr(p1)} → {repr(desc)} (freq={count})")

            self.merges.append((pair, new_id))
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token

            # 在 data 中执行合并
            new_data = []
            j = 0
            while j < len(data):
                if j < len(data) - 1 and data[j] == pair[0] and data[j + 1] == pair[1]:
                    new_data.append(new_id)
                    j += 2
                else:
                    new_data.append(data[j])
                    j += 1
            data = new_data

        if verbose:
            print(f"\n最终 vocab: {len(self.vocab)} tokens (256 字节 + {len(self.merges)} merges)")
            print(f"压缩率: {len(raw_bytes)} 字节 → {len(data)} tokens ({len(data)/len(raw_bytes)*100:.1f}%)")

    def encode(self, text: str) -> list[int]:
        """编码: 文本 → UTF-8 字节 → 按 merge 顺序合并。"""
        data = list(text.encode('utf-8'))
        for pair, new_id in self.merges:
            new_data = []
            j = 0
            while j < len(data):
                if j < len(data) - 1 and data[j] == pair[0] and data[j + 1] == pair[1]:
                    new_data.append(new_id)
                    j += 2
                else:
                    new_data.append(data[j])
                    j += 1
            data = new_data
        return data

    def decode(self, ids: list[int]) -> str:
        """解码: token ids → 字节 → UTF-8 文本。"""
        raw = b''.join(self.id_to_token.get(i, bytes([i % 256])) for i in ids)
        return raw.decode('utf-8', errors='replace')


# ============================================================
# 演示
# ============================================================
def demo():
    corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"

    print("=" * 65)
    print("BPE (字符级) 训练演示")
    print("=" * 65)

    bpe = BPETokenizer()
    bpe.train(corpus, num_merges=10, verbose=True)

    test = "low newest"
    ids = bpe.encode(test)
    decoded = bpe.decode(ids)
    print(f"\n编码 '{test}' → {ids}")
    print(f"解码 {ids} → '{decoded}'")

    print(f"\n{'=' * 65}")
    print("BBPE (字节级) 训练演示")
    print("=" * 65)

    # 含中文的语料 — 展示 BBPE 处理多语言的能力
    corpus_zh = "the cat sat on the mat the cat sat the cat the dog the dog sat"

    bbpe = ByteLevelBPE()
    bbpe.train(corpus_zh, num_merges=15, verbose=True)

    test_zh = "the cat"
    ids_zh = bbpe.encode(test_zh)
    decoded_zh = bbpe.decode(ids_zh)
    print(f"\n编码 '{test_zh}' → {ids_zh}")
    print(f"解码 {ids_zh} → '{decoded_zh}'")

    # BPE vs BBPE 对比
    print(f"\n{'=' * 65}")
    print("BPE vs BBPE 核心区别")
    print("=" * 65)
    print(f"{'':>20} {'BPE':>15} {'BBPE':>15}")
    print(f"{'-'*55}")
    print(f"{'初始 vocab':>20} {'Unicode 字符':>15} {'256 字节':>15}")
    print(f"{'输入单位':>20} {'字符':>15} {'UTF-8 字节':>15}")
    print(f"{'OOV 风险':>20} {'有':>15} {'无':>15}")
    print(f"{'多语言支持':>20} {'需扩充 vocab':>15} {'天然支持':>15}")
    print(f"{'中文处理':>20} {'每字符=1 token':>15} {'每字=3 字节起':>15}")
    print(f"{'代表实现':>20} {'原始 BPE':>15} {'GPT-2/tiktoken':>15}")


if __name__ == "__main__":
    demo()
