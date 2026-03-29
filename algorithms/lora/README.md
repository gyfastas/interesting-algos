# LoRA：低秩适配的原理与实现

## 问题描述

**LoRA 的公式是什么？怎么 apply 到模型上？训练和推理时分别怎么工作？**

LoRA (Low-Rank Adaptation) 是大模型微调的事实标准。核心思想极其简单：**冻结原始权重，用两个小矩阵的乘积来近似权重更新**。

## 核心公式

原始线性层：

$$y = Wx$$

LoRA 修改后：

$$y = Wx + \frac{\alpha}{r} \cdot BAx$$

$$= (W + \frac{\alpha}{r} \cdot BA)x$$

其中：
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$ — 冻结的原始权重
- $A \in \mathbb{R}^{r \times d_{in}}$ — 下投影（降维）
- $B \in \mathbb{R}^{d_{out} \times r}$ — 上投影（升维）
- $r \ll \min(d_{in}, d_{out})$ — 秩（rank），通常 4~64
- $\alpha$ — 缩放因子，控制 LoRA 的影响幅度

**$\Delta W = \frac{\alpha}{r} \cdot BA$ 就是 LoRA 学到的权重增量。**

### 初始化

- $A$：Kaiming 均匀初始化（和正常 Linear 一样）
- $B$：**全零初始化**

这保证了训练开始时 $\Delta W = B \cdot A = 0$，模型行为和原始模型完全一致。

### 为什么是低秩？

$W$ 是 $d_{out} \times d_{in}$ 的矩阵，参数量 $d_{out} \cdot d_{in}$。

$BA$ 是 $d_{out} \times d_{in}$ 的矩阵，但只有 rank $r$。参数量 $(d_{out} + d_{in}) \cdot r$。

当 $r \ll d$ 时，参数量大幅减少：

$$\text{比例} = \frac{(d_{out} + d_{in}) \cdot r}{d_{out} \cdot d_{in}} \approx \frac{2r}{d}$$

以 $d = 4096, r = 8$ 为例：$\frac{2 \times 8}{4096} = 0.39\%$。

## 数据流：训练时 vs 推理时

### 训练时：两条路径并行

```
         ┌─────────────────────────────┐
         │         W (冻结)             │
x ──────►│    y_base = W @ x           │──► y_base
         └─────────────────────────────┘
    │                                        │
    │    ┌──────┐    ┌──────┐                │
    └──► │ A    │──► │ B    │──► (α/r) ──► + ──► y
         │(r,in)│    │(out,r)│   scale       │
         └──────┘    └──────┘                │
         ← 只训练这两个 →                     │
```

- W 的梯度不计算（`requires_grad=False`）
- 只有 A 和 B 接收梯度

### 推理时：Merge 成一个矩阵

$$W' = W + \frac{\alpha}{r} \cdot BA$$

```
x ──────► W' @ x ──► y
```

Merge 后就是一个普通的 `nn.Linear`，**零额外推理开销**。没有旁路、没有额外的矩阵乘法。

## 怎么 Apply 到模型上

### Step 1: 选择目标模块

不是每个 Linear 都需要加 LoRA。常见选择：

| 目标模块 | 说明 |
|---------|------|
| `q_proj, v_proj` | Attention 的 Q 和 V（最常见，原始论文推荐） |
| `q_proj, k_proj, v_proj, o_proj` | 全部 Attention 投影 |
| 所有 Linear | 包括 FFN，效果更好但参数更多 |

### Step 2: 替换 Linear 为 LoRALinear

遍历模型的所有 `nn.Linear`，如果名字匹配目标模块，就替换为 `LoRALinear`：

```python
def apply_lora(model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"]):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
            lora_layer = LoRALinear(module, rank, alpha)
            # 在父模块中替换
            parent = get_parent(model, name)
            setattr(parent, name.split(".")[-1], lora_layer)
    return model
```

### Step 3: 只训练 LoRA 参数

```python
# apply_lora 内部已经冻结了原始权重
# 优化器只收到 LoRA 参数
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

### Step 4: 推理时 Merge

```python
# 训练完成后
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.merge()
# 现在 model 就是普通的 Transformer，可以正常部署
```

## LoRALinear 完整实现

```python
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank

        # 冻结原始权重
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        # LoRA 旁路
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    def forward(self, x):
        base = self.linear(x)
        if self.merged:
            return base
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * lora

    def merge(self):
        if not self.merged:
            self.linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.linear.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False
```

## $\alpha$ 和 $r$ 的关系

$$\text{scaling} = \frac{\alpha}{r}$$

- **$r$ 越大**：LoRA 的表达能力越强，但参数越多
- **$\alpha$ 越大**：LoRA 的影响越大，偏离原始模型越远
- **惯例**：$\alpha = 2r$ 或 $\alpha = r$，常见组合如 $r=8, \alpha=16$

调参时通常固定 $\alpha$（如 16），只调 $r$。这样改变 $r$ 时 scaling 自动调整，不需要同时调学习率。

## 参数量对比

| 模型 | 全量微调 | LoRA (r=8, q+v) | 比例 |
|------|---------|-----------------|------|
| 7B (d=4096, 32层) | ~6.7B | 4.2M | 0.06% |
| 13B (d=5120, 40层) | ~13B | 6.6M | 0.05% |
| 70B (d=8192, 80层) | ~70B | 21.0M | 0.03% |

## 扩展：QLoRA

QLoRA = 量化基座 + LoRA：

1. 把基座模型量化到 4-bit（NF4 格式）
2. LoRA 的 A、B 仍然是 bf16/fp16
3. Forward 时：4-bit W 反量化 → fp16 → 加上 LoRA 旁路
4. 只有 LoRA 参数接收梯度

效果：70B 模型微调只需 ~48GB 显存（单张 A100），而全量微调需要 >1TB。

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化矩阵分解、参数量对比和 apply 流程。

## 答案与总结

| 要点 | 结论 |
|------|------|
| 核心公式 | $y = Wx + \frac{\alpha}{r} BA x$，两个小矩阵近似权重增量 |
| Apply | 遍历 model，替换目标 Linear 为 LoRALinear，冻结原始权重 |
| 训练 | 只有 A、B 有梯度，参数量 < 0.1% |
| 推理 | Merge: $W' = W + \frac{\alpha}{r}BA$，零额外开销 |
| 初始化 | A=Kaiming, B=零 → 训练开始时 $\Delta W = 0$ |

**一句话总结**：LoRA 就是在冻结的大矩阵旁边放两个小矩阵做乘法，训练时走旁路，推理时合并回去——参数省 1000 倍，效果几乎一样。
