# 手写 LayerNorm 与 RMSNorm

## 问题描述

面试常考：**手写 LayerNorm 和 RMSNorm 的 forward**。

这两个是 Transformer 里最基础的组件。GPT-2 / BERT 用 LayerNorm，而 Llama / Gemma / DeepSeek 等现代大模型几乎全部换成了 **RMSNorm**。为什么？差异在哪？

## 直觉分析

### 为什么需要归一化？

神经网络训练时，每一层的输入分布会随着前面层的参数更新而不断变化（Internal Covariate Shift）。归一化的作用就是**把每一层的输入拉回到稳定的分布**，让训练更快更稳定。

### BatchNorm vs LayerNorm

BatchNorm 是沿 batch 维度归一化——对同一个特征，在 batch 内的所有样本上求均值方差。问题：
- 依赖 batch size，推理时需要 running mean/var
- 序列长度可变时，不同位置的统计量不一致

LayerNorm 换了个方向：**对同一个 token，在特征维度上归一化**。和 batch size 无关，天然适合自回归生成。

```
BatchNorm: 在 batch 维上归一化（跨样本，同一特征）
LayerNorm: 在特征维上归一化（同一样本，跨特征）

输入 (B, S, D):
  BatchNorm → 对 B×S 个位置求统计量，每个特征一组 (D 组)
  LayerNorm → 对 D 个特征求统计量，每个位置一组 (B×S 组)
```

### LayerNorm vs RMSNorm

LayerNorm 做两件事：① 减去均值（中心化）② 除以标准差（缩放）。

RMSNorm 的发现：**中心化（减均值）对效果几乎没有贡献**，去掉它可以省约 40% 计算量。

## 数学建模

### LayerNorm

给定输入向量 $\mathbf{x} \in \mathbb{R}^D$（一个 token 的特征）：

$$\mu = \frac{1}{D}\sum_{i=1}^{D} x_i$$

$$\sigma^2 = \frac{1}{D}\sum_{i=1}^{D} (x_i - \mu)^2$$

$$\text{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}$$

其中 $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^D$ 是可学习参数，$\epsilon$ 是数值稳定常数。

归一化后的输出满足：$\text{mean} \approx \beta$，$\text{var} \approx \gamma^2$（当 $\gamma=1, \beta=0$ 时，输出均值 0、方差 1）。

### RMSNorm

去掉均值，直接用 RMS (Root Mean Square) 做缩放：

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}$$

$$\text{RMSNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

没有 $\beta$（偏移项），也没有减均值操作。

归一化后：$\frac{1}{D}\sum x_i^2 \approx 1$（均方为 1），但均值不一定为 0。

### 两者的关系

LayerNorm 可以分解为：

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

而标准差 $\sigma$ 和 RMS 的关系是：

$$\sigma^2 = \text{RMS}(\mathbf{x} - \mu)^2 = \frac{1}{D}\sum(x_i - \mu)^2$$

$$\text{RMS}(\mathbf{x})^2 = \frac{1}{D}\sum x_i^2 = \sigma^2 + \mu^2$$

所以 **RMS ≥ σ**，当且仅当 $\mu = 0$ 时相等。RMSNorm 等价于"假设均值为零的 LayerNorm"。

## 代码实现

### LayerNorm

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (..., D)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

5 步：求均值 → 中心化 → 求方差 → 归一化 → 仿射变换（γ, β）。

### RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (..., D)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
```

3 步：求平方均值 → 开根号得 RMS → 缩放（只有 γ，没有 β）。

## 为什么现代大模型用 RMSNorm？

### 1. 计算更快

| 操作 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 求均值 $\mu$ | D 次加法 | — |
| 中心化 $x - \mu$ | D 次减法 | — |
| 求方差/RMS | D 次乘加 | D 次乘加 |
| 归一化 | D 次除法 | D 次除法 |
| 缩放 γ | D 次乘法 | D 次乘法 |
| 偏移 β | D 次加法 | — |
| **总计** | **~5D** | **~3D** |

省掉的是求均值 + 中心化 + 偏移，约 **40% 计算量**。在大模型中，Norm 操作出现在每层的两处（attention 前 + FFN 前），几十层下来节省可观。

### 2. 效果几乎一样

原论文 (Zhang & Sennrich, 2019) 在机器翻译任务上实验表明，去掉均值中心化后模型效果没有显著下降。直觉理解：

- 可学习的 $\gamma$ 已经能调节每个特征的尺度
- 在深层网络中，残差连接 + 归一化的组合，使得"是否中心化"对最终表征影响很小
- $\beta$ 偏移的功能可以被后续线性层的 bias 吸收

### 3. 参数更少

RMSNorm 每层少了 D 个 $\beta$ 参数。虽然占总参数量微不足道，但在工程上少一个 reduce 操作（求均值）意味着 kernel 更简单、更容易融合优化。

### 4. 实际使用情况

| 模型 | 归一化方式 | 位置 |
|------|-----------|------|
| GPT-2, BERT | LayerNorm | Post-Norm |
| GPT-3 | LayerNorm | Pre-Norm |
| Llama 1/2/3 | RMSNorm | Pre-Norm |
| Gemma | RMSNorm | Pre-Norm |
| DeepSeek-V2/V3 | RMSNorm | Pre-Norm |
| Qwen 2.5 | RMSNorm | Pre-Norm |

### Pre-Norm vs Post-Norm

顺带一提，现代模型还从 Post-Norm 切换到了 **Pre-Norm**：

```
Post-Norm (GPT-2):  x = LayerNorm(x + Attn(x))
Pre-Norm  (Llama):  x = x + Attn(RMSNorm(x))
```

Pre-Norm 让残差路径保持"干净"（没有经过归一化的扰动），梯度流更稳定，训练更容易收敛。

## 动画演示

> 打开 `animation.html` 查看交互动画，可视化归一化前后的分布变化和计算流程对比。

## 答案与总结

| 要点 | 结论 |
|------|------|
| LayerNorm | 减均值 + 除标准差 + γ缩放 + β偏移，5 步 |
| RMSNorm | 除 RMS + γ缩放，3 步，省 ~40% 计算 |
| 核心差异 | RMSNorm 去掉了均值中心化和偏移项 |
| 为什么可以去掉 | 中心化对深层网络效果贡献极小，γ 已足够 |
| 工程意义 | 少一次 reduce、少 D 个参数、kernel 更好优化 |

**一句话总结**：RMSNorm 是 LayerNorm 的"够用版"——去掉了不必要的均值中心化，用更少的计算达到几乎一样的效果。
