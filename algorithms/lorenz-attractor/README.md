# 洛伦兹吸引子（Lorenz Attractor）

> 动力系统 · 混沌理论 · 数值模拟 · ⭐⭐⭐

## 问题描述

1963 年，气象学家 Edward Lorenz 在研究大气对流时，将复杂的 Navier-Stokes 方程简化为一组三维常微分方程。他意外发现：这个看似简单的确定性系统，却能产生极其复杂、不可预测的轨迹——这就是**混沌**现象的数学起点。

Lorenz 系统的方程为：

$$
\begin{cases}
\displaystyle \frac{dx}{dt} = \sigma(y - x) \\[8pt]
\displaystyle \frac{dy}{dt} = x(\rho - z) - y \\[8pt]
\displaystyle \frac{dz}{dt} = xy - \beta z
\end{cases}
$$

标准参数：$\sigma = 10, \; \rho = 28, \; \beta = 8/3$

> 这组方程没有解析解，必须通过**数值方法**模拟。

## 直觉分析

想象一杯被底部加热的水：底部热水上升，顶部冷水下沉，形成对流环。Lorenz 用三个变量刻画这个系统的状态：
- $x$：对流环的流速
- $y$：上升流与下降流的温差
- $z$：竖直方向温度分布的非线性偏差

最惊人的发现是**蝴蝶效应**：即使你知道系统的全部方程和极其精确的初始条件，长期来看轨迹仍然是不可预测的——因为微小的初始误差会被指数级放大。

> "巴西的蝴蝶扇动翅膀，可能在德克萨斯引发龙卷风。" —— 混沌理论的通俗比喻

## 数学建模

### 方程的物理意义

| 变量 | 物理意义 | 作用 |
|------|---------|------|
| $x$ | 对流速度 | 描述环流的强度 |
| $y$ | 水平温差 | 驱动对流的温差力 |
| $z$ | 竖直温度偏差 | 反映温度分布的非对称性 |
| $\sigma$ | Prandtl 数 | 流体粘性 vs 热扩散的比值 |
| $\rho$ | Rayleigh 数 | 浮力 vs 粘性的比值（控制混沌程度） |
| $\beta$ | 几何因子 | 对流环的宽高比 |

### 不动点分析

令 $dx/dt = dy/dt = dz/dt = 0$，解得三个不动点：

1. **原点** $(0, 0, 0)$：无对流状态，当 $\rho > 1$ 时不稳定
2. **对称非零不动点** $C^\pm = (\pm\sqrt{\beta(\rho-1)}, \; \pm\sqrt{\beta(\rho-1)}, \; \rho-1)$

当 $\rho > 24.74$ 时，$C^\pm$ 也失去稳定性，系统进入**混沌态**，轨迹在 $C^+$ 和 $C^-$ 之间来回切换，但永不重复——形成著名的"蝴蝶"形奇异吸引子。

## 求解过程

### 数值积分：Euler 法 vs RK4

由于 Lorenz 方程没有解析解，我们用数值方法近似。对比两种方法：

**Euler 法（一阶）**：

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t \cdot f(\mathbf{x}_n)
$$

简单但精度差，在混沌系统中误差会迅速累积导致轨迹完全失真。

**Runge-Kutta 4 阶（RK4）**：

$$
\begin{aligned}
\mathbf{k}_1 &= f(\mathbf{x}_n) \\
\mathbf{k}_2 &= f\left(\mathbf{x}_n + \frac{\Delta t}{2}\mathbf{k}_1\right) \\
\mathbf{k}_3 &= f\left(\mathbf{x}_n + \frac{\Delta t}{2}\mathbf{k}_2\right) \\
\mathbf{k}_4 &= f(\mathbf{x}_n + \Delta t \cdot \mathbf{k}_3) \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}
$$

RK4 的局部截断误差为 $O(\Delta t^5)$，是混沌模拟的标准工具。

### Lyapunov 指数：混沌的数学判据

混沌系统的核心特征是**对初始条件的敏感依赖**。用 Lyapunov 指数 $\lambda$ 量化：

$$
\delta(t) \approx \delta_0 \cdot e^{\lambda t}
$$

- $\lambda > 0$：相邻轨迹指数分离 → **混沌**
- $\lambda = 0$：中性（周期/准周期）
- $\lambda < 0$：轨迹收敛 → 稳定

Lorenz 系统的理论最大 Lyapunov 指数约为 $\lambda \approx 0.906$，意味着大约每 $t \approx 0.77$（约 77 个时间步，$\Delta t = 0.01$）初始偏差就会翻倍。这也被称为系统的**可预测时间尺度**。

## 代码实现

### 核心 RK4 积分

```python
def lorenz_rhs(state):
    x, y, z = state
    return (
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    )

def rk4_step(state, dt):
    x, y, z = state
    k1x, k1y, k1z = lorenz_rhs((x, y, z))
    k2x, k2y, k2z = lorenz_rhs((x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z))
    k3x, k3y, k3z = lorenz_rhs((x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z))
    k4x, k4y, k4z = lorenz_rhs((x + dt * k3x, y + dt * k3y, z + dt * k3z))

    x_new = x + dt / 6.0 * (k1x + 2*k2x + 2*k3x + k4x)
    y_new = y + dt / 6.0 * (k1y + 2*k2y + 2*k3y + k4y)
    z_new = z + dt / 6.0 * (k1z + 2*k2z + 2*k3z + k4z)
    return x_new, y_new, z_new
```

### Lyapunov 指数估算

```python
def estimate_lyapunov(state0, dt=0.01, n_steps=15000, perturbation=1e-10):
    ref = state0
    perturbed = (state0[0] + perturbation, state0[1], state0[2])
    sum_log = 0.0
    count = 0

    for step in range(n_steps):
        ref = rk4_step(ref, dt)
        perturbed = rk4_step(perturbed, dt)

        if step % 10 == 0 and step > 0:
            dx = perturbed[0] - ref[0]
            dy = perturbed[1] - ref[1]
            dz = perturbed[2] - ref[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > 1e-15:
                sum_log += math.log(dist / perturbation)
                count += 1
            # 重新归一化
            scale = perturbation / max(dist, 1e-15)
            perturbed = (ref[0] + dx*scale, ref[1] + dy*scale, ref[2] + dz*scale)

    return sum_log / (n_steps * dt)
```

## 动画演示

动画包含两个核心可视化：

- **3D 轨迹视图**：实时计算 RK4 轨迹，用彩虹色渐变绘制蝴蝶形奇异吸引子。支持鼠标拖拽旋转视角，可实时调节 $\sigma$、$\rho$、$\beta$ 三个参数观察吸引子形态变化
- **蝴蝶效应视图**：两条轨迹从仅相差 $10^{-10}$ 的初始条件出发，实时展示它们如何指数分离。同时绘制偏差距离的指数增长曲线
- **Monte Carlo**：从多个随机初始条件同时发射粒子，展示它们最终都收敛到同一吸引子上
- **交互控制**：播放/暂停/速度调节/重置/参数滑块

> 打开 `animation.html` 查看交互动画

## 答案与总结

**核心 Insight**：Lorenz 吸引子是**确定性混沌**的完美范例——系统完全由确定性方程支配，没有随机性，但长期行为却不可预测。

**关键发现**：
1. $\rho < 1$：系统收敛到原点（无对流）
2. $1 < \rho < 24.74$：系统收敛到稳定的对流状态（$C^\pm$）
3. $\rho > 24.74$：混沌态，蝴蝶形奇异吸引子

**数值要点**：
- 必须用 RK4 等高阶方法，Euler 法在混沌系统中迅速失效
- 步长 $\Delta t$ 不能太大，否则数值误差会掩盖真实动力学
- Lyapunov 指数 $\lambda \approx 0.906$ 是可预测性的极限：大约 $t \approx \ln(2)/\lambda \approx 0.77$（约 77 步）后误差翻倍

**复杂度**：
- 时间：$O(N)$，$N$ 为模拟步数
- 空间：$O(N)$，存储轨迹点（若实时绘制可优化为 $O(1)$）
