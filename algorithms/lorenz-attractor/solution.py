"""
Lorenz Attractor（洛伦兹吸引子）模拟
======================================

经典的三维混沌动力系统，由 Edward Lorenz (1963) 提出，
最初用于模拟大气对流。

微分方程组:
  dx/dt = σ(y - x)
  dy/dt = x(ρ - z) - y
  dz/dt = xy - βz

标准参数: σ=10, ρ=28, β=8/3

数值方法:
  - Euler 法（一阶，稳定性差，用于教学对比）
  - RK4（四阶 Runge-Kutta，精度高，标准方法）

验证:
  - 轨迹收敛到奇异吸引子（蝴蝶形）
  - 蝴蝶效应：初始条件敏感依赖
  - Lyapunov 指数估算
"""

import math
import random


# =============================================================================
# Lorenz 方程
# =============================================================================

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz(state: tuple[float, float, float]) -> tuple[float, float, float]:
    """计算 Lorenz 方程的右端项 f(x, y, z)。"""
    x, y, z = state
    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    return dx, dy, dy, dz


def lorenz_rhs(state):
    """返回 (dx, dy, dz) 三元组。"""
    x, y, z = state
    return (
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    )


# =============================================================================
# 数值积分方法
# =============================================================================

def euler_step(state, dt):
    """Euler 单步推进。"""
    dx, dy, dz = lorenz_rhs(state)
    x, y, z = state
    return x + dx * dt, y + dy * dt, z + dz * dt


def rk4_step(state, dt):
    """四阶 Runge-Kutta 单步推进。"""
    x, y, z = state

    k1x, k1y, k1z = lorenz_rhs((x, y, z))
    k2x, k2y, k2z = lorenz_rhs((x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z))
    k3x, k3y, k3z = lorenz_rhs((x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z))
    k4x, k4y, k4z = lorenz_rhs((x + dt * k3x, y + dt * k3y, z + dt * k3z))

    x_new = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_new = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    z_new = z + dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
    return x_new, y_new, z_new


def simulate(state0, dt, n_steps, method='rk4'):
    """
    数值模拟 Lorenz 系统。

    参数:
        state0: 初始状态 (x0, y0, z0)
        dt: 时间步长
        n_steps: 步数
        method: 'euler' 或 'rk4'
    返回:
        xs, ys, zs: 三个坐标的轨迹列表
    """
    step_func = euler_step if method == 'euler' else rk4_step
    xs, ys, zs = [state0[0]], [state0[1]], [state0[2]]
    state = state0
    for _ in range(n_steps):
        state = step_func(state, dt)
        xs.append(state[0])
        ys.append(state[1])
        zs.append(state[2])
    return xs, ys, zs


# =============================================================================
# 验证：蝴蝶效应 & Lyapunov 指数
# =============================================================================

def butterfly_effect_test(state0, perturbation=1e-10, dt=0.01, n_steps=5000):
    """
    蝴蝶效应验证：两个极其接近的初始条件，轨迹偏差随时间指数增长。

    返回:
        times, distances: 时间序列和对应的两轨距离
    """
    state1 = state0
    state2 = (state0[0] + perturbation, state0[1], state0[2])

    times = [0.0]
    distances = [perturbation]

    for i in range(1, n_steps + 1):
        state1 = rk4_step(state1, dt)
        state2 = rk4_step(state2, dt)
        dist = math.sqrt(
            (state1[0] - state2[0]) ** 2 +
            (state1[1] - state2[1]) ** 2 +
            (state1[2] - state2[2]) ** 2
        )
        times.append(i * dt)
        distances.append(dist)

    return times, distances


def estimate_lyapunov(state0, dt=0.01, n_steps=8000, perturbation=1e-10, rescale_interval=10):
    """
    估算最大 Lyapunov 指数。

    方法：每隔 rescale_interval 步，将偏离轨重置回参考轨的切方向，
    累加 log(距离/perturbation) / (总时间)。

    正的 Lyapunov 指数是混沌的数学标志。
    """
    ref = state0
    perturbed = (state0[0] + perturbation, state0[1], state0[2])
    sum_log = 0.0
    count = 0

    for step in range(n_steps):
        ref = rk4_step(ref, dt)
        perturbed = rk4_step(perturbed, dt)

        if step % rescale_interval == 0 and step > 0:
            dx = perturbed[0] - ref[0]
            dy = perturbed[1] - ref[1]
            dz = perturbed[2] - ref[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > 1e-15:
                sum_log += math.log(dist / perturbation)
                count += 1
            # 重新归一化到 perturbation 方向
            scale = perturbation / max(dist, 1e-15)
            perturbed = (ref[0] + dx * scale, ref[1] + dy * scale, ref[2] + dz * scale)

    total_time = n_steps * dt
    lyapunov = sum_log / total_time if total_time > 0 else 0
    return lyapunov


# =============================================================================
# 演示与验证
# =============================================================================

def demo():
    print("=" * 60)
    print("Lorenz Attractor 模拟")
    print("=" * 60)
    print(f"参数: σ={SIGMA}, ρ={RHO}, β={BETA:.4f}")
    print()

    # 1. 标准轨迹
    print("[1] 生成标准轨迹 (RK4, dt=0.01, 8000步)")
    state0 = (1.0, 1.0, 1.0)
    xs, ys, zs = simulate(state0, dt=0.01, n_steps=8000, method='rk4')
    print(f"  初始: ({state0[0]}, {state0[1]}, {state0[2]})")
    print(f"  终点: ({xs[-1]:.4f}, {ys[-1]:.4f}, {zs[-1]:.4f})")
    print(f"  轨迹范围: x∈[{min(xs):.2f}, {max(xs):.2f}], y∈[{min(ys):.2f}, {max(ys):.2f}], z∈[{min(zs):.2f}, {max(zs):.2f}]")
    print()

    # 2. Euler vs RK4 对比
    print("[2] Euler vs RK4 对比 (1000步)")
    xs_e, ys_e, zs_e = simulate(state0, dt=0.01, n_steps=1000, method='euler')
    xs_r, ys_r, zs_r = simulate(state0, dt=0.01, n_steps=1000, method='rk4')
    diff = math.sqrt(
        (xs_e[-1] - xs_r[-1]) ** 2 +
        (ys_e[-1] - ys_r[-1]) ** 2 +
        (zs_e[-1] - zs_r[-1]) ** 2
    )
    print(f"  Euler 终点: ({xs_e[-1]:.4f}, {ys_e[-1]:.4f}, {zs_e[-1]:.4f})")
    print(f"  RK4   终点: ({xs_r[-1]:.4f}, {ys_r[-1]:.4f}, {zs_r[-1]:.4f})")
    print(f"  两者偏差: {diff:.4f} （Euler 已经明显偏离）")
    print()

    # 3. 蝴蝶效应
    print("[3] 蝴蝶效应验证")
    print("  初始条件: (1.0, 1.0, 1.0) vs (1.0+1e-10, 1.0, 1.0)")
    times, dists = butterfly_effect_test((1.0, 1.0, 1.0), perturbation=1e-10, dt=0.01, n_steps=5000)
    print(f"  t=5s  时偏差: {dists[500]:.6e}")
    print(f"  t=10s 时偏差: {dists[1000]:.6e}")
    print(f"  t=20s 时偏差: {dists[2000]:.6e}")
    print(f"  t=50s 时偏差: {dists[5000]:.6e}")
    print()

    # 4. Lyapunov 指数
    print("[4] 最大 Lyapunov 指数估算")
    lyap = estimate_lyapunov((1.0, 1.0, 1.0), dt=0.01, n_steps=15000)
    print(f"  估算值: λ ≈ {lyap:.4f}")
    print(f"  理论参考: λ ≈ 0.9056 (Sprott, 2010)")
    print(f"  判定: {'✓ 正指数 → 混沌系统' if lyap > 0.5 else '✗ 结果异常'}")
    print()

    # 5. 统计验证：从不同初始条件出发，长期分布的一致性
    print("[5] 长期统计一致性（混沌吸引子的遍历性）")
    samples = 20
    all_z_max = []
    all_z_min = []
    for _ in range(samples):
        s0 = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 50))
        _, _, zs_s = simulate(s0, dt=0.01, n_steps=6000, method='rk4')
        # 丢弃前 2000 步（瞬态）
        zs_ss = zs_s[2000:]
        all_z_max.append(max(zs_ss))
        all_z_min.append(min(zs_ss))
    print(f"  {samples} 组不同初始条件，z 轴极大值均值: {sum(all_z_max)/len(all_z_max):.2f} ± {math.sqrt(sum((x-sum(all_z_max)/len(all_z_max))**2 for x in all_z_max)/len(all_z_max)):.2f}")
    print(f"  {samples} 组不同初始条件，z 轴极小值均值: {sum(all_z_min)/len(all_z_min):.2f} ± {math.sqrt(sum((x-sum(all_z_min)/len(all_z_min))**2 for x in all_z_min)/len(all_z_min)):.2f}")
    print("  → 不同初始条件最终都落在同一吸引子上")


if __name__ == "__main__":
    demo()
