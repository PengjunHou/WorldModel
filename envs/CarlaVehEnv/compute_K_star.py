import math
import numpy as np


def delta_n(n: int) -> float:
    # 你的 Δ_n = (n-1)/[2(n+1)]
    return (n - 1) / (2.0 * (n + 1))

def build_rj_pj(N: int, cover_prob: float):
    """
    构造 r_j(降序) 与 p_j，对应 n=N..1 的映射。
    cover_prob = γ = Pr{单车覆盖该区域}
    """
    p_n = np.array([math.comb(N, n) * (cover_prob ** n) * ((1 - cover_prob) ** (N - n))
                    for n in range(N + 1)], dtype=float)  # n=0..N
    # 只取 n>=1，因为 n=0 时 Δ_0 无意义/为0；保留也行但不会进 Top-K
    ns = np.arange(1, N + 1)
    r_vals = np.array([delta_n(int(n)) for n in ns], dtype=float)  # Δ_1..Δ_N，递增
    p_vals = p_n[1:]  # 对应 n=1..N

    # 将 r 按值降序排列（即 n 从 N 到 1）
    order = np.argsort(-r_vals)
    r_sorted = r_vals[order]
    p_sorted = p_vals[order]
    return r_sorted, p_sorted  # 长度 N

def expected_topk_sum(U: int, K: int, r: np.ndarray, p: np.ndarray) -> float:
    """
    精确计算 E[R(K)] = sum_j r_j * E[X_j]
    其中 X_j = Top-K 中取到“值==r_j”的样本个数（期望）
    通过 (A, C) 的多项分布期望 min(C, max(0, K-A)) 来算。
    复杂度 O(N * U^2)，U<=几百时可用；更大可改近似/MC。
    """
    assert 0 <= K <= U
    N = len(r)
    # 累积得到 s_j, q_j
    s = np.zeros(N)
    q = np.zeros(N)
    cum = 0.0
    for j in range(N):
        s[j] = cum              # prob(> r_j)
        cum += p[j]
    for j in range(N):
        q[j] = 1.0 - s[j] - p[j]

    def multinom_coeff(u, a, c):
        # (U choose a) * (U-a choose c)
        return math.comb(u, a) * math.comb(u - a, c)

    E_sum = 0.0
    for j in range(N):
        sj, pj, qj = s[j], p[j], q[j]
        # 避免数值误差
        sj = float(max(0.0, min(1.0, sj)))
        pj = float(max(0.0, min(1.0, pj)))
        qj = float(max(0.0, min(1.0, qj)))

        # 计算 E[X_j]
        EX = 0.0
        for a in range(0, U + 1):           # a: > r_j 的数量
            # 若 a >= K，则等于 r_j 的都进不了 Top-K
            need = K - a
            if need <= 0:
                # 仍需把 (A=a) 的概率加起来，但 min(C,0)=0，贡献为 0
                # 只为数值稳定可略过 C 遍历
                continue
            # 对每个 c: 等于 r_j 的数量
            max_c = U - a
            # PMF(A=a, C=c) = Multinomial(U; a,c,U-a-c | s,p,q)
            # = C(U; a,c,U-a-c) * s^a * p^c * q^(U-a-c)
            # 贡献: min(c, need)
            for c in range(0, max_c + 1):
                mcoef = multinom_coeff(U, a, c)
                prob = mcoef * (sj ** a) * (pj ** c) * (qj ** (U - a - c))
                EX += min(c, need) * prob
        E_sum += r[j] * EX
    return E_sum

def latency_discount(K: int, s0_MB: float, s1_MB: float, Rul_Mbps: float, Rdl_Mbps: float, beta: float = None, c0: float = 0.0):
    """
    返回 e^{-tau(K)}。
    - 精确工程版：tau(K)=K*(8*s1/Rul + 8*s0/Rdl) + c0
    - 若你一定要用论文里 tau = beta*(s0+s1)*K + c0 的形态，则传入 beta 覆盖。
    """
    if beta is None:
        tau = K * (8.0 * s1_MB / max(1e-9, Rul_Mbps) + 8.0 * s0_MB / max(1e-9, Rdl_Mbps)) + c0
    else:
        tau = beta * (s0_MB + s1_MB) * K + c0
    return math.exp(-tau)

def feasible_K_max(T_s: float, Rul_Mbps: float, s1_MB: float):
    """
    T_s: period of frame
    Rul_Mbps: uplink rate
    s1_MB: size of each area
    帧内上行预算给出的 K 上界： floor( (R_UL * T) / (8 * s1) )
    """
    if s1_MB <= 0:
        return 0
    return int(max(0, math.floor((Rul_Mbps * T_s) / (8.0 * s1_MB))))

def find_K_star(N: int, U: int, cover_prob: float,
                s0_MB: float, s1_MB: float,
                Rul_Mbps: float, Rdl_Mbps: float,
                T_s: float,
                K_max_cap: int = None,
                beta: float = None):
    """
    综合“拓扑收益 + 时延折扣 + 帧预算”，枚举 K 得到 K*。
    - 若 K_max_cap 给出，则 K_max = min(预算上界, K_max_cap)
    - 若 beta 给出，则使用 tau = beta*(s0+s1)*K；否则用工程速率表达。
    返回: K_star, table(dict: K -> objective, ERK, discount, Kmax)
    """
    # 1) r_j, p_j
    r, p = build_rj_pj(N, cover_prob)

    # 2) 帧级上行预算约束 -> K_max
    K_budget = feasible_K_max(T_s, Rul_Mbps, s1_MB)
    K_max = K_budget if K_max_cap is None else min(K_budget, K_max_cap)

    # 3) 枚举 K
    obj = {}
    best_K, best_val = 0, -1.0
    for K in range(0, K_max + 1):
        ERK = expected_topk_sum(U, K, r, p)     # 期望 Top-K 和（未折扣）
        disc = latency_discount(K, s0_MB, s1_MB, Rul_Mbps, Rdl_Mbps, beta=beta, c0=0.0)
        val = ERK * disc
        obj[K] = {"ERK": ERK, "disc": disc, "value": val}
        if val > best_val:
            best_val, best_K = val, K

    return best_K, {"Kmax_budget": K_budget, "Kmax_used": K_max, "table": obj}


def expected_topk_sum_mc(U: int, K: int, r: np.ndarray, p: np.ndarray,
                         trials: int = 5000, rng: np.random.Generator | None = None,
                         batch: int = 2000) -> float:
    """
    Monte Carlo 近似：E[R(K)] = E[Top-K sum]，适合 U 很大 (>=1e3).
    思路：
      - 对每次试验，先采样 (C1..CN) ~ Multinomial(U; p) 作为各“值档”的计数，
        然后从最高值档开始“装满” Top-K：sum_j r_j * min(C_j, remain)。
      - 每次试验 O(N)，总复杂度 O(trials * N)。与 U 无关。
    参数：
      - trials: 试验次数，建议 2e3~2e4；越大越准。
      - batch:  分批调用 multinomial，避免一次性 trials 太大占内存。
      - r, p:   已按值降序（与你的 build_rj_pj 保持一致）。
    """
    # assert 0 <= K <= U
    if K == 0 or U == 0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng()

    N = len(r)
    total = 0.0
    remain_trials = trials
    while remain_trials > 0:
        b = min(batch, remain_trials)
        # (b, N) 的计数矩阵，每行一组 Multinomial 采样结果
        counts = rng.multinomial(U, p, size=b)  # int matrix
        # 累加每个试验的 Top-K 和
        # 向量化按值档从高到低装满 K
        # 这里用逐行处理（N 很小，逐行循环即可）
        for row in counts:
            need = K
            s = 0.0
            # r 已降序；遍历值档
            for j in range(N):
                if need <= 0:
                    break
                take = row[j] if row[j] <= need else need
                if take > 0:
                    s += r[j] * take
                    need -= take
            total += s
        remain_trials -= b

    return total / float(trials)

def find_K_star_mc(N: int, U: int, cover_prob: float,
                   s0_MB: float, s1_MB: float,
                   Rul_Mbps: float, Rdl_Mbps: float,
                   T_s: float,
                   K_max_cap: int = None,
                   beta: float = None,
                   trials: int = 5000,
                   latecy_fn = None,
                   rng: np.random.Generator | None = None):
    """
    return: K_star, {"Kmax_budget", "Kmax_used", "table": {K: {"ERK","disc","value"}}}
    """
    # 1) 值-概率表（降序）
    r, p = build_rj_pj(N, cover_prob)

    # 2) 帧内上行预算上界
    K_budget = feasible_K_max(T_s, Rul_Mbps, s1_MB)
    K_max = K_budget if K_max_cap is None else min(K_budget, K_max_cap)

    # 3) 枚举 K
    table = {}
    best_K, best_val = 0, -1.0
    for K in range(0, K_max + 1):
        ERK = expected_topk_sum_mc(U, K, r, p, trials=trials, rng=rng)
        if not latecy_fn:
            disc = latency_discount(K, s0_MB, s1_MB, Rul_Mbps, Rdl_Mbps, beta=beta, c0=0.0)     ## TODO: 
        else:
            disc = latecy_fn(U, K)
        
        val = ERK * disc
        # print(f"ERK {ERK}, disc {disc}, value {val}")
        table[K] = {"ERK": ERK, "disc": disc, "value": val}
        if val > best_val:
            best_val, best_K = val, K

    return best_K, {"Kmax_budget": K_budget, "Kmax_used": K_max, "table": table}