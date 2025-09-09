from utils import PARSER, init_log
from env import make_env

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, csv

def _load_series_from_csv_pd(csv_path):
    df = pd.read_csv(csv_path)
    tcol = [c for c in df.columns if "time" in c.lower()][0]
    vcol = [c for c in df.columns if "conf" in c.lower() or "value" in c.lower() or "sum" in c.lower()][0]
    return df[tcol].to_numpy(), df[vcol].to_numpy()

def eval_banchmark_at_time(config_args, root_path, strategies, K, out_path):
    '''
    Plot figures to compare performance of different methods at each time step
    - config_args:  configs of environment (unused here but kept for interface compatibility)
    - root_path:    the dir storing results of each method
    - strategies:   methods to compare, e.g. ['Single','RL','Random','Greedy','Detection']
    - K:            specific K to select files like "{strategy}_{K}.csv"
    - out_path:     directory to save the plot
    return:         saved plot path
    '''
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    for s in strategies:
        name = f"{s}.csv"
        data_file = os.path.join(root_path, name)
        t, y = _load_series_from_csv_pd(data_file)
        ax.plot(t, y, linewidth=1.8, label=f"{s} (K={K})")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Sum of confidence")
    ax.set_title(f"Benchmark over time (K={K})")
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def eval_banchmark_at_upbound(config_args, root_path, strategies, K, out_path):
    '''
    Plot figures to compare performance of different methods under variuos K limits
    - config_args: configs of environment
    - root_path: the dir storing results of each method
    - strategies: all the mothods need to be compared
        # Single, RL, Random, Greedy, Detection(todo)
    - K: K 
    return: None
    '''
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    for s in strategies:
        ks, means = [], []
        for k in K:
            csv_path = os.path.join(root_path, f"{s}_{k}.csv")
            t, y = _load_series_from_csv_pd(csv_path)
            ks.append(k)
            means.append(float(np.mean(y)))
        if ks:
            order = np.argsort(ks)
            ax.plot(np.array(ks)[order], np.array(means)[order], marker="o", linewidth=1.8, label=s)

    ax.set_xlabel("K")
    ax.set_ylabel("Mean sum of confidence")
    ax.set_title("Benchmark vs K")
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def eval_perf_at_thresholds(config_args, root_files, settings, K, out_path):
    '''
    Plot figures to show the performance of proposed model under difference K thresholds
    - config_args: configs of environment
    - root_files: the dir storing results of each method
    - settings: different thresholds
        # 0, 5, 10, 20, 50
    return: None
    '''
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    xs, ys = [], []
    for thr in settings:
        csv_path = root_files[thr]          # RL_K50_thr0.csv, RL_K50_thr5.csv
        t, y = _load_series_from_csv_pd(csv_path)
        xs.append(thr)
        ys.append(np.mean(y))        #  np.max/np.median/AUC

    ax.plot(xs, ys, marker="o", linewidth=1.8, label="Our Method")
    ax.set_xlabel("Threshold (±)")
    ax.set_ylabel("Mean sum of confidence")
    ax.set_title("Performance vs Threshold (fixed K)")
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path 

def eval_theoretical_diff(config_args, root_files, settings, out_path, show_errorbar=False):
    '''
    Plot figures to compare difference of optimal K between theory and practice under difference settings
    - config_args: configs of environment
    - root_files: the dir storing results of each method
    - settings: titles of different settings (including communication and computing models)
        # 
    return: None
    '''
    theory_perf, ours_perf = [], []
    theory_k, ours_k = [], []
    theory_perf_err, ours_perf_err = [], []
    theory_k_err, ours_k_err = [], []

    for setting in settings:
        paths = root_files[setting]

        # === theoritical value ===
        paths_theory = paths["theory"] if isinstance(paths["theory"], list) else [paths["theory"]]
        perf_list, k_list = [], []
        for p in paths_theory:
            df = pd.read_csv(p)
            idx = df["performance"].idxmax()
            perf_list.append(df.loc[idx, "performance"])
            k_list.append(df.loc[idx, "K"])
        theory_perf.append(np.mean(perf_list))
        theory_k.append(np.mean(k_list))
        theory_perf_err.append(np.std(perf_list))
        theory_k_err.append(np.std(k_list))

        # === our method ===
        paths_ours = paths["ours"] if isinstance(paths["ours"], list) else [paths["ours"]]
        perf_list, k_list = [], []
        for p in paths_ours:
            df = pd.read_csv(p)
            idx = df["performance"].idxmax()
            perf_list.append(df.loc[idx, "performance"])
            k_list.append(df.loc[idx, "K"])
        ours_perf.append(np.mean(perf_list))
        ours_k.append(np.mean(k_list))
        ours_perf_err.append(np.std(perf_list))
        ours_k_err.append(np.std(k_list))

    x = np.arange(len(settings))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10,5))

    # 左轴：性能对比 (bar + 可选误差棒)
    if show_errorbar:
        ax1.bar(x - width/2, theory_perf, width, yerr=theory_perf_err, capsize=4, label="Theory Perf")
        ax1.bar(x + width/2, ours_perf, width, yerr=ours_perf_err, capsize=4, label="Ours Perf")
    else:
        ax1.bar(x - width/2, theory_perf, width, label="Theory Perf")
        ax1.bar(x + width/2, ours_perf, width, label="Ours Perf")

    ax1.set_ylabel("Performance")
    ax1.set_xlabel("Environment Setting")
    ax1.set_xticks(x)
    ax1.set_xticklabels(settings)
    ax1.legend(loc="upper left")

    # 右轴：最优K对比 (line + 可选误差棒)
    ax2 = ax1.twinx()
    if show_errorbar:
        ax2.errorbar(x, theory_k, yerr=theory_k_err, fmt="-o", color="blue", capsize=4, label="Theory K*")
        ax2.errorbar(x, ours_k, yerr=ours_k_err, fmt="-s", color="orange", capsize=4, label="Ours K*")
    else:
        ax2.plot(x, theory_k, "-o", color="blue", label="Theory K*")
        ax2.plot(x, ours_k, "-s", color="orange", label="Ours K*")
    ax2.set_ylabel("Optimal K")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path

def eval_ablation(config_args, root_files, modules, settings, out_path):
    '''
    Plot figures to analyze impacts of difference modules in the proposed method
    - config_args: configs of environment
    - root_files: the dir storing results of each modules
    - settings: different K
    - modules: 
        # attention, gnn, cnn
    return: None
    '''
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    for m in modules:
        csv_path = root_files[m]
        t, y = _load_series_from_csv_pd(csv_path)
        ax.plot(t, y, linewidth=1.8, label=m)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Performance (sum_confidence)")
    ax.set_title("Ablation Study: Module Impact")
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def get_perf_strategy(config_args, evaluation_maps, max_k, strategy, out_path):
    '''
    Get result of the startegy
    - config_args: configs of environment
    - evaluation_maps: the dir storing results of each modules
    - max_k:  K
    - strategy: Single, RL, Random, Greedy, Detection
    return: None
    '''
    file = out_path
    if out_path is None:
        # file = f"{config_args.strategy}_{max_k}.csv"
        file = f"{config_args.strategy}.csv"
    os.makedirs(config_args.result_path, exist_ok=True)
    result_file = os.path.join(config_args.result_path, file)

    lengths = len(evaluation_maps)
    results = []
    for time_step in range(lengths):
        cur_confidence_value = 0
        for agent_i in range(config_args.num_vehicles):
            maps = None
            if strategy == 'Single':
                maps = evaluation_maps[time_step][0][agent_i+1]['local_map']
            else:
                maps = evaluation_maps[time_step][0][agent_i+1]['cur_fused_map']

            print(f"car {agent_i + 1} cur fused map sum 7: {np.sum(maps)}")
            cur_confidence_value += np.sum(maps)
        results.append(cur_confidence_value)
    
    # 将result写到文件中
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step", "sum_confidence"])
        for t, val in enumerate(results):
            writer.writerow([t, f"{val:.6f}"])


if __name__ == '__main__':
    root_demo = "/home/peh324/Codes/WorldModel/results/evaluation"
    os.makedirs(root_demo, exist_ok=True)

    rng = np.random.default_rng(7)
    T = 120  # 120 timesteps
    strategies = ["Single", "RL", "Random", "Greedy"]
    K = 20

    # Generate four series with different characteristics (random-walk-like)
    base = np.cumsum(rng.normal(loc=0.0, scale=1.0, size=T)) + K
    series = {
        "Single": base + np.cumsum(rng.normal(0, 0.5, size=T)) - 5,
        "RL": base + np.linspace(0, 25, T) + np.cumsum(rng.normal(0, 0.4, size=T)),  # best upward trend
        "Random": base + np.cumsum(rng.normal(0, 1.2, size=T)) - 10,
        "Greedy": base + np.linspace(0, 15, T) + np.cumsum(rng.normal(0, 0.6, size=T)),
    }

    # Ensure all values are positive-ish and write CSVs
    for s, y in series.items():
        y = np.maximum(y, 0.0)
        csv_path = os.path.join(root_demo, f"{s}_{K}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_step", "sum_confidence"])
            for t, v in enumerate(y):
                w.writerow([t, f"{float(v):.6f}"])

    # ---------- Run the plotting function ----------
    fig_dir = f"/home/peh324/Codes/WorldModel/results/evaluation/eval_benchmark_at_time_{K}"
    plot_path = eval_banchmark_at_time(None, root_demo, strategies, K, fig_dir)

    fig_dir = f"/home/peh324/Codes/WorldModel/results/evaluation/eval_benchmark_at_upbound"
    plot_path = eval_banchmark_at_upbound(None, root_demo, strategies, [20, 50, 100], fig_dir)

    fig_dir = f"/home/peh324/Codes/WorldModel/results/evaluation/eval_benchmark_at_upbound"
    plot_path = eval_banchmark_at_upbound(None, root_demo, strategies, [20, 50, 100], fig_dir)



    settings = ["LowUpload", "HighUpload", "FastCompute"]
    root_files = {}
    Ks = np.arange(10, 101, 10)

    for setting in settings:
        theory_paths, ours_paths = [], []
        for run in range(3):  # 3 runs per setting
            # Theory
            peak_theory = rng.integers(30, 80)
            perf_theory = -0.002*(Ks - peak_theory)**2 + rng.uniform(1.0, 1.4)
            df_theory = pd.DataFrame({"K": Ks, "performance": perf_theory})
            path_theory = os.path.join(root_demo, f"theory_{setting}_run{run}.csv")
            df_theory.to_csv(path_theory, index=False)
            theory_paths.append(path_theory)

            # Ours
            peak_ours = peak_theory + rng.integers(-10, 10)
            perf_ours = -0.002*(Ks - peak_ours)**2 + rng.uniform(0.9, 1.3)
            df_ours = pd.DataFrame({"K": Ks, "performance": perf_ours})
            path_ours = os.path.join(root_demo, f"ours_{setting}_run{run}.csv")
            df_ours.to_csv(path_ours, index=False)
            ours_paths.append(path_ours)

        root_files[setting] = {"theory": theory_paths, "ours": ours_paths}

    # ---- Run plotting function ----
    out_path = f"/home/peh324/Codes/WorldModel/results/evaluation/theory_vs_practice_demo.png"
    fig_path = eval_theoretical_diff(None, root_files, settings, out_path, show_errorbar=True)


    
    # Full model baseline
    full = np.cumsum(rng.normal(loc=0.2, scale=0.1, size=T)) + 50
    # Remove attention => drop some performance
    no_att = full - np.linspace(1, 5, T)
    # Remove gnn => bigger drop
    no_gnn = full - np.linspace(2, 8, T)
    # Remove cnn => moderate drop
    no_cnn = full - np.linspace(1.5, 6, T)

    series = {
        "Full model": full,
        "No Attention": no_att,
        "No GNN": no_gnn,
        "No CNN": no_cnn,
    }

    root_files = {}
    time = np.arange(T)
    for name, y in series.items():
        df = pd.DataFrame({"time_step": time, "sum_confidence": y})
        path = os.path.join(root_demo, f"{name.replace(' ','_')}.csv")
        df.to_csv(path, index=False)
        root_files[name] = path

    modules = list(series.keys())
    out_path = f"/home/peh324/Codes/WorldModel/results/evaluation/ablation_demo.png"
    fig_path = eval_ablation(None, root_files, modules, settings=[50], out_path=out_path)
