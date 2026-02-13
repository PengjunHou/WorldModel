import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from math import sin, cos, radians

def _get_ts(rec):
    # 兼容 time_step / timestep / frame
    for k in ["time_step", "timestep", "frame", "ts"]:
        if k in rec:
            return int(rec[k])
    return 0

def _as_int(x):
    try:
        return int(x)
    except Exception:
        return int(str(x))

def load_logs(jsonl_path):
    """
    支持两种日志：
    A) frame-per-line: {"time_step":t, "obs_states":{aid:{...}}, "image_paths":{aid:path}, ...}
    B) record-per-line: {"time_step":t, "actor_id":aid, "group_id":g, "x":..,"y":..,"vx":..,"vy":..,"yaw":.., "image_path":..}
    返回：
      frames[t][aid] = state_dict
      img_paths[t][aid] = image_path (可为空)
      group_map[aid] = group_id (若日志提供)
    """
    frames = defaultdict(dict)     # t -> aid -> st
    img_paths = defaultdict(dict)  # t -> aid -> path
    group_map = {}                # aid -> group_id

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # A) frame-per-line
            if "obs_states" in rec and isinstance(rec["obs_states"], dict):
                t = _get_ts(rec)
                for aid_str, st in rec["obs_states"].items():
                    aid = _as_int(aid_str)
                    frames[t][aid] = st
                # image_paths 可选
                if "image_paths" in rec and isinstance(rec["image_paths"], dict):
                    for aid_str, p in rec["image_paths"].items():
                        aid = _as_int(aid_str)
                        if p is not None:
                            img_paths[t][aid] = str(p)
                continue

            # B) record-per-line
            t = _get_ts(rec)
            aid = _as_int(rec.get("actor_id", rec.get("id", 0)))
            frames[t][aid] = rec
            if "group_id" in rec:
                group_map[aid] = int(rec["group_id"])
            if "image_path" in rec:
                img_paths[t][aid] = str(rec["image_path"])

    return frames, img_paths, group_map

def extract_state(st):
    """
    st 必须包含 x,y,vx,vy,yaw（yaw缺失则视为0）
    产出 [x,y,vx,vy,sin(yaw),cos(yaw)]，yaw按“度”处理（CARLA rotation.yaw 通常是度）
    """
    x = float(st["x"]); y = float(st["y"])
    vx = float(st["vx"]); vy = float(st["vy"])
    yaw_deg = float(st.get("yaw", 0.0))
    yaw_rad = radians(yaw_deg)
    return np.array([x, y, vx, vy, sin(yaw_rad), cos(yaw_rad)], dtype=np.float32)

def build_npz_paths(
    jsonl_path: str,
    out_npz_path: str,
    L: int = 6,
    H: int = 20,
    actor_ids=None,
    strict: bool = False,
    image_path_fallback=None,
    image_path_root = "/home/peh324/Codes/WorldModel/"
):
    """
    strict=False: 允许某些aid在某些t缺失，用 node_mask 标0，并用0填充状态
    image_path_fallback: callable(t, aid)->path，用于日志里没存 image_path 时按规则拼
    """
    frames, img_paths, group_map = load_logs(jsonl_path)
    all_ts = sorted(frames.keys())
    if len(all_ts) < L + H:
        raise ValueError(f"Not enough steps: {len(all_ts)} < L+H={L+H}")

    # 选定 actor_ids
    if actor_ids is None:
        # 默认用“出现频率最高的一批”作为固定N（更鲁棒）
        freq = defaultdict(int)
        for t in all_ts:
            for aid in frames[t].keys():
                freq[aid] += 1
        # 取全程都出现的优先
        max_freq = max(freq.values()) if freq else 0
        common = [aid for aid, c in freq.items() if c == max_freq]
        actor_ids = sorted(common) if common else sorted(freq.keys())
    else:
        actor_ids = sorted([_as_int(x) for x in actor_ids])

    N = len(actor_ids)
    D = 6

    # group_id: 没有就填 -1
    gid = np.array([group_map.get(aid, -1) for aid in actor_ids], dtype=np.int32)

    X_paths_past = []
    X_paths_future = [] 
    X_state = []
    Y_state = []
    Masks = []

    # 估计字符串长度上限（存unicode数组需要定长）
    # 先扫一遍已有路径
    max_len = 128
    for t in all_ts:
        for aid, p in img_paths[t].items():
            max_len = max(max_len, len(str(p)))
    max_len = min(max_len, 512)  # 防止太夸张

    for idx in range(0, len(all_ts) - (L + H) + 1):
        t_hist = all_ts[idx: idx + L]
        t_fut  = all_ts[idx + L: idx + L + H]

        paths_past = np.full((L, N), "", dtype=f"<U{max_len}")
        paths_future = np.full((H, N), "", dtype=f"<U{max_len}")
        x_past = np.zeros((L, N, D), dtype=np.float32)
        y_fut  = np.zeros((H, N, D), dtype=np.float32)
        mask   = np.ones((N,), dtype=np.uint8)

        # history
        for li, t in enumerate(t_hist):
            for ni, aid in enumerate(actor_ids):
                if aid not in frames[t]:
                    if strict:
                        raise ValueError(f"Missing actor {aid} at t={t}")
                    mask[ni] = 0
                    continue
                x_past[li, ni] = extract_state(frames[t][aid])

                # image path
                p = img_paths[t].get(aid, "")
                if (not p) and (image_path_fallback is not None):
                    p = image_path_fallback(t, aid)
                paths_past[li, ni] = image_path_root+str(p) if p else ""
                
        # future labels (+ future image paths)
        for hi, t in enumerate(t_fut):
            for ni, aid in enumerate(actor_ids):
                if aid not in frames[t]:
                    if strict:
                        raise ValueError(f"Missing actor {aid} at t={t}")
                    mask[ni] = 0
                    continue

                # state
                y_fut[hi, ni] = extract_state(frames[t][aid])

                # image path ✅
                p = img_paths[t].get(aid, "")
                if (not p) and (image_path_fallback is not None):
                    p = image_path_fallback(t, aid)
                paths_future[hi, ni] = image_path_root + str(p) if p else ""


        X_paths_past.append(paths_past)
        X_paths_future.append(paths_future)
        X_state.append(x_past)
        Y_state.append(y_fut)
        Masks.append(mask)

    image_paths_past   = np.stack(X_paths_past, axis=0)     # (B,L,N)
    image_paths_future = np.stack(X_paths_future, axis=0)   # (B,H,N)
    states_past      = np.stack(X_state, axis=0)     # (B,L,N,6)
    states_future    = np.stack(Y_state, axis=0)     # (B,H,N,6)
    node_mask        = np.stack(Masks, axis=0)       # (B,N)

    B = states_past.shape[0]
    group_id = np.tile(gid[None, :], (B, 1))

    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz_path,
        image_paths_past=image_paths_past,
        image_paths_future=image_paths_future,
        states_past=states_past,
        states_future=states_future,
        group_id=group_id.astype(np.int32),
        node_mask=node_mask.astype(np.uint8),
        actor_id=np.array(actor_ids, dtype=np.int32),
        L=np.array([L], dtype=np.int32),
        H=np.array([H], dtype=np.int32),
        D=np.array([D], dtype=np.int32),
    )

    print(f"Saved: {out_npz_path}")
    print(f"image_paths_past: {image_paths_past.shape} dtype={image_paths_past.dtype}")
    print(f"image_paths_future: {image_paths_future.shape} dtype={image_paths_future.dtype}")
    print(f"states_past:      {states_past.shape}")
    print(f"states_future:    {states_future.shape}")
    print(f"group_id:         {group_id.shape}")
    print(f"node_mask:        {node_mask.shape}  valid_ratio={node_mask.mean():.3f}")

if __name__ == "__main__":
    # 改成你的实际日志路径
    jsonl_path = "/home/peh324/Codes/WorldModel/data/GODE/records.jsonl"   # 或 frames.jsonl
    out_npz = "/home/peh324/Codes/WorldModel/data/GODE/dataset_paths.npz"

    build_npz_paths(
        jsonl_path=jsonl_path,
        out_npz_path=out_npz,
        L=20,
        H=10,
        actor_ids=None,     # 你也可以传 obs_vehicles 列表来固定 N 顺序
        strict=False,
        image_path_fallback=None,  # 如果你的日志里没存 image_path，可在这里写拼接规则
    )
