import os, json
import numpy as np
from math import radians, sin, cos

def build_episode_npz_from_jsonl(
    jsonl_path: str,
    out_npz: str,
    split_by_reset: bool = False,
    gap_threshold: int = 5,
):
    """
    将 records.jsonl 还原成 episode 级数据（单episode或多episode）。
    默认：当 timestep 不连续且断点 > gap_threshold 时，认为是 reset，切分 episode。
    如果你明确只采了一个episode，split_by_reset=False 即可。

    jsonl 每行至少包含：
      timestep, actor_id, group_id, x,y,z,vx,vy,vz,yaw, image_path
    """
    # ---- 1) 读 records ----
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) == 0:
        raise ValueError("jsonl is empty")

    # ---- 2) 按 timestep 排序 ----
    rows.sort(key=lambda r: int(r.get("timestep", 0)))

    # ---- 3) 可选：按 reset 切分多个 episode ----
    # 通过 timestep 断点识别（你也可以改成 frame 断点）
    episodes = []
    cur = [rows[0]]
    for i in range(1, len(rows)):
        prev_t = int(rows[i-1].get("timestep", 0))
        cur_t  = int(rows[i].get("timestep", 0))
        if split_by_reset and (cur_t - prev_t > gap_threshold):
            episodes.append(cur)
            cur = [rows[i]]
        else:
            cur.append(rows[i])
    episodes.append(cur)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    # ---- 4) 逐 episode 写 npz ----
    # 如果你只要一个episode，就写 out_npz；多个episode就 out_npz 替换成 out_ep000.npz 等
    def dump_one(ep_rows, out_path):
        # 收集该 episode 的所有 timestep
        timesteps = sorted(set(int(r["timestep"]) for r in ep_rows))
        T = len(timesteps)
        t2idx = {t:i for i,t in enumerate(timesteps)}

        # 收集该 episode 出现过的所有 actor_id（固定列顺序）
        actor_ids = sorted(set(int(r["actor_id"]) for r in ep_rows))
        N = len(actor_ids)
        a2idx = {a:i for i,a in enumerate(actor_ids)}

        # group_id：对每个 actor 取“最常见”的 group_id（或第一次出现的）
        group_id = np.full((N,), -1, dtype=np.int32)
        counts = {a:{} for a in actor_ids}
        for r in ep_rows:
            a = int(r["actor_id"])
            g = int(r.get("group_id", -1))
            counts[a][g] = counts[a].get(g, 0) + 1
        for a in actor_ids:
            best_g = max(counts[a].items(), key=lambda kv: kv[1])[0]
            group_id[a2idx[a]] = int(best_g)

        # 初始化数组
        image_paths = np.empty((T, N), dtype=object)
        image_paths[:] = ""

        states = np.zeros((T, N, 6), dtype=np.float32)
        node_mask = np.zeros((T, N), dtype=np.bool_)

        # 填充
        for r in ep_rows:
            t = int(r["timestep"]); ti = t2idx[t]
            a = int(r["actor_id"]); ai = a2idx[a]

            x = float(r.get("x", 0.0))
            y = float(r.get("y", 0.0))
            vx = float(r.get("vx", 0.0))
            vy = float(r.get("vy", 0.0))
            yaw = float(r.get("yaw", 0.0))
            sy = sin(radians(yaw))
            cy = cos(radians(yaw))

            states[ti, ai, :] = np.array([x, y, vx, vy, sy, cy], dtype=np.float32)
            node_mask[ti, ai] = True

            p = r.get("image_path", None)
            if isinstance(p, str):
                image_paths[ti, ai] = p

        np.savez_compressed(
            out_path,
            image_paths=image_paths,     # (T,N) object(str)
            states=states,               # (T,N,6)
            group_id=group_id,           # (N,)
            node_mask=node_mask,         # (T,N)
            timesteps=np.array(timesteps, dtype=np.int32),  # (T,)
            actor_ids=np.array(actor_ids, dtype=np.int32),  # (N,)
        )
        return {"T": T, "N": N, "out": out_path}

    infos = []
    if len(episodes) == 1:
        infos.append(dump_one(episodes[0], out_npz))
    else:
        base, ext = os.path.splitext(out_npz)
        for k, ep in enumerate(episodes):
            out_path = f"{base}_ep{k:03d}{ext}"
            infos.append(dump_one(ep, out_path))

    return infos


if __name__ == "__main__":

    jsonl_path = "/home/peh324/Codes/WorldModel/data/GODE/records.jsonl"   # 或 frames.jsonl
    out_npz = "/home/peh324/Codes/WorldModel/data/GODE/episode_paths.npz"
    infos = build_episode_npz_from_jsonl(
        jsonl_path=jsonl_path,
        out_npz=out_npz,
        split_by_reset=False,      # 如果你一个 run 有多个 reset，就 True
        gap_threshold=5
    )
    print(infos)
