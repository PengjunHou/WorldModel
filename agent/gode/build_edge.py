import torch

def build_edges_knn_same_group(xy: torch.Tensor, group_id: torch.Tensor, mask: torch.Tensor, k: int = 2):
    """
    xy: (N,2)
    group_id: (N,)
    mask: (N,) bool
    return:
      edge_index: (2,E) long, edges are directed (i -> nn)
      edge_attr: (E,4) float, [dx,dy,dvx,dvy] 这里先占位，后面由外部传入
    """
    N = xy.size(0)
    edges_src = []
    edges_dst = []

    valid_idx = torch.where(mask)[0]
    if valid_idx.numel() <= 1:
        return torch.empty(2, 0, dtype=torch.long, device=xy.device)

    for g in torch.unique(group_id[mask]):
        nodes = torch.where(mask & (group_id == g))[0]
        m = nodes.numel()
        if m <= 1:
            continue
        pts = xy[nodes]  # (m,2)
        dist = torch.cdist(pts, pts)  # (m,m)
        dist.fill_diagonal_(1e9)
        kk = min(k, m - 1)
        nn = torch.topk(dist, k=kk, largest=False).indices  # (m,kk)

        for i in range(m):
            src = nodes[i].item()
            for j in nn[i]:
                dst = nodes[j].item()
                edges_src.append(src)
                edges_dst.append(dst)

    if len(edges_src) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=xy.device)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=xy.device)
    return edge_index
