import os
import torch
import dgl
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor


def load_tile_graph(fname, tile_folder):
    tile_idx = int(fname.split("_")[1].split(".")[0])
    path = os.path.join(tile_folder, fname)
    data = np.load(path)

    if 'global_features' not in data:
        print(f"  Skipped {fname}: missing global_features")
        return None

    global_feats = torch.tensor(data['global_features'], dtype=torch.float32)

    if 'rrnode_features' in data:
        node_feats = torch.tensor(data['rrnode_features'], dtype=torch.float32)
        assert node_feats.numel() % 13 == 0, f"node_feats size not divisible by 13 in {fname}"
        num_nodes = node_feats.numel() // 13
        node_feats = node_feats.view(num_nodes, 13)
    else:
        num_nodes = 1
        node_feats = torch.zeros((1, 13))

    if 'rredge_src' in data and 'rredge_dst' in data and 'rredge_feats' in data:
        src = torch.tensor(data['rredge_src'], dtype=torch.int64)
        dst = torch.tensor(data['rredge_dst'], dtype=torch.int64)
        edge_feats = torch.tensor(data['rredge_feats'], dtype=torch.float32)
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['feat'] = edge_feats.view(-1, 4)
    else:
        g = dgl.graph(([], []), num_nodes=num_nodes)

    g.ndata['feat'] = node_feats
    g.ndata['global'] = global_feats.expand(num_nodes, -1)

    return tile_idx, g


def load_tile_graphs_parallel(tile_folder):
    tile_graphs = {}
    with ThreadPoolExecutor() as executor:
        tile_files = sorted(
            [fname for fname in os.listdir(tile_folder) if fname.startswith("tile_") and fname.endswith(".npz")],
            key=lambda fname: int(fname.split("_")[1].split(".")[0])
        )
        results = executor.map(lambda fname: load_tile_graph(fname, tile_folder), tile_files)
        for result in results:
            if result is not None:
                tile_idx, g = result
                tile_graphs[tile_idx] = g
    return tile_graphs


def process_all_tiles_in_root(root_dir):
    """
    遍历 root_dir 下所有子目录，查找 tiles 文件夹并生成 tile_graphs.pkl 文件
    """
    for subdir, dirs, files in os.walk(root_dir):
        if "tiles" in dirs:
            tile_folder = os.path.join(subdir, "tiles")
            print(f"Processing: {tile_folder}")

            tile_graphs = load_tile_graphs_parallel(tile_folder)
            print(f"  -> Loaded {len(tile_graphs)} tile graphs")

            out_path = os.path.join(subdir, "tiles.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(tile_graphs, f)

            print(f"  -> Saved to: {out_path}\n")


if __name__ == "__main__":
    # 顶层路径，例如包含多个 circuit 子文件夹的路径
    root_dir = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/tpu_like.large.os/"
    process_all_tiles_in_root(root_dir)
