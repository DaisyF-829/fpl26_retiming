import os
import numpy as np
import torch
import pickle

def load_ss_npz_data(folder_path):
    ss_data_dict = {}

    for filename in os.listdir(folder_path):
        if filename.startswith("ss_") and filename.endswith(".npz"):
            ss_index = filename.split("_")[1].split(".")[0]
            file_path = os.path.join(folder_path, filename)
            npz_data = np.load(file_path)

            try:
                src_rr_indexes = torch.tensor(
                    npz_data['src_rr_indexes'].reshape(-1, 2).T, dtype=torch.int32)
                sink_rr_indexes = torch.tensor(
                    npz_data['sink_rr_indexes'].reshape(-1, 2).T, dtype=torch.int32)
                net_delay = torch.tensor(npz_data['net_delay'], dtype=torch.float32)
                tiles = torch.tensor(npz_data['tiles'], dtype=torch.int32)
                tile_edge_src = torch.tensor(npz_data['tile_edge_src'], dtype=torch.int32)
                tile_edge_dst = torch.tensor(npz_data['tile_edge_dst'], dtype=torch.int32)
                tile_edges = torch.stack([tile_edge_src, tile_edge_dst], dim=0)
                tedge_id = torch.tensor(npz_data['tedge_id'], dtype=torch.int32)
                tile_edge_feats = torch.tensor(npz_data['tile_edge_feats'], dtype=torch.int32)
                rr_tile_edges = torch.tensor(
                    npz_data['rr_tile_edges'].reshape(2, -1), dtype=torch.int32)
            except KeyError as e:
                print(f"Missing key {e} in {file_path}, skipping.")
                continue

            ss_data_dict[ss_index] = {
                'src_rr_indexes': src_rr_indexes,
                'sink_rr_indexes': sink_rr_indexes,
                'tedge_id': tedge_id,
                'net_delay': net_delay,
                'tiles': tiles,
                'tile_edges': tile_edges,
                'tile_edge_feats': tile_edge_feats,
                'rr_tile_edges': rr_tile_edges
            }

    return ss_data_dict


def save_to_pkl(data_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✅ Saved {len(data_dict)} subgraphs to {output_path}")


def process_all_ss_dirs(root_dir):
    for subdir, _, files in os.walk(root_dir):
        if any(fname.startswith("ss_") and fname.endswith(".npz") for fname in files):
            out_pkl = os.path.join(root_dir, "ss_graph_data.pkl")
            # if os.path.exists(out_pkl):
            #     print(f"⏩ Skipping {subdir}, ss_graph_data.pkl already exists.")
            #     continue

            print(f"📂 Processing {subdir}")
            ss_data = load_ss_npz_data(subdir)

            if ss_data:
                save_to_pkl(ss_data, out_pkl)

                # 打印前几个样例（最多3个）
                for ss_id, data in list(ss_data.items())[:3]:
                    print(f"\nSubgraph {ss_id}:")
                    for key, value in data.items():
                        print(f"  {key}: shape = {tuple(value.shape)}, dtype = {value.dtype}")
                    print(f"  net_delay sample: {data['net_delay'][:1].tolist()}")
            else:
                print(f"⚠️  No valid ss_*.npz files found in {subdir}.")


if __name__ == "__main__":
    root_path = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/attention_layer/"  # 修改为你自己的顶层路径
    process_all_ss_dirs(root_path)
