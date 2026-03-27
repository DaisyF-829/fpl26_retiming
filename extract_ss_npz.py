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
            print(npz_data.files)

            # 提取并 reshape 数据，并转为 torch.Tensor

            src_rr_indexes = torch.tensor(
                npz_data['src_rr_indexes'].reshape(-1, 2).T, dtype=torch.int32)

            sink_rr_indexes = torch.tensor(
                npz_data['sink_rr_indexes'].reshape(-1, 2).T, dtype=torch.int32)
            
            net_delay = torch.tensor(npz_data['net_delay'], dtype=torch.float32)
            tedge_id = torch.tensor(npz_data['tedge_id'], dtype=torch.int32)
            tiles = torch.tensor(npz_data['tiles'], dtype=torch.int32)

            tile_edge_src = torch.tensor(npz_data['tile_edge_src'], dtype=torch.int32)
            tile_edge_dst = torch.tensor(npz_data['tile_edge_dst'], dtype=torch.int32)
            tile_edges = torch.stack([tile_edge_src, tile_edge_dst], dim=0)

            tile_edge_feats = torch.tensor(npz_data['tile_edge_feats'], dtype=torch.int32)

            rr_tile_edges = torch.tensor(
                npz_data['rr_tile_edges'].reshape(2, -1), dtype=torch.int32)

            # print(f"rr_tile_edges: {rr_tile_edges}")

            ss_data_dict[ss_index] = {
                'src_rr_indexes': src_rr_indexes,
                'sink_rr_indexes': sink_rr_indexes,
                'tedge_id': tedge_id,
                'net_delay': net_delay,
                'tiles': tiles,
                'tile_edges': tile_edges,  # shape (2, N)
                'tile_edge_feats': tile_edge_feats,
                'rr_tile_edges': rr_tile_edges  # shape (2, N)
            }

    return ss_data_dict

def save_to_pkl(data_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved {len(data_dict)} subgraphs to {output_path}")


def process_one_ss_folder(folder):
    output_pkl = os.path.join(os.path.dirname(folder), 'ss_graph_data.pkl')

    ss_graph_data = load_ss_npz_data(folder)
    save_to_pkl(ss_graph_data, output_pkl)

    print(f"\n=== 处理完成：{output_pkl} ===")
    print("=== 所有 tedge_id ===")
    all_tedge_ids = []
    for ss_id, data in ss_graph_data.items():
        tedge_id = data['tedge_id'].item() if hasattr(data['tedge_id'], 'item') else data['tedge_id']
        all_tedge_ids.append(tedge_id)
        print(f"Subgraph {ss_id}: tedge_id = {tedge_id}")

    for ss_id, data in list(ss_graph_data.items())[:3]:
        print(f"\nSubgraph {ss_id}:")
        for key, value in data.items():
            print(f"  {key}: shape = {tuple(value.shape)}, dtype = {value.dtype}")
        print(f"  tedge_id sample: {data['tedge_id'][:1].tolist()}")
        print(f"  net_delay sample: {data['net_delay'][:1].tolist()}")


# 示例调用
if __name__ == "__main__":
    # root_dir = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/lenet/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/"
    root_dir = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/attention_layer/"

    matched_folders = []

    for subdir, dirs, files in os.walk(root_dir):
        if os.path.basename(subdir) == 'ss_pairs':
            matched_folders.append(subdir)

    print(f"共找到 {len(matched_folders)} 个 ss_pairs 文件夹")

    for folder in matched_folders:
        process_one_ss_folder(folder)


