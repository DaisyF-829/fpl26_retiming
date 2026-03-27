import argparse
import pickle
import os
import torch
import pandas as pd
from dataset_ss import build_hetero_graph  # 你已有的函数

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_index', type=int, required=True)
args = parser.parse_args()
workdir = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/robot_rl/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing"

# 加载批次数据
batch_file = f'ss_data_batch_{args.batch_index:03d}.pkl'
with open(f'{workdir}/ss_batch_new/{batch_file}', 'rb') as f:
    ss_data_dict = pickle.load(f)

# 加载 tile embeddings
with open(f'{workdir}/tile_embeddings.pkl', 'rb') as f:
    tile_embeddings_dict = pickle.load(f)

# 处理数据
ss_pair_list = []
csv_records = []

for ss_key in ss_data_dict.keys():

    ss_data = ss_data_dict[ss_key]

    num_tiles = ss_data['tiles'].shape[0]

    tedge_id = ss_data['tedge_id'].item()

    graph, label, src_emb, sink_emb, tile_mean_emb = build_hetero_graph(ss_data, tile_embeddings_dict)

    record = {
        'net_delay': float(label),
        'tedge_id': tedge_id,
        'tiles_count': num_tiles,
        **{f'src_emb_{i}': v for i, v in enumerate(src_emb.tolist())},
        **{f'sink_emb_{i}': v for i, v in enumerate(sink_emb.tolist())},
        **{f'tile_mean_emb_{i}': v for i, v in enumerate(tile_mean_emb.tolist())}
    }

    ss_pair_list.append((graph, torch.tensor([label], dtype=torch.float32)))
    csv_records.append(record)

# 保存每个批次结果
pkl_path = f'{workdir}/ss_batch_datasets/ss_pair_batch_{args.batch_index:03d}.pkl'
csv_path = f'{workdir}/ss_batch_datasets/ss_summary_batch_{args.batch_index:03d}.csv'

# with open(pkl_path, 'wb') as f:
#     pickle.dump(ss_pair_list, f)

pd.DataFrame(csv_records).to_csv(csv_path, index=False)

print(f"✅ 批次 {args.batch_index} 保存完成：{len(ss_pair_list)} 个样本")
