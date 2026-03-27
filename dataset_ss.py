import pickle
import torch
import dgl
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_embeddings_from_ss(ss_data, tile_embeddings_dict, num_tiles):
    """
    从 ss_data 提取源节点和汇节点的平均嵌入（含位置处理）和几何距离。
    返回：
        src_embedding_avg, sink_embedding_avg
        distance: 欧几里得距离
        src_xy, sink_xy: 位置坐标
    """
    src_embeddings = []
    sink_embeddings = []

    for i in range(ss_data['src_rr_indexes'].shape[1]):
        src_tile_idx, src_rr_idx = ss_data['src_rr_indexes'][:, i]
        embed = tile_embeddings_dict[int(src_tile_idx.item())]['node_embed'][int(src_rr_idx.item())].clone()
        src_embeddings.append(embed)

    for i in range(ss_data['sink_rr_indexes'].shape[1]):
        sink_tile_idx, sink_rr_idx = ss_data['sink_rr_indexes'][:, i]
        embed = tile_embeddings_dict[int(sink_tile_idx.item())]['node_embed'][int(sink_rr_idx.item())].clone()
        sink_embeddings.append(embed)

    # === 求平均嵌入 ===
    if src_embeddings:
        src_embedding_avg = torch.mean(torch.stack(src_embeddings), dim=0)
    else:
        src_embedding_avg = torch.zeros_like(embed)

    if sink_embeddings:
        sink_embedding_avg = torch.mean(torch.stack(sink_embeddings), dim=0)
    else:
        sink_embedding_avg = torch.zeros_like(embed)

    # === 从 avg 中提取空间位置信息 ===
    # 注意：此处已在 avg 后处理
    src_embedding_avg = src_embedding_avg.clone()
    sink_embedding_avg = sink_embedding_avg.clone()

    src_embedding_avg[-3] /= 100.0
    sink_embedding_avg[-3] /= 100.0

    src_xy = src_embedding_avg[-2:].detach()
    sink_xy = sink_embedding_avg[-2:].detach()

    distance = torch.norm(src_xy - sink_xy, p=2).item()

    src_feat = torch.cat([src_embedding_avg, torch.tensor([distance, num_tiles], dtype=torch.float32)])
    sink_feat = torch.cat([sink_embedding_avg, torch.tensor([distance, num_tiles], dtype=torch.float32)])

    return src_feat, sink_feat, distance



def build_hetero_graph(ss_data, tile_embeddings_dict):

    num_tiles = ss_data['tiles'].shape[0]
    
    num_nodes_dict = {
        'tile': num_tiles,
        'rrnode': 2
    }

    src_embedding, sink_embedding, distance = extract_embeddings_from_ss(ss_data, tile_embeddings_dict, num_tiles)
    tile_edge_src, tile_edge_dst = ss_data['tile_edges']

    rr_tile_edges = ss_data['rr_tile_edges']
    rr_ids = rr_tile_edges[:, 0].tolist()
    tile_ids = rr_tile_edges[:, 1].tolist()

    rr_to_tile_src, rr_to_tile_dst = [], []
    tile_to_rr_src, tile_to_rr_dst = [], []

    for rr_id, tile_id in zip(rr_ids, tile_ids):
        if rr_id == 0:
            rr_to_tile_src.append(0)
            rr_to_tile_dst.append(tile_id)
        elif rr_id == 1:
            tile_to_rr_src.append(tile_id)
            tile_to_rr_dst.append(1)
        else:
            print("⚠️ rr_tile_edges 内容:", ss_data['rr_tile_edges'])
            raise ValueError(f'Invalid rr_id: {rr_id}')
    

    data_dict = {
        ('tile', 'tile_to_tile', 'tile'): (tile_edge_src.to(torch.int64), tile_edge_dst.to(torch.int64)),
        ('rrnode', 'to_tile', 'tile'): (torch.tensor(rr_to_tile_src, dtype=torch.int64), torch.tensor(rr_to_tile_dst, dtype=torch.int64)),
        ('tile', 'to_rrnode', 'rrnode'): (torch.tensor(tile_to_rr_src, dtype=torch.int64), torch.tensor(tile_to_rr_dst, dtype=torch.int64)),
    }

    graph = dgl.heterograph(data_dict, num_nodes_dict)

    # 假设 ss_data['tiles'] 是一个 tensor，包含多个 tile 的索引
    tile_ids = ss_data['tiles'].tolist()  # 转换为 Python 列表

    # 获取每个 tile 的嵌入
    tile_graph_embeds = []
    for tid in tile_ids:
        tid = int(tid)  # 确保每个 tid 是整数
        tile_graph_embeds.append(tile_embeddings_dict[tid]['graph_embed'])


    distance_tensor = torch.full((num_tiles, 1), distance, dtype=torch.float32)
    num_tiles_tensor = torch.full((num_tiles, 1), float(num_tiles), dtype=torch.float32)

    tile_graph_embeds_tensor = torch.stack(tile_graph_embeds, dim=0)

    # 拼接额外特征
    tile_graph_embeds_tensor = tile_graph_embeds_tensor.squeeze(1)

    augmented_tile_feat = torch.cat([tile_graph_embeds_tensor, distance_tensor, num_tiles_tensor], dim=1)  # [N, D+2]

    # 设置 tile 节点特征
    graph.nodes['tile'].data['feat'] = augmented_tile_feat

    graph.nodes['rrnode'].data['feat'] = torch.stack([src_embedding, sink_embedding])

    label = ss_data['net_delay']

    tile_mean_emb = torch.mean(graph.nodes['tile'].data['feat'], dim=0)

    return graph, label, src_embedding, sink_embedding, tile_mean_emb
 


# def build_hetero_graph(ss_data, tile_global_embedding, tile_nodes_embedding):
#     src_embedding, sink_embedding = extract_embeddings_from_ss(ss_data, tile_nodes_embedding)
#     tile_edge_src, tile_edge_dst = ss_data['tile_edges']
#     rr_ids, tile_ids = ss_data['rr_tile_edges']

#     rr_to_tile_src, rr_to_tile_dst = [], []
#     tile_to_rr_src, tile_to_rr_dst = [], []

#     for rr_id, tile_id in zip(rr_ids, tile_ids):
#         if rr_id == 0:
#             rr_to_tile_src.append(0)
#             rr_to_tile_dst.append(tile_id)
#         elif rr_id == 1:
#             tile_to_rr_src.append(tile_id)
#             tile_to_rr_dst.append(1)

#     data_dict = {
#         ('tile', 'tile_to_tile', 'tile'): (tile_edge_src.to(torch.int64), tile_edge_dst.to(torch.int64)),
#         ('rrnode', 'to_tile', 'tile'): (torch.tensor(rr_to_tile_src, dtype=torch.int64), torch.tensor(rr_to_tile_dst, dtype=torch.int64)),
#         ('tile', 'to_rrnode', 'rrnode'): (torch.tensor(tile_to_rr_src, dtype=torch.int64), torch.tensor(tile_to_rr_dst, dtype=torch.int64)),
#     }

#     num_tiles = ss_data['tiles'].shape[0]
#     num_nodes_dict = {
#         'tile': num_tiles,
#         'rrnode': 2
#     }

#     graph = dgl.heterograph(data_dict, num_nodes_dict)

#     graph.nodes['tile'].data['feat'] = tile_global_embedding[ss_data['tiles']]
#     graph.nodes['rrnode'].data['feat'] = torch.stack([src_embedding, sink_embedding])
#     label = ss_data['net_delay']

#     return graph, label, src_embedding, sink_embedding




if __name__ == '__main__':

    # === 第一步：读取tile embeddings ===
    with open('/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/robot_rl/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/tile_embeddings.pkl', 'rb') as f:
        tile_embeddings_dict = pickle.load(f)


    # === 第二步：读取ss数据 ===
    with open('/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/robot_rl/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/ss_pairs/ss_graph_data.pkl', 'rb') as f:
        ss_data_dict = pickle.load(f)

    # 用于存储异构图和标签
    ss_pair_list = []

    # 用于记录每个样本的信息，输出为 CSV
    csv_records = []


    # === 第三步：构建所有异构图并记录 ===
    for ss_key in tqdm(ss_data_dict.keys()):
        ss_data = ss_data_dict[ss_key]
        tile_ids = ss_data['tiles']

        # 获取所有tile嵌入
        tile_node_embeds = []
        tile_graph_embeds = []

        for tid in tile_ids:
            tile_info = tile_embeddings_dict[tid]
            tile_node_embeds.append(tile_info['node_embed'])      # list of [N_i, D]
            tile_graph_embeds.append(tile_info['graph_embed'])    # list of [1, D]

        # 拼接node_embed：List[tensor(N_i,D)] → Tensor[total_nodes, D]
        tile_nodes_embedding = torch.cat(tile_node_embeds, dim=0)

        # 拼接graph_embed：List[tensor(1,D)] → Tensor[num_tiles, D]
        tile_graph_embeds = torch.cat(tile_graph_embeds, dim=0)

        # 构建图
        graph, label, src_emb, sink_emb = build_hetero_graph(ss_data, tile_graph_embeds, tile_nodes_embedding)

        ss_pair_list.append((graph, label))

        # 记录CSV信息
        record = {
            'net_delay': float(label),
            'tiles_count': len(tile_ids)
        }

        # 添加 src_embedding
        for i, v in enumerate(src_emb.tolist()):
            record[f'src_emb_{i}'] = v

        # 添加 sink_embedding
        for i, v in enumerate(sink_emb.tolist()):
            record[f'sink_emb_{i}'] = v

        # 添加 tile graph embedding 的均值
        tile_mean = tile_graph_embeds.mean(dim=0)
        for i, v in enumerate(tile_mean.tolist()):
            record[f'tile_mean_emb_{i}'] = v

        csv_records.append(record)

    # === 第四步：保存为pkl和csv ===
    with open('/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/robot_rl/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/ss_pair_hg.pkl', 'wb') as f:
        pickle.dump(ss_pair_list, f)

    df = pd.DataFrame(csv_records)
    df.to_csv('/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/robot_rl/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/ss_pair_summary.csv', index=False)

    print(f"保存完成，共处理 {len(ss_pair_list)} 个样本")
