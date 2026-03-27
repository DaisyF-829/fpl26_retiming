import os
import json
import dgl
import argparse
import pickle
from copy import deepcopy

from model_autoencoder import build_graph_autoencoder
from dataset_tile import load_dataset_tile_batch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch
import math
from os.path import join
from model_encoder import GCNEncoder
from dataset_tile import load_dataset_tile


# 加载图并编码
def encode_all_tiles(tile_encoder, tile_graphs, device):
    """
    tile_graphs: [DGLGraph]  # 直接使用列表
    返回：
      - tile_embeddings_dict: dict, key: tile_id, value: (node_embed, graph_embed)
    """
    tile_encoder.eval()  # 设置为eval模式，不进行训练
    tile_ids = range(len(tile_graphs))  # 用索引作为键
    graphs = [tile_graphs[tid].to(device) for tid in tile_ids]
    tile_embeddings_dict = {}

    for tid, graph in zip(tile_ids, graphs):
        node_embed, graph_embed = tile_encoder(graph)

        global_feat = graph.ndata['global'][0]
        node_global_feat = graph.ndata['global']

        # print('global_feat', global_feat.shape)
        # print('graph_embed', graph_embed.shape)

        global_feat = global_feat.unsqueeze(0)

        graph_emb = torch.cat([graph_embed, global_feat], dim=-1)

        node_embed = torch.cat([node_embed, node_global_feat], dim=-1)

        # 将每个图的节点嵌入和图嵌入存储
        tile_embeddings_dict[tid] = {
            'node_embed': node_embed,  # 每个图的节点嵌入
            'graph_embed': graph_emb  # 每个图的图嵌入
        }

    return tile_embeddings_dict
    # # 将所有图批量化
    # batched_graph = dgl.batch(graphs)
    # node_embed, graph_embed = tile_encoder(batched_graph)

    # # 拆分 node embedding
    # node_counts = [g.num_nodes() for g in graphs]
    # split_node_embeds = torch.split(node_embed, node_counts, dim=0)

    # print(f"node_embed: {node_embed.shape}, graph_embed: {graph_embed.shape}")
    # # 拆分 graph embedding
    # graph_counts = [32] * len(graphs)  # 每个图一个图嵌入
    # split_graph_embeds = torch.split(graph_embed, graph_counts, dim=0)

    # tile_embeddings_dict = {}
    # for tid, (node_emb, graph_emb) in zip(tile_ids, zip(split_node_embeds, split_graph_embeds)):
    #     # 获取节点的全局特征并拼接到图嵌入
    #     global_feat = graphs[tid].ndata['global']  # 获取该图的全局特征（假设是节点级的）
    #     graph_emb = torch.cat([graph_emb, global_feat.mean(dim=0)], dim=0)  # 拼接全局特征到图嵌入

    #     tile_embeddings_dict[tid] = {
    #         'node_embed': node_emb,  # 每个图的节点嵌入
    #         'graph_embed': graph_emb  # 每个图的图嵌入
    #     }

    # return tile_embeddings_dict


# 保存嵌入到pkl文件
def save_embeddings_to_pkl(tile_embeddings_dict, output_path):
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(tile_embeddings_dict, f)
        print(f"Saved embeddings to {output_path}")
    except Exception as e:
        print(f"Failed to save embeddings: {e}")

def load_model(model, pt_path, device):
    try:
        # 加载模型参数
        checkpoint = torch.load(pt_path, map_location=device)
        model.load_state_dict(checkpoint['encoder_state_dict'])  # 加载 encoder 的参数
        print(f"Model loaded successfully from {pt_path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {pt_path}: {e}")
        return None


def find_workdirs(rootdir):
    workdirs = []
    for subdir, dirs, files in os.walk(rootdir):
        if 'timing_graph.npz' in files:
            workdirs.append(subdir)
    return workdirs


def process_workdir(workdir, model_path):
    print(f"[INFO] Processing {workdir}")
    tile_graphs = load_dataset_tile(root_dir=f"{workdir}/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tile_encoder = GCNEncoder(in_dim=17, hidden_dim=64, out_dim=32, num_layers=2, dropout=0.2, use_jk=False)
    tile_encoder = load_model(tile_encoder, model_path, device)

    tile_embeddings_dict = encode_all_tiles(tile_encoder, tile_graphs, device)
    save_embeddings_to_pkl(tile_embeddings_dict, f"{workdir}/tile_embeddings_32.pkl")


if __name__ == '__main__':

    rootdir = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/attention_layer/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/"
    model_path = "/home/wllpro/llwang/yfdai/plgnn/models/ckpt_gae/0/epoch161_val18.7743.pt"

    workdirs = find_workdirs(rootdir)
    for workdir in workdirs:
        process_workdir(workdir, model_path)