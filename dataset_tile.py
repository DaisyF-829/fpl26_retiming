import os
import dgl
import pickle
import random
from typing import Tuple
import torch
from dgl import function as fn
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader
from torch.utils.data import random_split

def load_dataset_tile(
    root_dir: str = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_htree0short_40nm/stereovision2/seed_4_inner_0.7_place_circuit_fix_free_algo_criticality_timing/",
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[GraphDataLoader, GraphDataLoader]:
    """
    加载 root_dir 下所有 tiles.pkl 文件中的图，返回训练集和验证集的 DataLoader

    参数:
        root_dir: 根目录，包含多个子目录，每个子目录下有 tiles.pkl
        val_ratio: 验证集比例
        seed: 随机种子
        batch_size: 每个 batch 的图数量
        shuffle: 是否在训练集上打乱
        num_workers: DataLoader 并行读取的线程数

    返回:
        train_loader: 训练集的 GraphDataLoader
        val_loader: 验证集的 GraphDataLoader
    """
    all_graphs = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("tiles.pkl"):  # 确保文件名以 "tiles.pkl" 结尾
                pkl_path = os.path.join(subdir, file)
                try:
                    with open(pkl_path, "rb") as f:
                        tile_dict = pickle.load(f)
                        tile_graphs = list(tile_dict.values())  # 获取图数据
                        all_graphs.extend(tile_graphs)  # 将图添加到 all_graphs
                        print(f"Loaded {len(tile_graphs)} tile graphs from {pkl_path}")
                except Exception as e:
                    print(f"Failed to load {pkl_path}: {e}")

    print(f"Total loaded tile graphs: {len(all_graphs)}")

    for g in all_graphs:
        # with g.local_scope():

        node_feats = g.ndata['feat']  # 假设节点特征的维度是 (num_nodes, node_feat_dim)

        # 检查 edata 是否包含 'feat' 特征，如果没有，则填充 4 维的零向量
        if 'feat' not in g.edata:
            num_edges = g.number_of_edges()
            g.edata['feat'] = torch.zeros(num_edges, 4)  # 4 维零向量

        # 初始化入边特征为 4 维零向量
        num_nodes = g.num_nodes()
        in_feats = torch.zeros(num_nodes, 4)  # 初始化入边特征为 4 维零向量

        # 聚合每个节点的入边特征，计算每个节点的入边特征均值
        g.apply_edges(fn.copy_e('feat', 'm'))  # 将边特征 'feat' 复制到边的特征 'm'
        g.update_all(fn.copy_e('m', 'm'), fn.mean('m', 'in_feats'))  # 聚合入边特征到节点的 'in_feats'

        # 获取聚合后的入边特征（每个节点所有入边的均值）
        in_feats = g.ndata['in_feats']  # in_feats.shape = (num_nodes, 4)

        # 处理没有入边的节点，填充为0（已经初始化为零，实际上可以跳过这个步骤）
        no_in_edges = g.in_degrees(range(num_nodes)) == 0  # 标记没有入边的节点

        # 对于没有入边的节点，其对应的入边特征为 4 维零向量，默认已经是零向量
        in_feats[no_in_edges] = 0.0  # 已经是零向量，因此这步可以省略

        # 拼接节点特征和入边特征
        g.ndata['feat'] = torch.cat([node_feats, in_feats], dim=1) 

        print (g.ndata['feat'].shape)

    return all_graphs


def load_dataset_tile_batch(
    root_dir: str = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_htree0short_40nm/stereovision2/seed_4_inner_0.7_place_circuit_fix_free_algo_criticality_timing/",
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[GraphDataLoader, GraphDataLoader]:
    """
    加载 root_dir 下所有 tiles.pkl 文件中的图，返回训练集和验证集的 DataLoader

    参数:
        root_dir: 根目录，包含多个子目录，每个子目录下有 tiles.pkl
        val_ratio: 验证集比例
        seed: 随机种子
        batch_size: 每个 batch 的图数量
        shuffle: 是否在训练集上打乱
        num_workers: DataLoader 并行读取的线程数

    返回:
        train_loader: 训练集的 GraphDataLoader
        val_loader: 验证集的 GraphDataLoader
    """
    all_graphs = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("tiles.pkl"):  # 确保文件名以 "tiles.pkl" 结尾
                pkl_path = os.path.join(subdir, file)
                try:
                    with open(pkl_path, "rb") as f:
                        tile_dict = pickle.load(f)
                        tile_graphs = list(tile_dict.values())  # 获取图数据
                        all_graphs.extend(tile_graphs)  # 将图添加到 all_graphs
                        print(f"Loaded {len(tile_graphs)} tile graphs from {pkl_path}")
                except Exception as e:
                    print(f"Failed to load {pkl_path}: {e}")

    print(f"Total loaded tile graphs: {len(all_graphs)}")

    for g in all_graphs:
        # with g.local_scope():

        node_feats = g.ndata['feat']  # 假设节点特征的维度是 (num_nodes, node_feat_dim)

        # 检查 edata 是否包含 'feat' 特征，如果没有，则填充 4 维的零向量
        if 'feat' not in g.edata:
            num_edges = g.number_of_edges()
            g.edata['feat'] = torch.zeros(num_edges, 4)  # 4 维零向量

        # 初始化入边特征为 4 维零向量
        num_nodes = g.num_nodes()
        in_feats = torch.zeros(num_nodes, 4)  # 初始化入边特征为 4 维零向量

        # 聚合每个节点的入边特征，计算每个节点的入边特征均值
        g.apply_edges(fn.copy_e('feat', 'm'))  # 将边特征 'feat' 复制到边的特征 'm'
        g.update_all(fn.copy_e('m', 'm'), fn.mean('m', 'in_feats'))  # 聚合入边特征到节点的 'in_feats'

        # 获取聚合后的入边特征（每个节点所有入边的均值）
        in_feats = g.ndata['in_feats']  # in_feats.shape = (num_nodes, 4)

        # 处理没有入边的节点，填充为0（已经初始化为零，实际上可以跳过这个步骤）
        no_in_edges = g.in_degrees(range(num_nodes)) == 0  # 标记没有入边的节点

        # 对于没有入边的节点，其对应的入边特征为 4 维零向量，默认已经是零向量
        in_feats[no_in_edges] = 0.0  # 已经是零向量，因此这步可以省略

        # 拼接节点特征和入边特征
        g.ndata['feat'] = torch.cat([node_feats, in_feats], dim=1) 

        print (g.ndata['feat'].shape)


    # 划分数据集
    random.seed(seed)
    random.shuffle(all_graphs)
    split_idx = int(len(all_graphs) * (1 - val_ratio))
    train_graphs = all_graphs[:split_idx]
    val_graphs = all_graphs[split_idx:]

    print(f"Train set size: {len(train_graphs)}")
    print(f"Validation set size: {len(val_graphs)}")

    # 创建 DGL 的 DataLoader
    train_loader = GraphDataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = GraphDataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_dataset_tile_batch(batch_size=4)

    print("Checking edata['feat'] for a few training batches...\n")
    for i, batched_graph in enumerate(train_loader):
        batched_graph = batched_graph.to(device)
        
        # 检查边特征
        if "feat" in batched_graph.edata:
            print(f"Batch {i}: edata['feat'] exists with shape {batched_graph.edata['feat'].shape}")
        else:
            print(f"Batch {i}: ❌ edata['feat'] NOT found!")
        
        if i >= 2:  # 只检查前3个batch
            break