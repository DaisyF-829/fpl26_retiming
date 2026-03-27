import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error
from dgl.nn import HeteroGraphConv, GraphConv, SAGEConv
from concurrent.futures import ThreadPoolExecutor
from dgl.nn.pytorch import JumpingKnowledge
import pickle
from tqdm import tqdm
import os

class TileRRGEncoder(nn.Module):
    def __init__(self, in_node_dim=13, in_global_dim=16, hidden_dim=64, out_dim=256, num_layers=4, jk_mode='cat'):
        super().__init__()
        self.input_dim = in_node_dim + in_global_dim
        self.in_global_dim = in_global_dim
        self.num_layers = num_layers

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            self.gnn_layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type='mean'))

        self.jk = JumpingKnowledge(mode=jk_mode)
        jk_out_dim = hidden_dim if jk_mode != 'cat' else hidden_dim * num_layers

        # 改为接收 global_feat 拼接
        self.project_graph = nn.Sequential(
            nn.Linear(jk_out_dim + in_global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, g):
        # 拼接 node feat 和 global feat
        x = torch.cat([g.ndata['feat'], g.ndata['global']], dim=1)
        h_list = []

        for layer in self.gnn_layers:
            x = layer(g, x)
            x = F.relu(x)
            h_list.append(x)

        x_jk = self.jk(h_list)
        g.ndata['h'] = x_jk

        # mean pooling 节点嵌入
        hg = dgl.mean_nodes(g, 'h')  # [batch_size, hidden_dim or hidden_dim*num_layers]

        # 取 global_feat（假设每个 tile 的节点的 global_feat 是一致的）
        global_feat = dgl.mean_nodes(g, 'global')

        input_to_mlp = torch.cat([hg, global_feat], dim=1)
        global_embed = self.project_graph(input_to_mlp)

        return x_jk, global_embed


class HeteroGNNGraphPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        # 第一层异构 GNN
        self.layer1 = HeteroGraphConv({
            ('tile', 'tile_to_tile', 'tile'): GraphConv(in_dim, hidden_dim),
            ('rrnode', 'to_tile', 'tile'): GraphConv(in_dim, hidden_dim),
            ('tile', 'to_rrnode', 'rrnode'): GraphConv(in_dim, hidden_dim),
        }, aggregate='sum')

        # 第二层异构 GNN
        self.layer2 = HeteroGraphConv({
            ('tile', 'tile_to_tile', 'tile'): GraphConv(hidden_dim, hidden_dim),
            ('rrnode', 'to_tile', 'tile'): GraphConv(hidden_dim, hidden_dim),
            ('tile', 'to_rrnode', 'rrnode'): GraphConv(hidden_dim, hidden_dim),
        }, aggregate='sum')

        # 图级聚合 MLP（使用tile节点的平均嵌入）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)  # 输出一个标量
        )

    def forward(self, g, inputs):
        """
        inputs: dict of node_type -> feature tensor
        """
        h = self.layer1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.layer2(g, h)
        h = {k: F.relu(v) for k, v in h.items()}

        # 图级 readout：取所有 tile 节点的平均作为图表示
        with g.local_scope():
            g.nodes['tile'].data['h'] = h['tile']
            hg = dgl.mean_nodes(g, 'h', ntype='tile')  # shape: [batch, hidden_dim]

        return self.mlp(hg)

def encode_all_tiles(tile_encoder, tile_graphs, device):
    """
    tile_graphs: {tile_id: DGLGraph}, tile_id 连续从0开始
    返回：
      - tile_node_embeddings: list, 每个元素为该 tile 的 node embedding [num_rrnodes, hidden_dim]
      - tile_graph_embeddings: tensor, shape [num_tiles, hidden_dim]
    """
    tile_encoder.train()

    tile_ids = sorted(tile_graphs.keys())  # 必须是从 0 开始连续
    # print(f"encode_all_tiles: {tile_ids}")
    graphs = [tile_graphs[tid].to(device) for tid in tile_ids]

    batched_graph = dgl.batch(graphs)
    node_embed, global_embed = tile_encoder(batched_graph)

    # 拆分 node embedding
    node_counts = [g.num_nodes() for g in graphs]
    split_node_embeds = torch.split(node_embed, node_counts, dim=0)

    tile_node_embeddings = {
        tid: emb for tid, emb in zip(tile_ids, split_node_embeds)
    }

    return tile_node_embeddings, global_embed


# 5. 定义从 ss_data 提取节点嵌入的函数
def extract_embeddings_from_ss(ss_data, tile_nodes_embedding):
    """
    从 ss_data 提取源节点和汇节点的嵌入
    ss_data: ss数据字典
    tile_nodes_embedding: 每个 tile 的嵌入
    """
    # 存储源节点和汇节点的嵌入
    src_embeddings = []
    sink_embeddings = []

    # 1. 提取源节点的嵌入
    for i in range(ss_data['src_rr_indexes'].shape[1]):
        src_tile_idx, src_rr_idx = ss_data['src_rr_indexes'][:, i]
        
        # 获取源节点嵌入
        # print(f"src_tile_idx: {src_tile_idx}, src_rr_idx: {src_rr_idx}")
        # print(f"tile_nodes_embedding keys: {list(tile_nodes_embedding.keys())}")
        src_tile_embeds = tile_nodes_embedding[src_tile_idx.item()]
        src_embedding = src_tile_embeds[src_rr_idx.item()]
        src_embeddings.append(src_embedding)

    # 2. 提取汇节点的嵌入
    for i in range(ss_data['sink_rr_indexes'].shape[1]):
        sink_tile_idx, sink_rr_idx = ss_data['sink_rr_indexes'][:, i]

        # 获取汇节点嵌入
        sink_embedding = tile_nodes_embedding[sink_tile_idx.item()][sink_rr_idx.item()]
        sink_embeddings.append(sink_embedding)

    # 3. 计算源节点和汇节点的平均嵌入
    src_embedding_avg = torch.mean(torch.stack(src_embeddings), dim=0) if len(src_embeddings) > 0 else torch.zeros_like(src_embeddings[0])
    sink_embedding_avg = torch.mean(torch.stack(sink_embeddings), dim=0) if len(sink_embeddings) > 0 else torch.zeros_like(sink_embeddings[0])

    return src_embedding_avg, sink_embedding_avg


def build_hetero_graph(ss_data, tile_global_embedding, tile_nodes_embedding):
    """
    构建异构图，图中包含tile和rrnode两种节点类型，及相关的边
    ss_data: ss数据字典
    tile_global_embedding: 全局特征的嵌入
    tile_nodes_embedding: 节点特征的嵌入
    """
    # 提取源节点和汇节点的嵌入
    src_embedding, sink_embedding = extract_embeddings_from_ss(ss_data, tile_nodes_embedding)

    # 提取边
    tile_edge_src, tile_edge_dst = ss_data['tile_edges']

    rr_ids, tile_ids = ss_data['rr_tile_edges']
    rr_to_tile_src = []
    rr_to_tile_dst = []
    tile_to_rr_src = []
    tile_to_rr_dst = []

    for rr_id, tile_id in zip(rr_ids, tile_ids):
        if rr_id == 0:
            # rrnode[0] → tile
            rr_to_tile_src.append(0)
            rr_to_tile_dst.append(tile_id)
        elif rr_id == 1:
            # tile → rrnode[1]
            tile_to_rr_src.append(tile_id)
            tile_to_rr_dst.append(1)

    data_dict = {
        ('tile', 'tile_to_tile', 'tile'): (tile_edge_src.to(torch.int64), tile_edge_dst.to(torch.int64)),
        ('rrnode', 'to_tile', 'tile'): (torch.tensor(rr_to_tile_src, dtype=torch.int64), torch.tensor(rr_to_tile_dst, dtype=torch.int64)),
        ('tile', 'to_rrnode', 'rrnode'): (torch.tensor(tile_to_rr_src, dtype=torch.int64), torch.tensor(tile_to_rr_dst, dtype=torch.int64)),
    }

    # 提取节点
    num_tiles = ss_data['tiles'].shape[0]
    num_nodes_dict = {
        'tile': num_tiles,  # tile 节点的数量
        'rrnode': 2         # 固定为 2 个 rrnode 节点
    }

    graph = dgl.heterograph(data_dict, num_nodes_dict)

    # 设置节点特征
    # 为tile节点设置特征，从tile_global_embedding中查找对应的嵌入
    graph.nodes['tile'].data['feat'] = tile_global_embedding[ss_data['tiles']]

    # 为rrnode节点设置特征，使用从ss_data中提取的嵌入
    graph.nodes['rrnode'].data['feat'] = torch.stack([src_embedding, sink_embedding])

    label = ss_data['net_delay']

    return graph, label


def collate(samples):
    graphs, labels = zip(*samples)
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(labels)

class SSGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_label_pairs):
        self.data = graph_label_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_all_hetero_graphs(ss_data_dict, tile_node_embeds, tile_graph_embeds):
    # for tile_id, emb in tile_node_embeds.items():
    #     print(f"tile {tile_id}: embedding shape = {emb.shape}")
    # print (f"tile_graph_embeds: {tile_graph_embeds.shape}")
    results = []
    def _build(k):
        g, label = build_hetero_graph(ss_data_dict[k], tile_graph_embeds, tile_node_embeds)
        return g, torch.tensor([label], dtype=torch.float32)

    with ThreadPoolExecutor() as executor:
        for result in executor.map(_build, ss_data_dict.keys()):
            results.append(result)

    return results



def train_joint_batch(tile_graphs, ss_data_dict, epochs=500, lr=1e-4, log_path="log/train.log"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tile_encoder = TileRRGEncoder(in_node_dim=13, in_global_dim=2).to(device)
    hetero_predictor = HeteroGNNGraphPredictor(in_dim=256, hidden_dim=64, out_dim=1).to(device)
    optimizer = torch.optim.Adam(list(tile_encoder.parameters()) + list(hetero_predictor.parameters()), lr=lr)
    loss_fn = torch.nn.MSELoss()
    epsilon = 1e-6

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as log_file:
        for epoch in range(epochs):
            log_file.write(f"Epoch {epoch+1}/{epochs}\n")
            log_file.flush()

            tile_encoder.train()
            hetero_predictor.train()
            optimizer.zero_grad()

            total_loss = 0.0
            total_mape = 0.0
            total_samples = 0

            for ss_key in ss_data_dict.keys():
                # === 每个子图都重算 tile embedding ===
                tile_node_embeds, tile_graph_embeds = encode_all_tiles(tile_encoder, tile_graphs, device)

                # === 构建该 ss 对应的异构图 ===
                g, label = build_hetero_graph(ss_data_dict[ss_key], tile_graph_embeds, tile_node_embeds)
                g = g.to(device)
                label = label.to(device)

                inputs = {
                    'tile': g.nodes['tile'].data['feat'],
                    'rrnode': g.nodes['rrnode'].data['feat']
                }

                pred = hetero_predictor(g, inputs).squeeze()
                label = label.squeeze()

                loss = loss_fn(pred, label)
                loss.backward()  # 当前图独立反传

                total_loss += loss.item()
                total_mape += torch.abs((pred - label) / (label + epsilon)).item()
                total_samples += 1

            optimizer.step()

            avg_loss = total_loss / total_samples
            avg_mape = total_mape / total_samples
            log_file.write(f"  Avg MSE Loss = {avg_loss:.6f}, Avg MAPE = {avg_mape:.4%}\n")
            log_file.flush()

    return tile_encoder, hetero_predictor



if __name__ == '__main__':
    
    # 加载 tile_graph 和 ss_data
    with open('tiles.pkl', 'rb') as f:
        tile_graphs = pickle.load(f)


    with open('ss_graph_data.pkl', 'rb') as f:
        ss_data_dict = pickle.load(f)

    # 开始训练
    tile_encoder, predictor = train_joint_batch(tile_graphs, ss_data_dict)
