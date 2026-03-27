import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
import logging
import sys
from torch.utils.data import random_split

from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset
from dgl.readout import sum_nodes, mean_nodes, max_nodes


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

def safe_mape(y_true, y_pred, epsilon=1e-1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


class HeteroGNNLayer(nn.Module):
    def __init__(self, in_dim_tile, in_dim_rr, edge_dim_tile, hidden_dim):
        super().__init__()
        self.tile2tile_mlp = nn.Sequential(
            nn.Linear(in_dim_tile * 2 + edge_dim_tile, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rr2tile_mlp = nn.Sequential(
            nn.Linear(in_dim_rr + in_dim_tile, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tile2rr_mlp = nn.Sequential(
            nn.Linear(in_dim_tile + in_dim_rr, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def softmin_aggregate(self, messages, tau):
        norms = messages.norm(dim=-1)  # [B, N]
        weights = torch.softmax(-norms / tau, dim=1)
        return torch.sum(messages * weights.unsqueeze(-1), dim=1)

    def forward(self, g, temperature):
        with g.local_scope():
            if g.num_edges('tile_to_tile') > 0:
                g.apply_edges(lambda edges: {
                    'msg': self.tile2tile_mlp(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))
                }, etype='tile_to_tile')
                g.update_all(
                    lambda edges: {'msg': edges.data['msg']},
                    lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], temperature)},
                    etype='tile_to_tile'
                )

            if g.num_edges('to_tile') > 0:
                g.apply_edges(lambda edges: {
                    'msg': self.rr2tile_mlp(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))
                }, etype='to_tile')
                g.update_all(
                    lambda edges: {'msg': edges.data['msg']},
                    lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], temperature)},
                    etype='to_tile'
                )

            if g.num_edges('to_rrnode') > 0:
                g.apply_edges(lambda edges: {
                    'msg': self.tile2rr_mlp(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))
                }, etype='to_rrnode')
                g.update_all(
                    lambda edges: {'msg': edges.data['msg']},
                    lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], temperature)},
                    etype='to_rrnode'
                )

        return g


# class StackedHeteroGNN(nn.Module):
#     def __init__(self, in_dim_tile, in_dim_rr, edge_dim_tile, hidden_dim, num_layers=2, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.layers = nn.ModuleList()

#         for i in range(num_layers):
#             layer = HeteroGNNLayer(
#                 in_dim_tile=hidden_dim if i > 0 else in_dim_tile,
#                 in_dim_rr=hidden_dim if i > 0 else in_dim_rr,
#                 edge_dim_tile=edge_dim_tile,
#                 hidden_dim=hidden_dim
#             )
#             self.layers.append(layer)

#     def forward(self, g):
#         for layer in self.layers:
#             g = layer(g, self.temperature)

#         rr_feats = g.nodes['rrnode'].data['feat']
#         is_sink = g.nodes['rrnode'].data['is_sink']
#         sink_feats = rr_feats[is_sink]

#         return sink_feats


class StackedHeteroGNN(nn.Module):
    def __init__(self, in_dim_tile, in_dim_rr, edge_dim_tile, hidden_dim, num_layers=2, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = HeteroGNNLayer(
                in_dim_tile=hidden_dim if i > 0 else in_dim_tile,
                in_dim_rr=hidden_dim if i > 0 else in_dim_rr,
                edge_dim_tile=edge_dim_tile,
                hidden_dim=hidden_dim
            )
            self.layers.append(layer)

    def forward(self, g):
        for layer in self.layers:
            g = layer(g, self.temperature)

        rr_feats = g.nodes['rrnode'].data['feat']           # 所有 rr 节点的特征
        is_sink = g.nodes['rrnode'].data['is_sink']         # Bool mask
        sink_feats = rr_feats[is_sink]

        if sink_feats.shape[0] == 0:
            raise ValueError("No sink node found in the graph.")

        # 最后一个 sink 节点
        last_sink_feat = sink_feats[-1].unsqueeze(0)                     # shape: [hidden_dim]

        # 图级 readout（mean pooling）
        rr_readout = mean_nodes(g, 'feat', ntype='rrnode')  # shape: [hidden_dim]

        # 拼接为图的最终特征
        final_feat = torch.cat([last_sink_feat, rr_readout], dim=-1)  # shape: [2 * hidden_dim]

        return final_feat.squeeze(0)  # shape: [2 * hidden_dim]



# ------------------------
# SoftMin 异构 GNN 模型
# ------------------------
class HeteroGNNWithSoftMin(nn.Module):
    def __init__(self, in_dim_tile, in_dim_rr, edge_dim_tile, hidden_dim, temperature=0.1):
        super().__init__()
        self.tile2tile_mlp = nn.Sequential(
            nn.Linear(in_dim_tile * 2 + edge_dim_tile, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rr2tile_mlp = nn.Sequential(
            nn.Linear(in_dim_rr + in_dim_tile, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tile2rr_mlp = nn.Sequential(
            nn.Linear(in_dim_tile + in_dim_rr, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.temperature = temperature

    def softmin_aggregate(self, messages, tau):
        norms = messages.norm(dim=-1)  # [B, N]
        weights = torch.softmax(-norms / tau, dim=1)
        return torch.sum(messages * weights.unsqueeze(-1), dim=1)

    def message_func_tile2tile(self, edges):
        # feat = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1)
        feat = torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)
        return {'msg': self.tile2tile_mlp(feat)}

    def message_func_rr2tile(self, edges):
        feat = torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)
        return {'msg': self.rr2tile_mlp(feat)}

    def message_func_tile2rr(self, edges):
        feat = torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)
        return {'msg': self.tile2rr_mlp(feat)}

    def forward(self, g):
        with g.local_scope():
            if g.num_edges('tile_to_tile') > 0:
                g.apply_edges(self.message_func_tile2tile, etype='tile_to_tile')
                g.update_all(
                    lambda edges: {'msg': edges.data['msg']},
                    lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], self.temperature)},
                    etype='tile_to_tile'
                )
            else:
                print('no tile_to_tile edges')


            if g.num_edges('to_tile') > 0:
                g.apply_edges(self.message_func_rr2tile, etype='to_tile')
                g.update_all(
                    lambda edges: {'msg': edges.data['msg']},
                    lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], self.temperature)},
                    etype='to_tile'
                )
            else:
                print('no to_tile edges')
            

            if g.num_edges('to_rrnode') > 0:
                g.apply_edges(self.message_func_tile2rr, etype='to_rrnode')
                g.update_all(
                    lambda edges: {'msg': edges.data['msg']},
                    lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], self.temperature)},
                    etype='to_rrnode'
                )
            else:
                print('no to_rrnode edges')

            rr_feats = g.nodes['rrnode'].data['feat']
            is_sink = g.nodes['rrnode'].data['is_sink']
            sink_feats = rr_feats[is_sink]

            return sink_feats  # sink 节点


# ------------------------
# MLP 回归器
# ------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)



def log_to_file(log_path, message):
    with open(log_path, "a") as f:
        f.write(message + "\n")






class TopKPool(nn.Module):
    def __init__(self, in_dim, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(in_dim, 1)

    def forward(self, x):
        scores = self.score_layer(x).squeeze()
        k = max(1, int(self.ratio * x.size(0)))
        _, topk_idx = torch.topk(scores, k)
        return x[topk_idx], topk_idx


class HeteroGNNWithSoftMinTopK(nn.Module):
    def __init__(self, in_dim_tile, in_dim_rr, edge_dim_tile, hidden_dim, temperature=0.1, pool_ratio=0.5):
        super().__init__()
        self.temperature = temperature

        # 初始传播
        self.tile2tile_mlp = nn.Sequential(
            nn.Linear(in_dim_tile * 2 + edge_dim_tile, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rr2tile_mlp = nn.Sequential(
            nn.Linear(in_dim_rr + in_dim_tile, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tile2rr_mlp = nn.Sequential(
            nn.Linear(in_dim_tile + in_dim_rr, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Pooling 层
        self.pool = TopKPool(hidden_dim, ratio=pool_ratio)

        # Pooling 后传播
        self.post_tile2tile = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim_tile, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.post_rr2tile = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.post_tile2rr = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def softmin_aggregate(self, messages, tau):
        norms = messages.norm(dim=-1)
        weights = torch.softmax(-norms / tau, dim=1)
        return torch.sum(messages * weights.unsqueeze(-1), dim=1)

    def hetero_message_passing(self, g, t2t_mlp, r2t_mlp, t2r_mlp):
        if g.num_edges('tile_to_tile') > 0:
            g.apply_edges(lambda edges: {
                'msg': t2t_mlp(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))
            }, etype='tile_to_tile')
            g.update_all(lambda edges: {'msg': edges.data['msg']},
                         lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], self.temperature)},
                         etype='tile_to_tile')

        if g.num_edges('to_tile') > 0:
            g.apply_edges(lambda edges: {
                'msg': r2t_mlp(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))
            }, etype='to_tile')
            g.update_all(lambda edges: {'msg': edges.data['msg']},
                         lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], self.temperature)},
                         etype='to_tile')

        if g.num_edges('to_rrnode') > 0:
            g.apply_edges(lambda edges: {
                'msg': t2r_mlp(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))
            }, etype='to_rrnode')
            g.update_all(lambda edges: {'msg': edges.data['msg']},
                         lambda nodes: {'feat': self.softmin_aggregate(nodes.mailbox['msg'], self.temperature)},
                         etype='to_rrnode')

    def forward(self, g):
        with g.local_scope():
            # 初始传播
            self.hetero_message_passing(g, self.tile2tile_mlp, self.rr2tile_mlp, self.tile2rr_mlp)

            # rrnode 特征 + TopK Pooling
            rr_feats = g.nodes['rrnode'].data['feat']
            pooled_feats, mask = self.pool(rr_feats)

            # 将选中的节点特征放回去
            new_rr_feats = torch.zeros_like(rr_feats)
            new_rr_feats[mask] = pooled_feats
            g.nodes['rrnode'].data['feat'] = new_rr_feats

            # 再次传播
            self.hetero_message_passing(g, self.post_tile2tile, self.post_rr2tile, self.post_tile2rr)

            # sink 节点特征
            rr_feats = g.nodes['rrnode'].data['feat']
            is_sink = g.nodes['rrnode'].data['is_sink']
            sink_feat = rr_feats[is_sink]

            # 全图 readout (tile 和 rrnode)
            tile_readout = dgl.readout_nodes(g, 'feat', ntype='tile' , op='sum')

            # print(sink_feat.shape, tile_readout.shape)

            final_output = torch.cat([
                sink_feat,
                tile_readout
            ], dim=-1)

            # print(final_output.shape)
            
            return final_output








def train_joint_gnn_mlp(ss_pair_list, log_dir="gnn_log_nd", feat_dim=32, hidden_dim=64, temperature=0.1, epochs=100):
    layer = 4
    name = "concat_gnn_sag"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{layer}.txt")
    setup_logger(log_file)
    logging.info(f"Training started with feat_dim={feat_dim}, hidden_dim={hidden_dim}, temperature={temperature}, epochs={epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gnn = HeteroGNNWithSoftMin(in_dim_tile=feat_dim, in_dim_rr=feat_dim, edge_dim_tile=0,
    #                            hidden_dim=hidden_dim, temperature=temperature).to(device)
    mlp = MLPRegressor(in_dim=hidden_dim*2, hidden_dim=hidden_dim).to(device)

    gnn = HeteroGNNWithSoftMinTopK(
        in_dim_tile=38,
        in_dim_rr=38,
        edge_dim_tile=0,
        hidden_dim=38,
        temperature=0.05
    )


    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(mlp.parameters()),
        lr=0.005,
        weight_decay=0.99
    )

    # 过滤标签合法样本
    filtered = [(g.to(device), float(label)) for g, label in ss_pair_list if label > 0.1]
    
    for g, label in filtered:
        g.nodes['rrnode'].data['is_sink'] = torch.zeros(g.num_nodes('rrnode'), dtype=torch.bool)
        g.nodes['rrnode'].data['is_sink'][1] = True


    dataset = GraphLabelDataset(filtered)

    train_len = int(len(dataset) * 0.8)
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = GraphDataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = GraphDataLoader(test_set, batch_size=64, shuffle=False)


    best_loss = float("inf")

    for epoch in range(epochs):
        gnn.train()
        mlp.train()
        total_loss = 0.0

        for batch_g, batch_y in train_loader:
            batch_g = batch_g.to(device)
            batch_y = batch_y.to(device)

            pred = mlp(gnn(batch_g)).squeeze()
            loss = F.mse_loss(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluation
        gnn.eval()
        mlp.eval()
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for batch_g, batch_y in test_loader:
                batch_g = batch_g.to(device)
                batch_y = batch_y.to(device)
                pred = mlp(gnn(batch_g)).squeeze()

                all_y_true.append(batch_y.cpu())
                all_y_pred.append(pred.cpu())

        y_true = torch.cat(all_y_true).numpy()
        y_pred = torch.cat(all_y_pred).numpy()

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = safe_mape(y_true, y_pred)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save({
                "epoch": epoch,
                "gnn_state_dict": gnn.state_dict(),
                "mlp_state_dict": mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss
            }, os.path.join(log_dir, f"{name}_{layer}.pt"))

        logging.info(f"Epoch {epoch:03d} - TrainLoss: {total_loss:.4f} - Test MSE: {mse:.4f} - Test MAPE: {mape:.4f} - Test R²: {r2:.4f}")


def train_and_compare_with_logs(ss_pair_list, log_dir="gnn_log_nd", feat_dim=32, hidden_dim=64, temperature=0.1, epochs=100):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.txt")
    setup_logger(log_file)
    logging.info(f"Training started with feat_dim={feat_dim}, hidden_dim={hidden_dim}, temperature={temperature}, epochs={epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gnn = HeteroGNNWithSoftMin(in_dim_tile=feat_dim, in_dim_rr=feat_dim, edge_dim_tile=0,
                            #    hidden_dim=hidden_dim, temperature=temperature).to(device)
    gnn = StackedHeteroGNN(
        in_dim_tile=38,
        in_dim_rr=38,
        edge_dim_tile=0,
        hidden_dim=38,
        num_layers=1,
        temperature=0.05
    )
    mlp = MLPRegressor(in_dim=hidden_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(mlp.parameters()), lr=1e-2, weight_decay=1e-4)

    # 提取 sink 特征
    sink_feats, ys = [], []
    for g, label in ss_pair_list:
        if label > 0.1:
            g = g.to(device)
            with torch.no_grad():
                sink_feat = gnn(g).cpu().numpy()
            sink_feats.append(sink_feat)
            ys.append(float(label))

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(sink_feats, ys, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # ---------- XGBoost ----------
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_test_pred = xgb_model.predict(X_test)

    # ---------- MLP ----------
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    best_train_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="Training MLP"):
        mlp.train()
        pred = mlp(X_train_tensor)
        loss = F.mse_loss(pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate on test set
        mlp.eval()
        with torch.no_grad():
            mlp_test_pred = mlp(X_test_tensor).cpu().numpy()

        mse = mean_squared_error(y_test, mlp_test_pred)
        r2 = r2_score(y_test, mlp_test_pred)
        mape = safe_mape(y_test, mlp_test_pred)

        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            torch.save({
                "epoch": epoch,
                "model_state_dict": mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss.item()
            }, os.path.join(log_dir, "best_mlp_checkpoint_1.pt"))

        logging.info(f"Epoch {epoch:03d} - TrainLoss: {loss.item():.4f} - Test MSE: {mse:.4f} - Test MAPE: {mape:.4f} - Test R²: {r2:.4f}")

    # ---------- 最终测试日志 ----------
    logging.info("\n[Final Test Evaluation]")
    logging.info(f"XGBoost - MSE: {mean_squared_error(y_test, xgb_test_pred):.4f}, MAPE: {safe_mape(y_test, xgb_test_pred):.4f}, R²: {r2_score(y_test, xgb_test_pred):.4f}")


class GraphLabelDataset(Dataset):
    def __init__(self, ss_pair_list):
        self.graphs = []
        self.labels = []
        for g, label in ss_pair_list:
            if label > 0.1:
                self.graphs.append(g)
                self.labels.append(label)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return min(5000, len(self.graphs))  # 只取前 5000 个图

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]




if __name__ == '__main__':
    
    workdir = '/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/robot_rl/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing'
    batch_result_dir = os.path.join(workdir, "ss_batch_results")

    # 自动加载所有 batch_*.pkl 文件
    ss_pair_list = []
    for filename in sorted(os.listdir(batch_result_dir)):
        if filename.startswith("ss_pair_batch_") and filename.endswith(".pkl"):
            filepath = os.path.join(batch_result_dir, filename)
            with open(filepath, "rb") as f:
                batch_data = pickle.load(f)
                ss_pair_list.extend(batch_data)

    print(f"✅ 共加载 {len(ss_pair_list)} 个图样本")

    # # 启动训练
    # train_and_compare_with_logs(
    #     ss_pair_list=ss_pair_list,
    #     log_dir="gnn_log_nd",
    #     feat_dim=38,
    #     hidden_dim=38,
    #     temperature=0.05,
    #     epochs=5000
    # )

    train_joint_gnn_mlp(
        ss_pair_list=ss_pair_list,
        log_dir="gnn_log_nd",
        feat_dim=38,
        hidden_dim=38,
        temperature=0.05,
        epochs=5000
    )