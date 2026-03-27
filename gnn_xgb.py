import os
import torch
import pickle
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from hg_nd import StackedHeteroGNN, GraphLabelDataset
from dgl.dataloading import GraphDataLoader
import logging

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def extract_features_with_gnn(dataset, gnn, device):
    loader = GraphDataLoader(dataset, batch_size=64, shuffle=False)
    features, labels = [], []
    gnn.eval()
    with torch.no_grad():
        for g, y in loader:
            g = g.to(device)
            sink_feat = gnn(g).cpu().numpy()
            features.extend(sink_feat)
            labels.extend(y.numpy())
    return np.array(features), np.array(labels)

def train_xgb_from_ckpt(ss_pair_list, ckpt_path, log_dir="xgb_log", feat_dim=38, hidden_dim=38, layer=5):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "xgb_from_gnn.txt")
    setup_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 GNN 模型并加载 checkpoint
    gnn = StackedHeteroGNN(
        in_dim_tile=feat_dim,
        in_dim_rr=feat_dim,
        edge_dim_tile=0,
        hidden_dim=hidden_dim,
        num_layers=layer,
        temperature=0.05
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    gnn.load_state_dict(checkpoint["gnn_state_dict"])
    logging.info(f"✅ Loaded GNN checkpoint from: {ckpt_path}")

    # 预处理数据
    filtered = [(g.to(device), float(label)) for g, label in ss_pair_list if label > 0.1]
    for g, label in filtered:
        g.nodes['rrnode'].data['is_sink'] = torch.zeros(g.num_nodes('rrnode'), dtype=torch.bool)
        g.nodes['rrnode'].data['is_sink'][1] = True

    dataset = GraphLabelDataset(filtered)

    # 提取特征
    X, y = extract_features_with_gnn(dataset, gnn, device)
    logging.info(f"✅ Extracted features shape: {X.shape}, Labels shape: {y.shape}")

    # 划分数据集并训练 XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)

    logging.info(f"\n[✅ XGBoost Test Result]")
    logging.info(f"MSE: {mse:.4f}, MAPE: {mape:.4f}, R²: {r2:.4f}")


if __name__ == "__main__":

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

    train_xgb_from_ckpt(
        ss_pair_list,
        ckpt_path="gnn_log_nd/best_joint_checkpoint_1.pt",
        log_dir="xgb_result",
        feat_dim=38,
        hidden_dim=38,
        layer=1
    )