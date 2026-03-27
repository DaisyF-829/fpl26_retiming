import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import re


def load_timing_graph(npz_path):
    data = np.load(npz_path)
    tnode_type = data['tnode_type']
    tnode_rt_time = data['tnode_rt_time']
    tedge_src = data['tedge_src']
    tedge_dst = data['tedge_dst']
    tedge_delay = data['tedge_delay']
    return tnode_type, tnode_rt_time, tedge_src, tedge_dst, tedge_delay


def build_graph(tedge_src, tedge_dst, tedge_delay):
    G = nx.DiGraph()
    for s, d, delay in zip(tedge_src, tedge_dst, tedge_delay):
        G.add_edge(int(s), int(d), delay=float(delay))
    return G


def compute_rt_time(G, num_nodes):
    rt_time = np.zeros(num_nodes, dtype=float)
    in_degrees = dict(G.in_degree())
    topo_order = list(nx.topological_sort(G))
    delay_dict = {(u, v): data['delay'] for u, v, data in G.edges(data=True)}

    for node in topo_order:
        if in_degrees[node] == 0:
            rt_time[node] = 0
        else:
            preds = list(G.predecessors(node))
            rt_time[node] = max(rt_time[pred] + delay_dict[(pred, node)] for pred in preds)
    return rt_time


def evaluate_metrics(y_true, y_pred, valid_mask):
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]
    mape = mean_absolute_percentage_error(y_true_v, y_pred_v)
    r2 = r2_score(y_true_v, y_pred_v)
    return mape, r2


def weighted_metrics(y_true, y_pred, top_ratio=0.1):
    mape_all, r2_all = evaluate_metrics(y_true, y_pred, valid_mask=np.ones_like(y_true, dtype=bool))
    top_k = int(len(y_true) * top_ratio)
    idx_sorted = np.argsort(y_true)[::-1]
    idx_top = idx_sorted[:top_k]
    mape_top = mean_absolute_percentage_error(y_true[idx_top], y_pred[idx_top])
    r2_top = r2_score(y_true[idx_top], y_pred[idx_top])
    return mape_all, r2_all, mape_top, r2_top, idx_top


def topk_overlap(ids1, ids2, k=100):
    set1 = set(ids1[:k])
    set2 = set(ids2[:k])
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0
    return intersection, jaccard


def main(npz_path, csv_path=None, out_csv_path="timing_prediction_results.csv"):
    tnode_type, tnode_rt_time, tedge_src, tedge_dst, tedge_delay_orig = load_timing_graph(npz_path)
    num_nodes = len(tnode_type)
    valid_mask = tnode_rt_time > 0.1

    # 1. 先用真实label做边延时跑一次compute_rt_time，得到rt_time_label
    if csv_path is None:
        print("需要csv_path来获取label进行修正")
        return

    df = pd.read_csv(csv_path)

    # 2. 构造基于label的边延时数组（覆盖原始延时）
    tedge_delay_label = tedge_delay_orig.copy()
    for idx, label_delay in zip(df['id'], df['label']):
        idx = int(idx)
        if 0 <= idx < len(tedge_delay_label):
            tedge_delay_label[idx] = float(label_delay)
    G_label = build_graph(tedge_src, tedge_dst, tedge_delay_label)
    rt_time_label = compute_rt_time(G_label, num_nodes)

    # 3. 计算修正量 = 真实到达时间 - rt_time_label
    correction = tnode_rt_time - rt_time_label
    pred_rt_time_label = tnode_rt_time

    # 4. 用csv预测y构造预测的边延时
    tedge_delay_y = tedge_delay_orig.copy()
    for idx, y_delay in zip(df['id'], df['y']):
        idx = int(idx)
        if 0 <= idx < len(tedge_delay_y):
            tedge_delay_y[idx] = float(y_delay)
    G_y = build_graph(tedge_src, tedge_dst, tedge_delay_y)
    pred_rt_time_y = compute_rt_time(G_y, num_nodes)

    # 5. 用原始延时跑一遍预测
    G_orig = build_graph(tedge_src, tedge_dst, tedge_delay_orig)
    pred_rt_time_orig = compute_rt_time(G_orig, num_nodes)

    # 6. 对y预测和原始预测分别加修正量
    pred_rt_time_y_corr = pred_rt_time_y + correction
    pred_rt_time_orig_corr = pred_rt_time_orig + correction

    # 7. 计算各种指标
    print("=== 基于 CSV label 预测的rt_time_label ===")
    mape_label, r2_label, mape_label_top, r2_label_top, idx_label_top = weighted_metrics(tnode_rt_time[valid_mask], pred_rt_time_label[valid_mask])
    print(f"MAPE={mape_label:.6f}, R2={r2_label:.6f}")

    

    print("=== 直接用 CSV y 预测 (不修正) ===")
    mape_y, r2_y, mape_y_top, r2_y_top, idx_y_top = weighted_metrics(tnode_rt_time[valid_mask], pred_rt_time_y[valid_mask])
    print(f"MAPE={mape_y:.6f}, R2={r2_y:.6f}, Top10% MAPE={mape_y_top:.6f}, Top10% R2={r2_y_top:.6f}")

    print("=== 修正后用 CSV y 预测 ===")
    mape_yc, r2_yc, mape_yc_top, r2_yc_top, idx_yc_top = weighted_metrics(tnode_rt_time[valid_mask], pred_rt_time_y_corr[valid_mask])
    print(f"MAPE={mape_yc:.6f}, R2={r2_yc:.6f}, Top10% MAPE={mape_yc_top:.6f}, Top10% R2={r2_yc_top:.6f}")

    print("=== 直接用原始延时预测 (不修正) ===")
    mape_o, r2_o, mape_o_top, r2_o_top, idx_o_top = weighted_metrics(tnode_rt_time[valid_mask], pred_rt_time_orig[valid_mask])
    print(f"MAPE={mape_o:.6f}, R2={r2_o:.6f}, Top10% MAPE={mape_o_top:.6f}, Top10% R2={r2_o_top:.6f}")

    print("=== 修正后用原始延时预测 ===")
    mape_oc, r2_oc, mape_oc_top, r2_oc_top, idx_oc_top = weighted_metrics(tnode_rt_time[valid_mask], pred_rt_time_orig_corr[valid_mask])
    print(f"MAPE={mape_oc:.6f}, R2={r2_oc:.6f}, Top10% MAPE={mape_oc_top:.6f}, Top10% R2={r2_oc_top:.6f}")

    # 8. 计算前100条路径id重合度，这里用节点id模拟路径id
    intersection_orig, jaccard_orig = topk_overlap(np.arange(num_nodes)[valid_mask][idx_oc_top], np.arange(num_nodes)[valid_mask][idx_label_top])
    intersection_y, jaccard_y = topk_overlap(np.arange(num_nodes)[valid_mask][idx_yc_top], np.arange(num_nodes)[valid_mask][idx_label_top])

    print(f"前100路径重合度（原始 vs 修正）交集={intersection_orig}, Jaccard={jaccard_orig:.4f}")
    print(f"前100路径重合度（CSV y vs 修正）交集={intersection_y}, Jaccard={jaccard_y:.4f}")

    # 9. 保存结果到csv
    df_save = pd.DataFrame({
        "node_id": np.arange(num_nodes)[valid_mask],
        "true_rt_time": tnode_rt_time[valid_mask],
        "rt_time_label": rt_time_label[valid_mask],
        "pred_y": pred_rt_time_y[valid_mask],
        "pred_y_corrected": pred_rt_time_y_corr[valid_mask],
        "pred_orig": pred_rt_time_orig[valid_mask],
        "pred_orig_corrected": pred_rt_time_orig_corr[valid_mask],
    })

    # plot_two_preds_top10_scatter(tnode_rt_time, pred_rt_time_y_corr, pred_rt_time_orig_corr, valid_mask)


    df_save.to_csv(out_csv_path, index=False)
    print(f"结果保存到 {out_csv_path}")

    ratios = compute_topk_error_ratios(tnode_rt_time[valid_mask], pred_rt_time_y_corr[valid_mask])
    for k, v in ratios.items():
        print(f"{k}: {v:.2%}")
    
    ratios = compute_topk_error_ratios(tnode_rt_time[valid_mask], pred_rt_time_orig_corr[valid_mask])
    for k, v in ratios.items():
        print(f"{k}: {v:.2%}")


def plot_two_preds_top10_scatter(tnode_rt_time, pred1, pred2, valid_mask, out_png='top10_delay_two_preds_scatter.png'):
    # 选前10%真实最大delay的有效节点索引
    valid_true = tnode_rt_time[valid_mask]
    valid_indices = np.arange(len(tnode_rt_time))[valid_mask]
    top_k = int(len(valid_true) * 0.1)
    top_indices = valid_indices[np.argsort(valid_true)[-top_k:]]

    y_true = tnode_rt_time[top_indices]
    y_pred1 = pred1[top_indices]
    y_pred2 = pred2[top_indices]

    plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()

    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']

    metrics = []

    for idx, (name, y_pred) in enumerate([
        ("Corrected XGBoost Prediction", y_pred1),
        ("Corrected Original Prediction", y_pred2),
    ]):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r = np.corrcoef(y_true, y_pred)[0,1]
        r2 = r2_score(y_true, y_pred)
        metrics.append(f"{name}: MAPE={mape:.2f}%, R={r:.3f}, R²={r2:.3f}")

        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df.to_csv(f"{safe_name}_top10_results.csv", index=False)

        plt.scatter(
            y_true, y_pred,
            c=colors[idx],
            marker=markers[idx],
            alpha=0.6,
            edgecolors='w',
            linewidths=0.5,
            label=metrics[-1],
            s=80
        )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)

    plt.xlabel('True Arrival Time', fontsize=20)
    plt.ylabel('Predicted Arrival Time', fontsize=20)
    plt.title('True vs. Predicted Arrival Time (Top 10% Delay Nodes)', fontsize=20, pad=20)

    plt.legend(
        loc='upper left',
        frameon=True,
        fontsize=10,
        markerscale=1.2,
        edgecolor='black'
    )
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {out_png}")


def compute_topk_error_ratios(y_true, y_pred, top_percent=0.1, thresholds=[0.02, 0.05, 0.10, 0.15, 0.20]):
    """
    For the top `top_percent` largest y_true, compute the percentage of samples whose
    absolute percentage error is less than each threshold in `thresholds`.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    top_k = int(len(y_true) * top_percent)
    idx_top = np.argsort(y_true)[-top_k:]  # Top-k by ground-truth

    ape_topk = np.abs((y_true[idx_top] - y_pred[idx_top]) / y_true[idx_top])  # APE

    results = {}
    for t in thresholds:
        ratio = np.mean(ape_topk < t)
        results[f"Top {int(top_percent*100)}% Error < {int(t*100)}%"] = ratio

    return results


def topk_overlap(ids1, ids2, k=10):
    set1 = set(ids1[:k])
    set2 = set(ids2[:k])
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0
    return intersection, jaccard


if __name__ == "__main__":
    npz_path = '/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/robot_rl/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/timing_graph.npz'
    csv_path = "/home/wllpro/llwang/yfdai/plgnn/models/xgboost_nd/nd_results.csv"
    main(npz_path, csv_path, out_csv_path="timing_pred_corrected.csv")
