import pickle
import os

def split_ss_data(ss_data_path, output_dir, num_splits=100):
    with open(ss_data_path, 'rb') as f:
        ss_data_dict = pickle.load(f)

    missing_tedge_count = 0
    for key, data in ss_data_dict.items():
        if 'tedge_id' not in data:
            missing_tedge_count += 1
            # print(f"⚠️ 警告: 键 {key} 的数据缺少 tedge_id")
    
    if missing_tedge_count > 0:
        print(f" path: {ss_data_path}")
    #     print(f"\n❌ 共发现 {missing_tedge_count} 条数据缺少 tedge_id")
    # else:
    #     print("✅ 所有数据条目均包含 tedge_id")

    keys = list(ss_data_dict.keys())
    batch_size = len(keys) // num_splits + (len(keys) % num_splits > 0)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_splits):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(keys))
        batch_keys = keys[start:end]

        split_data = {k: ss_data_dict[k] for k in batch_keys}
        out_path = os.path.join(output_dir, f'ss_data_batch_{i:03d}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(split_data, f)

    print(f"✅ 完成分割：{ss_data_path} → {output_dir}，共生成 {num_splits} 个子文件")

def process_all_ss_dirs(root_dir, num_splits=100):
    for subdir, _, _ in os.walk(root_dir):
        ss_path = os.path.join(subdir, 'ss_graph_data.pkl')
        if os.path.isfile(ss_path):
            print(f"\n📂 发现目标文件: {ss_path}")
            output_dir = os.path.join(subdir, 'ss_batch_new')
            split_ss_data(ss_path, output_dir, num_splits)
        else:
            continue

if __name__ == "__main__":
    root_dir = "/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/dla_like.small/"  # 替换为你的根目录路径
    process_all_ss_dirs(root_dir, num_splits=100)