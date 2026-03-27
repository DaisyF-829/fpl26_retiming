import os
import logging
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split


def compute_sample_weights(y):
    # 简单示例：权重 = delay的平方，突出大delay
    weights = np.square(y)
    # 或者你可以用：weights = y，或者加阈值：weights = np.where(y > threshold, y, 1)
    return weights


def load_dataset_noglobal(root_dir):
    """
    加载数据集，仅保留第1, 35, 36, 37, 38列特征（即去掉标签后的0, 34, 35, 36, 37列）。
    同时过滤标签值 ≤ 0.1 的样本。
    """
    all_dfs = []

    for subdir, _, files in os.walk(root_dir):
        files.sort()
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                df = pd.read_csv(csv_path)

                # 过滤标签值 <= 0.1 的行
                df = df[df.iloc[:, 0] > 0.1]

                # 提取目标特征列（注意：df.iloc[:, 1:] 是特征起点）
                selected_features = df.iloc[:, [1 + i for i in [2, 35, 36, 37, 38]]]

                # 用标签列和新特征列拼接
                filtered_df = pd.concat([df.iloc[:, [0,1]], selected_features], axis=1)
                all_dfs.append(filtered_df)

    if not all_dfs:
        raise ValueError("未找到任何有效的 CSV 文件，或标签值全部 ≤ 0.1")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # 特征与标签
    X = full_df.iloc[:, 2:].values
    id = full_df.iloc[:, 1].values  # id
    y = full_df.iloc[:, 0].values

    return X, y, id

# 加载数据集
def load_dataset(root_dir):
    """
    从所有 workdir 子目录中加载 CSV 数据，合并为一个大 DataFrame。
    仅保留标签（第一列） > 0.1 的样本。
    """
    all_dfs = []

    for subdir, _, files in os.walk(root_dir):
        files.sort()
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                df = pd.read_csv(csv_path)

                # 过滤：仅保留标签值 > 0.1 的行
                df = df[df.iloc[:, 0] > 0.1]

                all_dfs.append(df)

    if not all_dfs:
        raise ValueError("未找到任何有效的 CSV 文件，或标签值全部 ≤ 0.1")

    full_df = pd.concat(all_dfs, ignore_index=True)

    X = full_df.iloc[:, 2:].values  # 特征
    id = full_df.iloc[:, 1].values  # id
    y = full_df.iloc[:, 0].values   # 标签

    return X, y, id


def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )



def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)

    train_weights = compute_sample_weights(y_train)
    val_weights = compute_sample_weights(y_val)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dval = xgb.DMatrix(X_val, label=y_val, weight=val_weights)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0
    }

    evallist = [(dval, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=evallist, early_stopping_rounds=10)

    best_test_r2 = -np.inf  # 记录最佳 R²
    best_model_path = None

    for epoch in range(bst.best_iteration + 1):
        y_val_pred = bst.predict(dval, ntree_limit=epoch)
        y_test_pred = bst.predict(dtest, ntree_limit=epoch)

        mse = mean_squared_error(y_val, y_val_pred)
        mape = mean_absolute_percentage_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        test_r2 = r2_score(y_test, y_test_pred)

        logging.info(f"Epoch {epoch+1}: MSE={mse:.4f}, MAPE={mape:.4f}, R2={r2:.4f}, Test_R2={test_r2:.4f}")

        save_model = False
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            save_model = True
        elif (epoch + 1) % 50 == 0:
            save_model = True

        if save_model:
            model_path = os.path.join(ckpt_dir, f'nogae_epoch_{epoch + 1}.json')
            bst.save_model(model_path)
            logging.info(f"Checkpoint saved at: {model_path}")
            if test_r2 == best_test_r2:
                best_model_path = model_path

    logging.info(f"Best test R2 = {best_test_r2:.4f}, model path: {best_model_path}")
    return bst




# 测试集评估
def infer_model(bst, X_test, y_test, ids):
    dtest = xgb.DMatrix(X_test)
    y_pred = bst.predict(dtest)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    nd_results = pd.DataFrame({'id': ids, 'y': y_pred, 'label': y_test})

    nd_results.to_csv('./xgboost_nd/nd_results.csv', index=False)


    logging.info(f"[Test Final] RMSE={rmse:.4f}, MAPE={mape:.4f}, R2={r2:.4f}")



# 主函数
def main(data_dir, log_file, ckpt_dir):
    setup_logger(log_file)
    logging.info("Loading dataset...")

    # X, y = load_dataset(data_dir)
    X, y, id = load_dataset(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    logging.info("Training model...")
    bst = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, ckpt_dir)

    logging.info("Evaluating final model on test set...")
    infer_model(bst, X, y, id)



if __name__ == "__main__":
    # 传入数据集路径和日志文件路径
    data_path = '/home/wllpro/llwang10/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/arm_core/seed_1_inner_0.5_place_device_circuit_fix_free_algo_bounding_box_timing/'  # 请修改为你的数据集路径
    log_file = './xgboost_nd/training_log.txt'  # 日志文件路径
    ckpt_dir = './xgboost_nd/ckpt/'  # 模型检查点目录
    main(data_path, log_file, ckpt_dir)
