import torch
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
from os.path import join
from tqdm import tqdm
import os, math, json
from collections import defaultdict

from model_autoencoder import build_graph_autoencoder
from dataset_tile import load_dataset_tile_batch
from copy import deepcopy
from itertools import product
import subprocess

# def train_one_epoch(model, data, optimizer, epoch, max_grad_norm, device, log_file):
#     model.train()
#     total_loss, total_rec_loss, total_kl = 0, 0, 0

#     for g in data:
#         g = g.to(device)
#         optimizer.zero_grad()
#         orig_feat, rec_feat, node_emb, graph_emb = model(g)
#         loss, rec_loss, kl_loss = model.loss_fn(orig_feat, rec_feat, node_emb=node_emb)

#         loss.backward()
#         if max_grad_norm > 0:
#             clip_grad_norm_(model.parameters(), max_grad_norm)
#         optimizer.step()

#         total_loss += loss.item()
#         total_rec_loss += rec_loss.item()
#         total_kl += kl_loss.item() if kl_loss is not None else 0

#     avg_loss = total_loss / len(data)
#     avg_rec_loss = total_rec_loss / len(data)
#     avg_kl = total_kl / len(data)

#     log_file.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}, Rec: {avg_rec_loss:.6f}, KL: {avg_kl:.6f}\n")



# def eval_model(model, dataloader, epoch, device, log_file):
#     model.eval()
#     total_loss, total_rec_loss, total_kl = 0, 0, 0
#     with torch.no_grad():
#         for batched_graph in dataloader:
#             batched_graph = batched_graph.to(device)
#             orig_feat, rec_feat, node_emb, graph_emb = model(batched_graph)
#             loss, rec_loss, kl_loss = model.loss_fn(orig_feat, rec_feat, node_emb=node_emb)

#             total_loss += loss.item()
#             total_rec_loss += rec_loss.item()
#             total_kl += kl_loss.item() if kl_loss is not None else 0

#     avg_loss = total_loss / len(data)
#     avg_rec_loss = total_rec_loss / len(data)
#     avg_kl = total_kl / len(data)

#     log_file.write(f"[Epoch {epoch}] Val   Loss: {avg_loss:.6f}, Rec: {avg_rec_loss:.6f}, KL: {avg_kl:.6f}\n")
#     return avg_loss



def save_checkpoint(encoder, decoder, optimizer, save_path, epoch):
    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, save_path)

# def main(config):
#     device = torch.device(config["device"])
#     os.makedirs(config["log_dir"], exist_ok=True)
#     os.makedirs(config["ckpt_dir"], exist_ok=True)

#     log_path = os.path.join(config["log_dir"], "train.log")
#     log_file = open(log_path, "w")

#     train_loader, val_loader = load_dataset_tile_batch()

#     # 固定输入维度
#     in_node_dim = 13
#     in_edge_dim = 4

#     model = build_graph_autoencoder(
#         model_type=config["model_type"],
#         in_node_dim=in_node_dim,
#         in_edge_dim=in_edge_dim,
#         hidden_dim=config["hidden_dim"],
#         latent_dim=config["latent_dim"],
#         num_layers=config["num_layers"],
#         dropout=config["dropout"],
#         use_jk=config.get("use_jk", False),
#         jk_mode=config.get("jk_mode", "last"),
#         kl_weight=config.get("kl_weight", 0.0)
#     ).to(device)

#     optimizer = Adam(model.parameters(), lr=config["lr"])
#     scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])

#     best_val_loss = math.inf

#     # if "checkpoint" in config and os.path.exists(config["checkpoint"]):
#     #     checkpoint = torch.load(config["checkpoint"])
#     #     model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
#     #     model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
#     #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     #     start_epoch = checkpoint["epoch"] + 1  # Continue from the next epoch
#     #     best_val_loss = checkpoint.get("best_val_loss", math.inf)
#     #     print(f"Loaded checkpoint from {config['checkpoint']} and continuing from epoch {start_epoch}")
#     # else:
#     #     start_epoch = 1
#     if kl_weight > 0:
#         config["checkpoint"] = "/home/wllpro/llwang/yfdai/plgnn/models/ckpt_gae/1/epoch1760_val8.5606.pt"
#     else:
#         config["checkpoint"] = "/home/wllpro/llwang/yfdai/plgnn/models/ckpt_gae/0/epoch1877_val7.6388.pt"


#     for epoch in tqdm(range(1, config["epochs"] + 1)):
#         train_one_epoch(model, train_loader, optimizer, epoch, config["max_grad_norm"], device, log_file)
#         val_loss = eval_model(model, val_loader, epoch, device, log_file)

#         if epoch % config["save_freq"] == 0 or val_loss < best_val_loss:
#             save_path = join(config["ckpt_dir"], f"epoch{epoch}_val{val_loss:.4f}.pt")
#             save_checkpoint(model.encoder, model.decoder, optimizer, save_path, epoch, device)
#             log_file.write(f"[Epoch {epoch}] ✅ Checkpoint saved to {save_path}\n")
#             best_val_loss = min(best_val_loss, val_loss)

#         if epoch % config["lr_decay_freq"] == 0:
#             scheduler.step()
#             log_file.write(f"[Epoch {epoch}] 🔻 Learning rate decayed\n")

#     log_file.close()
#     return best_val_loss


if __name__ == "__main__":
    # 超参数搜索空间（你可以调整范围）
    search_space = {
        "model_type": ["gcn"],
        "hidden_dim": [64],
        "latent_dim": [8],
        "num_layers": [2],
        "dropout": [0.2],
        "use_jk": [False],
        "kl_weight": [0.0, 0.2],
        "lr": [1e-3],
        "lr_decay": [0.99],
        "lr_decay_freq": [80],
        "save_freq": [20],
        "epochs": [2000],
        "max_grad_norm": [5.0],
        "jk_mode": ["concat"],
        "device": ["cpu"],
        
    }

    keys, values = zip(*search_space.items())
    combinations = list(product(*values))

    os.makedirs("log_gae", exist_ok=True)
    os.makedirs("ckpt_gae", exist_ok=True)


    for idx, comb in enumerate(combinations):
        cfg = dict(zip(keys, comb))
        cfg["id"] = idx + 2
        cfg["log_dir"] = f"log_gae/{idx}"
        cfg["ckpt_dir"] = f"ckpt_gae/{idx}"
        os.makedirs(cfg["log_dir"], exist_ok=True)
        os.makedirs(cfg["ckpt_dir"], exist_ok=True)
        os.makedirs("configs", exist_ok=True)

        # 保存配置
        cfg_path = f"configs/{idx}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # 构造 bsub 命令（假设你有 train_one.py）
        cmd = f'bsub -J "cfg_{idx}" -o {cfg["log_dir"]}/out.log -e {cfg["log_dir"]}/err.log -n 1 -I python train_one.py --config {cfg_path} &'
        print(f"Submitting: {cmd}")
        subprocess.Popen(cmd, shell=True)
