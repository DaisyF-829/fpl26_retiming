# train_one.py
import os
import json
import argparse
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



def train_one_epoch(model, dataloader, optimizer, epoch, max_grad_norm, device, log_file):
    model.train()
    total_loss, total_rec_loss, total_kl = 0, 0, 0

    for batched_graph in dataloader:
        batched_graph = batched_graph.to(device)
        optimizer.zero_grad()
        orig_feat, rec_feat, node_emb, graph_emb = model(batched_graph)
        loss, rec_loss, kl_loss = model.loss_fn(orig_feat, rec_feat, node_emb=node_emb)

        loss.backward()
        if max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_kl += kl_loss.item() if kl_loss is not None else 0

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_rec_loss = total_rec_loss / num_batches
    avg_kl = total_kl / num_batches

    log_file.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}, Rec: {avg_rec_loss:.6f}, KL: {avg_kl:.6f}\n")



def eval_model(model, dataloader, epoch, device, log_file):
    model.eval()
    total_loss, total_rec_loss, total_kl = 0, 0, 0
    with torch.no_grad():
        for batched_graph in dataloader:
            batched_graph = batched_graph.to(device)
            orig_feat, rec_feat, node_emb, graph_emb = model(batched_graph)
            loss, rec_loss, kl_loss = model.loss_fn(orig_feat, rec_feat, node_emb=node_emb)

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kl += kl_loss.item() if kl_loss is not None else 0

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_rec_loss = total_rec_loss / num_batches
    avg_kl = total_kl / num_batches

    log_file.write(f"[Epoch {epoch}] Val   Loss: {avg_loss:.6f}, Rec: {avg_rec_loss:.6f}, KL: {avg_kl:.6f}\n")
    return avg_loss



def save_checkpoint(encoder, decoder, optimizer, save_path, epoch):
    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, save_path)


def main(config):
    device = torch.device(config["device"])
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["ckpt_dir"], exist_ok=True)

    log_path = os.path.join(config["log_dir"], "train.log")
    log_file = open(log_path, "w")

    train_data, val_data = load_dataset_tile_batch(root_dir="/home/wllpro/llwang/yfdai/plgnn/raw_datasets/k6_frac_N10_frac_chain_mem32K_40nm/")

    in_node_dim = 17
    in_edge_dim = 4

    model = build_graph_autoencoder(
        model_type=config["model_type"],
        in_node_dim=in_node_dim,
        in_edge_dim=in_edge_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_jk=config.get("use_jk", False),
        jk_mode=config.get("jk_mode", "cat"),
        kl_weight=config.get("kl_weight", 0.0)
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["lr"])
    scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])

    best_val_loss = math.inf

    if config.get("kl_weight", 0.0) > 0:
        config["checkpoint"] = "/home/wllpro/llwang/yfdai/plgnn/models/ckpt_gae/1/epoch1909_val7.1873.pt"
    else:
        config["checkpoint"] = "/home/wllpro/llwang/yfdai/plgnn/models/ckpt_gae/0/epoch1519_val7.2281.pt"

    if "checkpoint" in config and os.path.exists(config["checkpoint"]):
        checkpoint = torch.load(config["checkpoint"])
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Continue from the next epoch
        best_val_loss = checkpoint.get("best_val_loss", math.inf)
        print(f"Loaded checkpoint from {config['checkpoint']} and continuing from epoch {start_epoch}")
    else:
        start_epoch = 1

    for epoch in tqdm(range(1, config["epochs"] + 1)):
        train_one_epoch(model, train_data, optimizer, epoch, config["max_grad_norm"], device, log_file)
        val_loss = eval_model(model, val_data, epoch, device, log_file)

        if epoch % config["save_freq"] == 0 or val_loss < best_val_loss:
            save_path = join(config["ckpt_dir"], f"epoch{epoch}_val{val_loss:.4f}.pt")
            save_checkpoint(model.encoder, model.decoder, optimizer, save_path, epoch)
            log_file.write(f"[Epoch {epoch}] ✅ Checkpoint saved to {save_path}\n")
            best_val_loss = min(best_val_loss, val_loss)

        if epoch % config["lr_decay_freq"] == 0:
            scheduler.step()
            log_file.write(f"[Epoch {epoch}] 🔻 Learning rate decayed\n")

    log_file.close()

    # 写入结果
    with open("results_all.csv", "a") as f:
        row = [config["id"], best_val_loss]
        f.write(",".join(map(str, row)) + "\n")

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    main(deepcopy(config))
