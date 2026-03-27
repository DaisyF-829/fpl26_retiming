
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_encoder import (
    DeepGCNIIEncoder, GCNEncoder, EGATEncoder, GINEEncoder
)
from model_decoder import (
    DeepGCNIIDecoder, GCNDecoder, EGATDecoder, GINEDecoder
)


def build_symmetric_decoder(
    model_type: str,
    out_node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    in_node_dim: int,
    num_layers: int,
    dropout: float,
    use_jk: bool = False,
    jk_mode: str = 'last'
):
    if model_type == 'deepgcn':
        return DeepGCNIIDecoder(
            out_node_dim, edge_dim, hidden_dim, in_node_dim,
            num_layers, dropout, use_jk, jk_mode
        )
    elif model_type == 'gcn':
        return GCNDecoder(
            out_node_dim, hidden_dim, in_node_dim,
            num_layers, dropout, use_jk, jk_mode
        )
    elif model_type == 'gat':
        return EGATDecoder(
            out_node_dim, edge_dim, hidden_dim, in_node_dim,
            num_layers, dropout, use_jk, jk_mode
        )
    elif model_type == 'gin':
        return GINEDecoder(
            out_node_dim, edge_dim, hidden_dim, in_node_dim,
            num_layers, dropout, use_jk, jk_mode
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, kl_weight=0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def forward(self, g):
        x_orig = g.ndata["feat"]  # 原始特征
        node_emb, graph_emb = self.encoder(g)
        g.ndata["h"] = node_emb   # 给 decoder 使用
        x_rec = self.decoder(g)
        return x_orig, x_rec, node_emb, graph_emb
    
    def loss_fn(self, x_orig, x_rec, valid_mask=None, node_emb=None):
        rec_loss = F.mse_loss(x_rec, x_orig, reduction='none')
        if valid_mask is not None:
            rec_loss = (rec_loss * valid_mask.unsqueeze(-1)).mean()
        else:
            rec_loss = rec_loss.mean()

        kl_loss = None
        if self.kl_weight > 0 and node_emb is not None:
            mu = node_emb.mean(dim=0)
            var = node_emb.var(dim=0)
            kl_loss = (var + mu ** 2 - 1 - torch.log(var + 1e-9)).mean()
            total_loss = rec_loss + self.kl_weight * kl_loss
        else:
            total_loss = rec_loss

        return total_loss, rec_loss, kl_loss


def build_graph_autoencoder(
    model_type: str,
    in_node_dim: int,
    in_edge_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_layers: int,
    dropout: float,
    use_jk: bool = False,
    jk_mode: str = 'last',
    kl_weight: float = 0.0
) -> nn.Module:
    # encoder 输入: 原始节点维度
    encoder_in_dim = in_node_dim
    encoder = None
    if model_type == 'deepgcn':
        encoder = DeepGCNIIEncoder(encoder_in_dim, in_edge_dim, hidden_dim, latent_dim, num_layers, dropout, use_jk, jk_mode)
    elif model_type == 'gcn':
        encoder = GCNEncoder(encoder_in_dim, hidden_dim, latent_dim, num_layers, dropout, use_jk, jk_mode)
    elif model_type == 'gat':
        encoder = EGATEncoder(encoder_in_dim, in_edge_dim, hidden_dim, latent_dim, num_layers, dropout, use_jk, jk_mode)
    elif model_type == 'gin':
        encoder = GINEEncoder(encoder_in_dim, in_edge_dim, hidden_dim, latent_dim, num_layers, dropout, use_jk, jk_mode)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    decoder = build_symmetric_decoder(
        model_type=model_type,
        out_node_dim=latent_dim,
        edge_dim=in_edge_dim,
        hidden_dim=hidden_dim,
        in_node_dim=in_node_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_jk=use_jk,
        jk_mode=jk_mode
    )

    return GraphAutoEncoder(encoder, decoder, kl_weight)
