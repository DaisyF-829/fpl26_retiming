import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GINEConv, EGATConv


class MLP(nn.Module):
    def __init__(self, in_dim, *hidden_dims, out_dim, dropout=0.0, use_bn=False):

        super().__init__()
        layers = []
        last_dim = in_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = dim

        layers.append(nn.Linear(last_dim, out_dim))  # 最后一层不加激活
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class JumpingKnowledge(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        assert mode in ['last', 'mean', 'concat'], f"Unsupported JK mode: {mode}"
        self.mode = mode

    def forward(self, xs):
        if self.mode == 'last':
            return xs[-1]
        elif self.mode == 'mean':
            return torch.stack(xs, dim=0).mean(dim=0)
        elif self.mode == 'concat':
            return torch.cat(xs, dim=1)


class AllConv(nn.Module):
    def __init__(self, node_dim, edge_dim, out_nf, dropout):
        super().__init__()
        self.MLP_msg = MLP(node_dim * 2 + edge_dim, 64, 64, 64, out_dim = out_nf, dropout=dropout)

    def edge_udf(self, edges):
        x = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1)
        return {"msg": self.MLP_msg(x)}

    def forward(self, g):
        with g.local_scope():
            g.apply_edges(self.edge_udf)
            g.update_all(dgl.function.copy_e('msg', 'm'), dgl.function.mean('m', 'nf_out'))
            return g.ndata['nf_out']


class DeepGCNIIDecoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, num_layers, dropout, use_jk=False, jk_mode='last'):
        super().__init__()
        self.use_jk = use_jk
        self.jk = JumpingKnowledge(jk_mode) if use_jk else None

        self.layer0 = AllConv(in_dim, edge_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList([
            AllConv(in_dim + hidden_dim, edge_dim, hidden_dim, dropout)
            for _ in range(num_layers - 2)
        ])
        self.layern = AllConv(hidden_dim, edge_dim, out_dim if not use_jk else hidden_dim, dropout)

    def forward(self, g):
        g.ndata['feat'] = g.ndata['h']
        x_list = []
        x = self.layer0(g)
        if self.use_jk:
            x_list.append(x)
        for layer in self.layers:
            g.ndata['feat'] = torch.cat([x, g.ndata['feat']], dim=1)
            x = layer(g) + x
            if self.use_jk:
                x_list.append(x)
        g.ndata['feat'] = x
        x = self.layern(g)
        if self.use_jk:
            x_list.append(x)
            x = self.jk(x_list)
        g.ndata['h'] = x
        return x


class GCNDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, use_jk=False, jk_mode='last'):
        super().__init__()
        self.use_jk = use_jk
        self.jk = JumpingKnowledge(jk_mode) if use_jk else None
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
        self.layers.append(GraphConv(hidden_dim, out_dim if not use_jk else hidden_dim))

    def forward(self, g):
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = g.ndata['h']
        x = g.ndata['feat']
        x_list = []
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
            if self.use_jk:
                x_list.append(x)
        if self.use_jk:
            x = self.jk(x_list)
        g.ndata['h'] = x
        return x


class EGATDecoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, num_layers, dropout, num_heads=4, use_jk=False, jk_mode='last'):
        super().__init__()
        self.use_jk = use_jk
        self.jk = JumpingKnowledge(jk_mode) if use_jk else None
        self.dropout = dropout
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        self.layers.append(EGATConv(in_dim, edge_dim, hidden_dim, hidden_dim, num_heads))
        for _ in range(num_layers - 2):
            self.layers.append(EGATConv(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads))
        self.layers.append(EGATConv(hidden_dim, hidden_dim, out_dim if not use_jk else hidden_dim, hidden_dim, num_heads))

    def forward(self, g):
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = g.ndata['h']
        x = g.ndata['feat']
        e = g.edata['feat']
        x_list = []
        for layer in self.layers:
            x, e = layer(g, x, e)
            x = x.mean(dim=1)
            e = e.mean(dim=1)
            x = F.elu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.use_jk:
                x_list.append(x)
        if self.use_jk:
            x = self.jk(x_list)
        g.ndata['h'] = x
        return x


# class GINEDecoder(nn.Module):
#     def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, num_layers, dropout, use_jk=False, jk_mode='last'):
#         super().__init__()
#         self.use_jk = use_jk
#         self.jk = JumpingKnowledge(jk_mode) if use_jk else None

#         self.edge_encoder = MLP(edge_dim, hidden_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout)
#         self.node_proj = MLP(in_dim, hidden_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout)
#         self.gine_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.gine_layers.append(
#                 GINEConv(MLP(hidden_dim, hidden_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout))
#             )
#         self.out_proj = MLP(hidden_dim, hidden_dim, hidden_dim, out_dim=out_dim, dropout=dropout)

#     def forward(self, g):
#         g.ndata['feat'] = g.ndata['h']
#         x = g.ndata['feat']
#         edge_feat = self.edge_encoder(g.edata['feat'])
#         x = self.node_proj(x)
#         x_list = []
#         for conv in self.gine_layers:
#             x = conv(g, x, edge_feat)
#             if self.use_jk:
#                 x_list.append(x)
#         x = self.out_proj(x)
#         if self.use_jk:
#             x = self.jk(x_list)
#         g.ndata['h'] = x
#         return x

class GINEDecoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, num_layers, dropout, use_jk=False, jk_mode='last'):
        super().__init__()
        self.use_jk = use_jk
        self.jk = JumpingKnowledge(jk_mode) if use_jk else None

        # Edge feature encoder (MLP for edge features)
        self.edge_encoder = MLP(edge_dim, hidden_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout)
        
        # Node feature projection (MLP for node features)
        self.node_proj = MLP(in_dim, hidden_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout)
        
        # GINE layers
        self.gine_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gine_layers.append(
                GINEConv(MLP(hidden_dim, hidden_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout))
            )
        
        # Output projection layer
        self.out_proj = MLP(hidden_dim, hidden_dim, hidden_dim, out_dim=out_dim, dropout=dropout)

    def forward(self, g):
        g.ndata['feat'] = g.ndata['h']
        # Retrieve node features and edge features
        x = g.ndata['feat']  # Node features
        edge_feat = self.edge_encoder(g.edata['feat'])  # Edge features
        
        # Project node features
        x = self.node_proj(x)

        # List to store intermediate results for Jumping Knowledge
        x_list = []

        # GINE layers: Apply GINEConv with node and edge features
        for conv in self.gine_layers:
            x = conv(g, x, edge_feat)
            if self.use_jk:
                x_list.append(x)
        
        # Output projection
        x = self.out_proj(x)

        # If Jumping Knowledge is used, apply it
        if self.use_jk:
            x = self.jk(x_list)
        
        # Store node embeddings in graph
        g.ndata['h'] = x

        return x
