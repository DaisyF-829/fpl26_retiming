import os
import torch
import dgl
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor

def load_tile_graph(fname, tile_folder):
    tile_idx = int(fname.split("_")[1].split(".")[0])
    path = os.path.join(tile_folder, fname)
    data = np.load(path)

    if 'global_features' not in data:
        print(f"  Skipped {fname}: missing global_features")
        return None  # Return None to skip this file

    # 1. Parse global features
    global_feats = torch.tensor(data['global_features'], dtype=torch.float32)

    # 2. Parse node features if available
    if 'rrnode_features' in data:
        node_feats = torch.tensor(data['rrnode_features'], dtype=torch.float32)
        assert node_feats.numel() % 13 == 0, f"node_feats size not divisible by 13 in {fname}"
        num_nodes = node_feats.numel() // 13
        node_feats = node_feats.view(num_nodes, 13)
    else:
        num_nodes = 1
        node_feats = torch.zeros((1, 13))  # dummy node with zero features

    # 3. Parse edges if available
    if 'rredge_src' in data and 'rredge_dst' in data and 'rredge_feats' in data:
        src = torch.tensor(data['rredge_src'], dtype=torch.int64)
        dst = torch.tensor(data['rredge_dst'], dtype=torch.int64)
        edge_feats = torch.tensor(data['rredge_feats'], dtype=torch.float32)
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['feat'] = edge_feats.view(-1, 4)
    else:
        g = dgl.graph(([], []), num_nodes=num_nodes)

    g.ndata['feat'] = node_feats
    g.ndata['global'] = global_feats.expand(num_nodes, -1)

    return tile_idx, g

def load_tile_graphs_parallel(tile_folder):
    tile_graphs = {}
    with ThreadPoolExecutor() as executor:
        # Collect all the tile files, sorted by numerical tile index
        tile_files = sorted(
            [fname for fname in os.listdir(tile_folder) if fname.startswith("tile_") and fname.endswith(".npz")],
            key=lambda fname: int(fname.split("_")[1].split(".")[0])  # Sort by numeric part of filename
        )
        
        # Use ThreadPoolExecutor to load tile graphs in parallel
        results = executor.map(lambda fname: load_tile_graph(fname, tile_folder), tile_files)
        
        # Collect the results and store in a dictionary
        for result in results:
            if result is not None:
                tile_idx, g = result
                tile_graphs[tile_idx] = g

    return tile_graphs

if __name__ == "__main__":
    folder = "tiles"
    tile_graphs = load_tile_graphs_parallel(folder)

    print(f"Loaded {len(tile_graphs)} tile graphs")

    # Save the loaded graphs as a pickle file
    with open("tiles.pkl", "wb") as f:
        pickle.dump(tile_graphs, f)

    # Print first 10 graphs
    for i, (tile_idx, g) in enumerate(tile_graphs.items()):
        print(f"Tile {tile_idx}:")
        
        # Print number of nodes
        print(f"  Num nodes: {g.num_nodes()}")
        
        # Print node features if they exist
        if 'feat' in g.ndata:
            print(f"  Node feature shape: {g.ndata['feat'].shape}")
            print(f"  Node features (first 5): \n{g.ndata['feat'][:5]}")
        else:
            print("  No node features available.")
        
        # Print edge features if they exist
        if 'feat' in g.edata:
            print(f"  Edge feature shape: {g.edata['feat'].shape}")
            print(f"  Edge features (first 5): \n{g.edata['feat'][:5]}")
        else:
            print("  No edge features available.")
        
        # Print global features
        print(f"  Global feature shape (broadcasted): {g.ndata['global'].shape}")
        print(f"  Global features (first 5): \n{g.ndata['global'][:5]}")
        
        print()  # Add space between each tile

        # Stop after printing the first 10 graphs
        if i >= 9:
            break
