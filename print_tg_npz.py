import numpy as np

def print_npz_file(npz_path):
    print(f"\n[INFO] Loading file: {npz_path}")
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"[ERROR] Failed to load npz file: {e}")
        return

    for key in data.files:
        arr = data[key]
        print(f"\n==== {key} ====")
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        
        if key in ("tedge_delay", "tnode_rt_time"):
            print(f"First 1000 values of {key}:")
            flat = arr.ravel()
            print(flat[:1000])
            if flat.size > 1000:
                print("... (truncated)")
        else:
            print(f"Values:\n{arr}")
            if arr.size > 100:
                print(f"Head (first 10 values): {arr.flat[:10]}")
                print("... (truncated)")
    print("\n[INFO] Done.\n")

if __name__ == "__main__":
    print_npz_file("timing_graph.npz")
