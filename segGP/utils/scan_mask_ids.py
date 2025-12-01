import os, sys
import numpy as np
from PIL import Image

def scan_mask_ids(mask_dir: str, limit: int | None = None):
    ids = set()
    counts = {}
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg"))]
    if limit:
        files = files[:limit]
    for i, f in enumerate(files, 1):
        p = os.path.join(mask_dir, f)
        try:
            arr = np.array(Image.open(p))
            # if RGB palette mask, convert to indexed if possible (keeps values as class ids in VOC)
            if arr.ndim == 3:
                # heuristic: convert to single channel by taking first channel; adjust if needed
                arr = arr[..., 0]
            vals, freq = np.unique(arr, return_counts=True)
            ids.update(vals.tolist())
            for v, c in zip(vals.tolist(), freq.tolist()):
                counts[v] = counts.get(v, 0) + c
        except Exception as e:
            print(f"[warn] failed {p}: {e}")
    print(f"Found {len(ids)} unique IDs")
    print("IDs:", sorted(ids))
    # top frequencies
    top = sorted(counts.items(), key=lambda x: -x[1])[:20]
    print("Top 20 IDs by pixel count:", top)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan_mask_ids.py /path/to/mask_dir [limit]")
        sys.exit(1)
    mask_dir = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    scan_mask_ids(mask_dir, limit)