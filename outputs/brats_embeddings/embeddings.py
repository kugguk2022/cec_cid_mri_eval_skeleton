import os
import glob
import struct
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage

# --- 1. Minimal NIfTI Reader (No external dependencies) ---
def read_nii_minimal(filepath):
    try:
        with open(filepath, 'rb') as f:
            try:
                f.seek(40)
                dims = struct.unpack('<8h', f.read(16))
                f.seek(70)
                datatype = struct.unpack('<h', f.read(2))[0]
                f.seek(108)
                vox_offset = struct.unpack('<f', f.read(4))[0]
                ndim = dims[0]
                shape = dims[1:ndim+1]
                dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}
                if datatype not in dtype_map: 
                    print(f"Error parsing header in {filepath}: unsupported datatype {datatype}")
                    return None
                np_dtype = dtype_map[datatype]
                num_items = np.prod(shape)
                f.seek(int(vox_offset))
                arr = np.frombuffer(f.read(), dtype=np_dtype, count=num_items)
                return arr.reshape(shape, order='F')
            except struct.error as e:
                print(f"Header parsing error in {filepath}: {e}")
                return None
            except Exception as e:
                print(f"Data reading/parsing error in {filepath}: {e}")
                return None
    except FileNotFoundError as e:
        print(f"File not found: {filepath}: {e}")
        return None
    except OSError as e:
        print(f"File access error for {filepath}: {e}")
        return None

    for filter_idx in range(N_FILTERS):
        resp = ndimage.convolve(small, filters[filter_idx], mode='constant')
# It acts as a shift-invariant random projection (Johnson-Lindenstrauss).
np.random.seed(42)
N_FILTERS = 64
filters = np.random.randn(N_FILTERS, 5, 5) # 64 filters of size 5x5

def encode_slice(slice_data):
    # Downsample to speed up convolution (approx 60x60)
    small = slice_data[::4, ::4] 
    features = []
    # Convolution + ReLU + Global Max Pool
    for k in range(N_FILTERS):
        resp = ndimage.convolve(small, filters[k], mode='constant')
        feat = np.max(np.maximum(0, resp)) # ReLU + MaxPool
        features.append(feat)
    return np.array(features)

def process_bids(bids_dir, out_file):
    print(f"Scanning {bids_dir}...")
    files = glob.glob(os.path.join(bids_dir, "**", "*.nii*"), recursive=True)
    # Filter for standard modalities
    files = [f for f in files if any(m in f for m in ['T1w', 'T2w', 'FLAIR'])]
    
    embeddings = []
    metadata = []
    
    print(f"Found {len(files)} MRI volumes. Processing...")
    
    for fpath in files:
        data = read_nii_minimal(fpath)
        if data is None: continue
        
        # BraTS usually (H, W, D). Iterate Axial (D)
        n_slices = data.shape[2]
        modality = fpath.split('_')[-1].split('.')[0]
        
        for i in range(0, n_slices, 2): # Stride 2
            slice_data = data[:, :, i]
            
            # --- FIX 3: Filter Empty ---
            if np.count_nonzero(slice_data) / slice_data.size < 0.05: continue
                
            # --- FIX 1: Robust Normalization ---
            p1, p99 = np.percentile(slice_data, [1, 99])
            if p99 - p1 == 0: continue
            img_norm = (slice_data - p1) / (p99 - p1)
            img_norm = np.clip(img_norm, 0, 1)
            
            # --- FIX 2: CNN Embedding ---
            vec = encode_slice(img_norm)
            
            # L2 Normalize
            norm = np.linalg.norm(vec)
            if norm > 0: vec = vec / norm
                
            embeddings.append(vec)
            metadata.append({'filename': fpath, 'slice': i, 'modality': modality})
            
    if embeddings:
        emb_arr = np.array(embeddings)
        df = pd.concat([pd.DataFrame(metadata), pd.DataFrame(emb_arr)], axis=1)
        df.to_csv(out_file, index=False)
        print(f"Saved {len(embeddings)} embeddings to {out_file}")
    else:
        print("No valid slices found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--out", default="embeddings.csv")
    args = parser.parse_args()
    process_bids(args.bids_dir, args.out)
