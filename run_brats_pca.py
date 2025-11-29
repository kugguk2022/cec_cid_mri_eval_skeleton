import os
import glob
import struct
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# --- MINIMAL NIFTI READER ---
def read_nii_minimal(filepath):
    try:
        with open(filepath, 'rb') as f:
            f.seek(40)
            dims = struct.unpack('<8h', f.read(16))
            f.seek(70)
            datatype = struct.unpack('<h', f.read(2))[0]
            f.seek(108)
            vox_offset = struct.unpack('<f', f.read(4))[0]
            ndim = dims[0]
            shape = dims[1:ndim+1]
            dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}
            if datatype not in dtype_map: return None
            np_dtype = dtype_map[datatype]
            num_items = np.prod(shape)
            f.seek(int(vox_offset))
            arr = np.frombuffer(f.read(), dtype=np_dtype, count=num_items)
            return arr.reshape(shape, order='F')
    except:
        return None

def process_dataset(bids_dir, out_file):
    print(f"Scanning {bids_dir}...")
    # Find all relevant NIfTI files
    files = glob.glob(os.path.join(bids_dir, "**", "*T1w*.nii*"), recursive=True) + \
            glob.glob(os.path.join(bids_dir, "**", "*FLAIR*.nii*"), recursive=True) + \
            glob.glob(os.path.join(bids_dir, "**", "*T2w*.nii*"), recursive=True)
            
    # Remove duplicates
    files = sorted(list(set(files)))
    print(f"Found {len(files)} files.")

    vectors = []
    metadata = []

    # 1. EXTRACT RAW VECTORS
    for fpath in files:
        data = read_nii_minimal(fpath)
        if data is None: continue
        
        n_slices = data.shape[2]
        modality = fpath.split('_')[-1].split('.')[0]
        
        # Iterate slices (stride 2)
        for i in range(0, n_slices, 2):
            slice_data = data[:, :, i]
            
            # Filter empty (<5% brain)
            if np.count_nonzero(slice_data) / slice_data.size < 0.05: continue
                
            # Robust Normalize
            p1, p99 = np.percentile(slice_data, [1, 99])
            if p99 - p1 == 0: continue
            img = (slice_data - p1) / (p99 - p1)
            img = np.clip(img, 0, 1)
            
            # Downsample (Stride 8 -> ~30x30 image)
            # This is "Patch Embedding" on a global scale
            vec = img[::8, ::8].flatten()
            
            vectors.append(vec)
            metadata.append({'filename': fpath, 'slice': i, 'modality': modality})
            
    print(f"Extracted {len(vectors)} valid slices.")
    
    # 2. PCA EMBEDDING
    # Convert to matrix
    X = np.array(vectors)
    
    print("Running PCA (32 dims)...")
    pca = PCA(n_components=32)
    X_pca = pca.fit_transform(X)
    
    # 3. L2 NORMALIZE (Critical for geometry)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_emb = X_pca / (norms + 1e-9)
    
    # 4. SAVE
    df_meta = pd.DataFrame(metadata)
    df_emb = pd.DataFrame(X_emb, columns=[f"dim_{k}" for k in range(32)])
    final_df = pd.concat([df_meta, df_emb], axis=1)
    
    final_df.to_csv(out_file, index=False)
    print(f"Saved embeddings to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_dir", required=True, help="Path to BIDS dataset")
    parser.add_argument("--out", default="brats_pca_embeddings.csv", help="Output CSV")
    args = parser.parse_args()
    
    process_dataset(args.bids_dir, args.out)