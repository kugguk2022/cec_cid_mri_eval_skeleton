import os
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- CONFIGURATION ---
BIDS_DIR = "datasets/BIDS_BRATS"  # Update this to your path
OUT_DIR = "outputs/brats_embeddings"
TARGET_MODALITY = "T1w"  # or "T1wCE", "FLAIR", "T2w"
SLICE_AXIS = 2  # 0=Sagittal, 1=Coronal, 2=Axial
BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)
DEVICE = torch.device("cpu")  # default; can be overridden in main

class SimpleConvEncoder(nn.Module):
    """
    Lightweight fallback if torchvision is unavailable.
    Produces 512-dim embeddings from a single-channel image.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 512),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

# --- FIX 2: THE ENCODER ---
def get_feature_extractor():
    """
    Uses ResNet18. Replaces the first layer to accept 1 channel (Grayscale)
    and removes the final classification layer to return 512-dim embeddings.
    """
    try:
        from torchvision import models

        # Handle both new (weights enum) and old (pretrained=True) APIs.
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=True)

        # Fix input to accept 1 channel (MRI) instead of 3 (RGB)
        original_weights = model.conv1.weight.data
        new_weights = torch.sum(original_weights, dim=1, keepdim=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = new_weights

        # Remove the final classification layer (fc) to get raw features
        model.fc = nn.Identity()
        encoder = model
    except Exception as exc:
        print(f"torchvision not available ({exc}); using lightweight CNN encoder.")
        encoder = SimpleConvEncoder()

    return encoder.to(DEVICE).eval()


def extract_slice(data_proxy, slice_idx, axis=None):
    """
    Grab a single slice along SLICE_AXIS.
    For 4D volumes, we take the first channel/time index unless the slice axis is the last dim.
    """
    axis = SLICE_AXIS if axis is None else axis
    shape = data_proxy.shape
    slicer = [slice(None)] * len(shape)
    slicer[axis] = slice_idx
    if len(shape) == 4 and axis != len(shape) - 1:
        slicer[-1] = 0  # take first channel / timepoint
    return np.asanyarray(data_proxy[tuple(slicer)])

# --- DATASET CLASS ---
class BrainSliceDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.target_size = (224, 224)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nifti_path, slice_idx = self.file_paths[idx]
        
        # Load the specific slice (optimization: ideally load volume once, but this is safer for memory)
        data_proxy = nib.load(nifti_path).dataobj
        slice_data = extract_slice(data_proxy, slice_idx)

        # --- FIX 1: ROBUST NORMALIZATION ---
        # 1. Handle NaNs
        slice_data = np.nan_to_num(slice_data)
        
        # 2. Percentile scaling (0 to 1 based on robust range)
        p1, p99 = np.percentile(slice_data, [1, 99])
        if p99 > p1:
            slice_data = (slice_data - p1) / (p99 - p1)
        
        # 3. Clip and cast
        slice_data = np.clip(slice_data, 0, 1).astype(np.float32)

        # Transform to Tensor and resize without torchvision
        slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        slice_tensor = F.interpolate(slice_tensor, size=self.target_size, mode="bilinear", align_corners=False)
        slice_tensor = slice_tensor.squeeze(0)  # (1, H, W)

        return slice_tensor, str(nifti_path), int(slice_idx)

def scan_dataset(bids_root, target_modality=None, slice_axis=None, stride=2, modality_aliases=None):
    """
    Scans BIDS dir for valid slices.
    --- FIX 3: FILTER EMPTY SLICES ---
    """
    valid_slices = []
    axis = SLICE_AXIS if slice_axis is None else slice_axis
    modality = target_modality or TARGET_MODALITY
    aliases = modality_aliases or []
    # If the modality ends with "w" (T1w/T2w), also accept the non-w suffix which is common in BRATS.
    if modality.endswith("w"):
        aliases.append(modality[:-1])
    search_modalities = [modality] + aliases

    print(f"Scanning {bids_root} for {', '.join(search_modalities)}...")

    # Recursive search for NIfTI files (.nii or .nii.gz)
    files = []
    for mod in search_modalities:
        pattern = os.path.join(bids_root, "**", f"*{mod}*.nii*")
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))

    for fpath in tqdm(files):
        try:
            # Quick load header/proxy to get shape
            img = nib.load(fpath)
            depth = img.shape[axis]
            data_proxy = img.dataobj # delayed load
            
            # Subsample slices to save time? (Optional: step=2)
            for i in range(0, depth, stride): 
                # Heuristic: check center pixel or quick stat to avoid loading full array if possible
                # For accuracy, we load the slice:
                s = extract_slice(data_proxy, i, axis=axis)
                
                # THRESHOLD: Skip if brain covers less than 5% of image
                mask = s > (np.mean(s) * 0.5) # simple background threshold
                if np.sum(mask) / s.size > 0.05:
                    valid_slices.append((fpath, i))
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return valid_slices

def parse_args():
    parser = argparse.ArgumentParser(description="Generate slice-level embeddings for BRATS BIDS data.")
    parser.add_argument("--bids_dir", default=BIDS_DIR, help="Root of BIDS-formatted BRATS data.")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Directory to write embeddings and metadata.")
    parser.add_argument("--modality", default=TARGET_MODALITY, help="Modality substring to match (e.g., T1w, FLAIR, T2).")
    parser.add_argument("--slice_axis", type=int, default=SLICE_AXIS, choices=[0,1,2,3], help="Axis to slice along.")
    parser.add_argument("--stride", type=int, default=2, help="Slice stride; 1 keeps every slice.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection. 'auto' picks CUDA if compatible, else CPU.")
    return parser.parse_args()

def select_device(choice: str) -> torch.device:
    """
    Choose device with compatibility guard for GPUs that PyTorch cannot target.
    If the detected compute capability is higher than the build supports, fall back to CPU.
    """
    if choice == "cpu":
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            return torch.device("cpu")

    # Guard against GPUs with compute capability newer than this build supports.
    try:
        major, minor = torch.cuda.get_device_capability()
        cc = major * 10 + minor
        # Current PyTorch wheels typically support up to sm_90; anything above is likely unsupported.
        if cc > 90:
            print(f"Detected GPU compute capability sm_{cc}, unsupported by this PyTorch build; using CPU.")
            return torch.device("cpu")
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Could not query CUDA capability ({exc}); using CPU.")
        return torch.device("cpu")

    return torch.device("cuda")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    args = parse_args()

    # Allow CLI override of globals used by helper functions.
    SLICE_AXIS = args.slice_axis
    DEVICE = select_device(args.device)
    print(f"Using device: {DEVICE}")

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Identify valid slices
    slice_list = scan_dataset(args.bids_dir, target_modality=args.modality, slice_axis=args.slice_axis, stride=args.stride)
    print(f"Found {len(slice_list)} valid slices containing brain tissue.")
    
    if len(slice_list) == 0:
        print("No files found. Check BIDS_DIR path or modality string.")
        exit()

    # 2. Setup Loader
    dataset = BrainSliceDataset(slice_list)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 3. Setup Model
    model = get_feature_extractor()
    
    embeddings = []
    metadata = []
    
    print("Generating Embeddings...")
    with torch.no_grad():
        for imgs, paths, idxs in tqdm(loader):
            imgs = imgs.to(DEVICE)
            
            # Forward pass -> (Batch, 512)
            feats = model(imgs)
            
            # --- CRITICAL FIX: L2 NORMALIZATION ---
            # This forces all vectors to lie on the unit hypersphere.
            # Max Euclidean distance becomes 2.0.
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            
            embeddings.append(feats.cpu().numpy())
            
            # Save metadata
            for p, i in zip(paths, idxs):
                metadata.append({'filename': p, 'slice_idx': i.item()})
    
    # 4. Concatenate and Save
    full_embeddings = np.concatenate(embeddings, axis=0) # Shape (N, 512)
    df_meta = pd.DataFrame(metadata)
    
    # Save for CEC
    np.save(os.path.join(args.out_dir, "embeddings.npy"), full_embeddings)
    df_meta.to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    
    print("-" * 30)
    print(f"Saved {full_embeddings.shape[0]} embeddings to {args.out_dir}")
    
    # --- SANITY CHECK FOR USER ---
    # Calculate distance between first two distinct images to prove it's not 0 or 10000
    if len(full_embeddings) > 1:
        dist = np.linalg.norm(full_embeddings[0] - full_embeddings[1])
        print(f"DEBUG: Euclidean distance between first two slices: {dist:.4f}")
        print("Note: If this is > 0 and < 2.0, the geometry is valid.")
