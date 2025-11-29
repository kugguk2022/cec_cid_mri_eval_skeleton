import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import os

# --- CONFIGURATION ---
INPUT_FILE = "outputs/brats_embeddings/embeddings.npy"
EPS_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # Expanded list to see the curve

def main():
    print(f"Loading embeddings from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    # Load data
    X = np.load(INPUT_FILE)
    N = X.shape[0]
    print(f"Loaded {N} samples.")

    # --- SUBSAMPLING STRATEGY ---
    # Calculating 30k^2 distances requires ~3.5GB RAM. 
    # If you have <8GB RAM, we subsample to 10k points.
    if N > 15000:
        print("Subsampling to 15,000 points for speed/memory...")
        indices = np.random.choice(N, 15000, replace=False)
        X = X[indices]
    
    print("Calculating pairwise distances (this may take a minute)...")
    # pdist returns a condensed distance matrix (1D array of all pairs)
    dists = pdist(X, metric='euclidean')
    
    # --- CORRELATION SUM C(eps) ---
    print("\nComputing Correlation Sum C(eps)...")
    log_eps = []
    log_C = []
    
    for eps in EPS_LIST:
        # Count how many pairs are closer than epsilon
        count = np.sum(dists < eps)
        fraction = count / len(dists)
        
        print(f"eps={eps:.2f}: {fraction*100:.2f}% of pairs connected")
        
        if fraction > 0:
            log_eps.append(np.log(eps))
            log_C.append(np.log(fraction))

    # --- CALCULATE DIMENSION (SLOPE) ---
    if len(log_eps) > 1:
        # Fit a line to the log-log data
        slope, intercept = np.polyfit(log_eps, log_C, 1)
        print(f"\nEstimated Intrinsic Dimension (d): {slope:.3f}")
        
        # --- PLOTTING ---
        plt.figure(figsize=(8, 6))
        plt.scatter(log_eps, log_C, color='red', label='Observed Data')
        plt.plot(log_eps, slope * np.array(log_eps) + intercept, 'k--', 
                 label=f'Fit (slope={slope:.2f})')
        
        plt.xlabel('log(epsilon)')
        plt.ylabel('log( Correlation Sum )')
        plt.title(f'BraTS Manifold Dimension\nEstimated d = {slope:.2f}')
        plt.legend()
        plt.grid(True)
        plt.savefig("brats_dimension_plot.png")
        print("Plot saved to brats_dimension_plot.png")
    else:
        print("Not enough data points to estimate dimension.")

if __name__ == "__main__":
    main()