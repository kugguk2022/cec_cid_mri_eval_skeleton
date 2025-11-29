import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

# --- CONFIG ---
INPUT_FILE = "outputs/brats_embeddings/embeddings.npy"
EPS_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def main():
    print(f"Loading {INPUT_FILE}...")
    X = np.load(INPUT_FILE)
    
    # 1. ANALYZE SPECTRUM
    print("Running PCA to find true manifold dimensionality...")
    pca = PCA(n_components=32)
    X_pca = pca.fit_transform(X)
    
    # Plot Variance Explained
    var_exp = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 4))
    
    # Subplot 1: PCA Elbow
    plt.subplot(1, 2, 1)
    plt.plot(var_exp, 'o-')
    plt.axhline(0.95, color='r', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.title('PCA Spectrum (The "True" Dim)')
    plt.legend()
    plt.grid(True)
    
    # Find elbow (95% variance)
    n_components_95 = np.argmax(var_exp >= 0.95) + 1
    print(f" > 95% of data variance is contained in just {n_components_95} dimensions.")
    
    # 2. RE-CALCULATE DIMENSION ON CLEANED DATA
    # We keep only the significant components to remove "random noise" dims
    X_clean = X_pca[:, :n_components_95]
    
    # Renormalize (Geometry requirement: max dist = 2.0)
    # L2 normalize the PCA-reduced vectors
    norms = np.linalg.norm(X_clean, axis=1, keepdims=True)
    X_clean = X_clean / (norms + 1e-9)
    
    print(f"\nRe-estimating Intrinsic Dimension on {n_components_95}D Cleaned Manifold...")
    
    # Subsample if large
    if len(X_clean) > 5000:
        idx = np.random.choice(len(X_clean), 5000, replace=False)
        X_clean = X_clean[idx]

    dists = pdist(X_clean, metric='euclidean')
    
    log_eps = []
    log_C = []
    
    print("\n   Epsilon | Connected % | log(C)")
    print("   " + "-"*30)
    
    for eps in EPS_LIST:
        fraction = np.sum(dists < eps) / len(dists)
        if fraction > 0:
            print(f"   {eps:.2f}    | {fraction*100:6.2f}%    | {np.log(fraction):.2f}")
            log_eps.append(np.log(eps))
            log_C.append(np.log(fraction))
            
    # Fit Slope
    if len(log_eps) >= 2:
        slope, intercept = np.polyfit(log_eps, log_C, 1)
        print(f"\n>>> NEW ESTIMATED DIMENSION (d): {slope:.3f}")
        
        # Subplot 2: Dimension Slope
        plt.subplot(1, 2, 2)
        plt.scatter(log_eps, log_C, color='red')
        plt.plot(log_eps, slope * np.array(log_eps) + intercept, 'k--', label=f'd={slope:.2f}')
        plt.xlabel('log(eps)')
        plt.ylabel('log(C)')
        plt.title(f'Intrinsic Dimension (Cleaned)\nd = {slope:.2f}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("brats_pca_dimension.png")
    print("Saved analysis to brats_pca_dimension.png")

if __name__ == "__main__":
    main()