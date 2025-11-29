import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

def run_synthetic_reproduction():
    print("--- CEC-CID REPRODUCTION: SYNTHETIC MANIFOLD ---")
    
    # 1. GENERATE GROUND TRUTH (Swiss Roll, d=2)
    # Embedded in high-dim space (R^100) to test robustness
    print("1. Generating Swiss Roll in R^100 (True Dim=2)...")
    n_samples = 2000
    X_clean, _ = make_swiss_roll(n_samples, noise=0.1)
    
    # Embed in noise (R^100)
    noise_dim = 100
    Q = np.random.randn(3, noise_dim) # Random projection matrix
    X_highdim = X_clean @ Q 
    # Add ambient noise
    X_highdim += 0.05 * np.random.randn(n_samples, noise_dim)
    
    # 2. GENERATOR PIPELINE (The "Budget" Process)
    # We simulate a generator improving by adding PCA components
    print("2. Simulating Generative Budget (PCA components)...")
    
    pca = PCA(n_components=10) # Budget = 10 components
    X_latent = pca.fit_transform(X_highdim)
    
    # L2 Normalize (Geometric Hygiene)
    norms = np.linalg.norm(X_latent, axis=1, keepdims=True)
    X_embedded = X_latent / (norms + 1e-9)
    
    # 3. MEASURE DIMENSION (The CEC-CID Signature)
    print("3. Calculating Correlation Sum C(eps)...")
    dists = pdist(X_embedded, metric='euclidean')
    
    eps_list = np.logspace(np.log10(0.05), np.log10(0.5), 10)
    log_eps = []
    log_C = []
    
    for eps in eps_list:
        count = np.sum(dists < eps)
        frac = count / len(dists)
        if frac > 0:
            log_eps.append(np.log(eps))
            log_C.append(np.log(frac))
            print(f"   eps={eps:.3f} -> C(eps)={frac:.4f}")

    # 4. FIT SLOPE
    slope, intercept = np.polyfit(log_eps, log_C, 1)
    print(f"\n>>> ESTIMATED DIMENSION d = {slope:.3f} (Expected ~2.0)")
    
    # 5. PLOT
    plt.figure(figsize=(6,6))
    plt.scatter(log_eps, log_C, c='r', label='Observed')
    plt.plot(log_eps, slope*np.array(log_eps) + intercept, 'k--', label=f'Fit d={slope:.2f}')
    plt.xlabel('log(epsilon)')
    plt.ylabel('log(Correlation Sum)')
    plt.title(f'Synthetic Reproduction (Swiss Roll)\nTrue d=2.0 | Est d={slope:.2f}')
    plt.legend()
    plt.grid(True)
    plt.savefig('synthetic_dimension.png')
    print("Plot saved to synthetic_dimension.png")

if __name__ == "__main__":
    run_synthetic_reproduction()