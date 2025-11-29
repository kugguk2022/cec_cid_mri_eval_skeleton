import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

def estimate_slope(X, label):
    # Helper to calculate d
    dists = pdist(X, metric='euclidean')
    eps_list = np.logspace(np.log10(0.05), np.log10(0.5), 10)
    log_eps, log_C = [], []
    for eps in eps_list:
        frac = np.sum(dists < eps) / len(dists)
        if frac > 0:
            log_eps.append(np.log(eps))
            log_C.append(np.log(frac))
    
    slope, _ = np.polyfit(log_eps, log_C, 1)
    return slope, log_eps, log_C

def run_cross_validation_test():
    print("\n--- ACID TEST: CROSS-VALIDATION ---")
    
    # 1. Generate TWO separate datasets (Same physics, different samples)
    # Train = 2000 samples, Test = 2000 samples
    N = 2000
    noise_dim = 100
    
    # Common projection matrix (The "Physics" of the world)
    Q = np.random.randn(3, noise_dim) 
    
    print("1. Generating Train Set (A) and Test Set (B)...")
    X_A_clean, _ = make_swiss_roll(N, noise=0.1)
    X_B_clean, _ = make_swiss_roll(N, noise=0.1) # New random samples
    
    X_A = X_A_clean @ Q + 0.05 * np.random.randn(N, noise_dim)
    X_B = X_B_clean @ Q + 0.05 * np.random.randn(N, noise_dim)
    
    # 2. FIT PCA ON A ONLY
    print("2. Fitting PCA on Set A (Training)...")
    pca = PCA(n_components=10)
    pca.fit(X_A) # Learn geometry from A
    
    # 3. TRANSFORM BOTH
    print("3. Projecting Set B using Set A's encoder...")
    # L2 Normalize both
    X_A_emb = pca.transform(X_A)
    X_A_emb /= np.linalg.norm(X_A_emb, axis=1, keepdims=True)
    
    X_B_emb = pca.transform(X_B) # B was never seen during fit!
    X_B_emb /= np.linalg.norm(X_B_emb, axis=1, keepdims=True)
    
    # 4. COMPARE DIMENSIONS
    d_A, x_A, y_A = estimate_slope(X_A_emb, "Train (A)")
    d_B, x_B, y_B = estimate_slope(X_B_emb, "Test (B)")
    
    print(f"\nRESULTS:")
    print(f"Dimension of Train Set A: {d_A:.3f}")
    print(f"Dimension of Test Set B:  {d_B:.3f}")
    
    diff = abs(d_A - d_B)
    if diff < 0.1:
        print(f">>> PASS. Difference {diff:.3f} is within tolerance.")
        print("    The geometry generalizes. It is NOT overfitting.")
    else:
        print(f">>> FAIL. Difference {diff:.3f} indicates overfitting.")

if __name__ == "__main__":
    # run_synthetic_reproduction() # From previous step
    run_cross_validation_test()