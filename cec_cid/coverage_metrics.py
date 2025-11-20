
# Covering numbers via k-center greedy + simple diversity metrics
import numpy as np

def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    sq = np.expand_dims(np.diag(G), 1) - 2*G + np.expand_dims(np.diag(G), 0)
    sq[sq < 0] = 0
    return np.sqrt(sq, dtype=np.float64)

def k_center_cover_count(X: np.ndarray, eps: float) -> int:
    """Approximate covering number N(A, eps) by greedy k-center selection."""
    n = X.shape[0]
    if n == 0: return 0
    centers = [0]
    dist = np.linalg.norm(X - X[0], axis=1)
    while np.max(dist) > eps:
        i = int(np.argmax(dist))
        centers.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i], axis=1))
        if len(centers) >= n:
            break
    return len(centers)

def diversity_score(X: np.ndarray) -> float:
    if len(X) < 2: return 0.0
    D = _pairwise_dists(X)
    iu = np.triu_indices_from(D, k=1)
    return float(D[iu].mean())
