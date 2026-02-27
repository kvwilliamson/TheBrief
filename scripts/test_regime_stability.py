import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.curdir))

from pipeline.summarization import calculate_convergence_score, load_config

def generate_mock_cluster(size, centroid):
    """Generates a mock cluster structure for the convergence formula."""
    return {
        "size": size,
        "brief_data": [{"embedding": centroid.tolist()}] * size
    }

def test_monotonicity():
    print("🧪 Running Regime Stability: Monotonicity Test...")
    config = load_config()
    dim = 768 # Gemini embedding dimension
    
    # Case 1: Identical Centroids (Max Convergence)
    v1 = np.ones(dim)
    v1 /= np.linalg.norm(v1)
    
    clusters_max = [
        generate_mock_cluster(10, v1),
        generate_mock_cluster(10, v1)
    ]
    score_max = calculate_convergence_score(clusters_max, 20, config)['score']
    
    # Case 2: Orthogonal Centroids (Low Convergence)
    v2 = np.zeros(dim)
    v2[0] = 1.0
    v3 = np.zeros(dim)
    v3[1] = 1.0
    
    clusters_low = [
        generate_mock_cluster(10, v2),
        generate_mock_cluster(10, v3)
    ]
    score_low = calculate_convergence_score(clusters_low, 20, config)['score']
    
    print(f"  - Max Convergence (Identical): {score_max:.4f}")
    print(f"  - Min Convergence (Orthogonal): {score_low:.4f}")
    
    assert score_max > score_low, "FAILED: Identical centroids must have higher score than orthogonal."
    print("  ✅ Monotonicity verified.")

def test_dominance_influence():
    print("🧪 Running Regime Stability: Dominance Test...")
    config = load_config()
    dim = 768
    v1 = np.ones(dim)
    v1 /= np.linalg.norm(v1)
    
    # High similarity but low dominance
    clusters_equal = [
        generate_mock_cluster(5, v1),
        generate_mock_cluster(5, v1),
        generate_mock_cluster(5, v1),
        generate_mock_cluster(5, v1)
    ]
    score_equal = calculate_convergence_score(clusters_equal, 20, config)['score']
    
    # High similarity AND high dominance (largest cluster = 17/20)
    clusters_dominant = [
        generate_mock_cluster(17, v1),
        generate_mock_cluster(3, v1)
    ]
    score_dominant = calculate_convergence_score(clusters_dominant, 20, config)['score']
    
    print(f"  - Equal Clusters: {score_equal:.4f}")
    print(f"  - Dominant Cluster (85%): {score_dominant:.4f}")
    
    assert score_dominant > score_equal, "FAILED: Dominant cluster should boost convergence score."
    print("  ✅ Dominance weighting verified.")

def test_edge_cases():
    print("🧪 Running Regime Stability: Edge Cases...")
    config = load_config()
    
    # Single cluster (should be 0)
    res = calculate_convergence_score([generate_mock_cluster(10, np.ones(768))], 10, config)
    assert res['score'] == 0.0
    
    # Small clusters (< min_size)
    clusters_tiny = [generate_mock_cluster(1, np.ones(768)), generate_mock_cluster(1, np.ones(768))]
    res_tiny = calculate_convergence_score(clusters_tiny, 2, config)
    assert res_tiny['score'] == 0.0
    
    print("  ✅ Edge cases handled correctly.")

if __name__ == "__main__":
    try:
        test_monotonicity()
        test_dominance_influence()
        test_edge_cases()
        print("\n🏆 ALL REGIME STABILITY TESTS PASSED.")
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        sys.exit(1)
