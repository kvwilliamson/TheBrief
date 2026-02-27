import os
import json
import numpy as np
from pipeline.summarization import calculate_percentile, update_convergence_history, load_config
from metrics.cross_cluster_correlation import CrossClusterCorrelation

def test_narrative_intelligence():
    print("🚀 Starting Strategic Intelligence Stress Test...")
    
    # 1. Cold Start Resilience
    history_path = "data/convergence_history.json"
    if os.path.exists(history_path):
        os.remove(history_path)
    
    print("Testing Cold Start rank calculation...")
    p = calculate_percentile("Biotech", 0.75)
    assert p == 0.0 # Expected fallback for new category
    
    # 2. Cross-Sector Correlation Detection
    print("Testing Cross-Sector Correlation logic...")
    config = load_config()
    correlator = CrossClusterCorrelation(config)
    
    # Mock data with overlapping narratives
    # Imagine 'Finance' and 'Metals' both discussing 'Yield Curve'
    mock_centroid = [1.0] * 768 # Dummy high-similarity vector
    
    mock_intel = {
        "Finance": {
            "clusters": [
                {"id": 1, "name": "Credit Markets Tighten", "centroid": mock_centroid}
            ]
        },
        "Metals": {
            "clusters": [
                {"id": 2, "name": "Monetary Policy Stress", "centroid": [0.99] * 768}
            ]
        }
    }
    
    correlations = correlator.detect(mock_intel)
    print(f"Detected {len(correlations)} correlations.")
    assert len(correlations) > 0
    assert correlations[0]["similarity"] > 0.95
    
    # 3. Dynamic Category Propagation (Direct JSON check)
    print("Verifying dynamic category propagation from channels.json...")
    channels_path = "channels.json"
    if os.path.exists(channels_path):
        with open(channels_path, "r") as f:
            data = json.load(f)
            channels = data.get("channels", [])
            categories = set(c.get("category") for c in channels if c.get("category"))
            print(f"Discovered categories: {categories}")
            assert len(categories) > 0
    
    print("✅ Strategic Intelligence Stress Test PASSED.")

if __name__ == "__main__":
    test_narrative_intelligence()
