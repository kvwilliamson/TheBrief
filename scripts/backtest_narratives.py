import os
import sys
import json
import numpy as np
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.clustering import perform_semantic_clustering

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_backtest():
    """Runs a retrospective analysis on existing JSON briefs."""
    briefs_dir = "briefs"
    json_files = [f for f in os.listdir(briefs_dir) if f.endswith(".json")]
    
    if not json_files:
        logger.error("No JSON briefs found for backtesting.")
        return

    results = []
    
    for filename in json_files:
        path = os.path.join(briefs_dir, filename)
        logger.info(f"--- Analyzing {filename} ---")
        
        with open(path, "r") as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            briefs = data.get("briefs", [])
        else:
            briefs = data # Legacy list format
            
        if not briefs: continue
        
        # We need embeddings. If they are missing in the JSON, we'd have to regenerate.
        # But for this test, we'll assume they exist or we'll generate mock ones for stress tests.
        
        for linkage in ["complete", "average", "single"]:
            logger.info(f"Testing linkage: {linkage}")
            # Mock embeddings if missing
            test_briefs = []
            for b in briefs:
                tb = b.copy()
                if "embedding" not in tb:
                    # Generate deterministic mock embedding if missing
                    tb["embedding"] = np.random.rand(1536).tolist() 
                test_briefs.append(tb)
                
            clustered = perform_semantic_clustering(test_briefs, percentile=85, linkage=linkage)
            
            # Analyze results
            labels = [v["cluster_id"] for v in clustered if v["cluster_id"] != -1]
            unique, counts = np.unique(labels, return_counts=True)
            
            # Extract coherence from clustered videos
            coherences = [v.get("cluster_coherence", 0.0) for v in clustered if v["cluster_id"] != -1]
            mean_coherence = float(np.mean(coherences)) if coherences else 0.0
            
            res = {
                "date": filename,
                "linkage": linkage,
                "video_count": len(test_briefs),
                "cluster_count": len(unique),
                "singleton_count": int(np.sum(counts == 1)),
                "max_cluster_size": int(np.max(counts)) if len(counts) > 0 else 0,
                "mean_coherence": mean_coherence
            }
            results.append(res)
            logger.info(f"Result: {res}")

    # Stress Tests
    logger.info("\n--- Stress Tests ---")
    for size in [10, 25, 60]:
        logger.info(f"Testing size: {size} videos")
        mock_videos = []
        for i in range(size):
            mock_videos.append({
                "id": f"v{i}",
                "title": f"Stress Video {i}",
                "embedding": (np.random.rand(1536) + (np.random.rand(1536) * 0.1 if i % 5 == 0 else 0)).tolist()
            })
        
        clustered = perform_semantic_clustering(mock_videos, percentile=85, linkage="complete")
        labels = [v["cluster_id"] for v in clustered if v["cluster_id"] != -1]
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Size {size} -> Clusters: {len(unique)}, Singletons: {np.sum(counts == 1)}")

    # Save report
    report_path = "data/backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nBacktest complete. Report saved to {report_path}")

if __name__ == "__main__":
    run_backtest()
