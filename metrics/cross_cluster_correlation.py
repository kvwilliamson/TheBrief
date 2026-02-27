import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List

class CrossClusterCorrelation:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config.get("clustering", {}).get("correlation_threshold", 0.75)

    def detect(self, category_intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects narrative overlaps between clusters across different categories.
        """
        all_clusters = []
        for cat_name, cat_data in category_intelligence.items():
            for cluster in cat_data.get("clusters", []):
                # We need the centroid. If not present, we can't correlate.
                # In summarization.py, we'll need to store the centroid in the cluster data.
                if "centroid" in cluster:
                    all_clusters.append({
                        "category": cat_name,
                        "cluster_id": cluster["id"],
                        "cluster_name": cluster["name"],
                        "centroid": cluster["centroid"]
                    })
        
        correlations = []
        if not all_clusters:
            return correlations

        centroids = np.array([c["centroid"] for c in all_clusters])
        sim_matrix = cosine_similarity(centroids)

        for i in range(len(all_clusters)):
            for j in range(i + 1, len(all_clusters)):
                if sim_matrix[i][j] > self.threshold:
                    cluster_a = all_clusters[i]
                    cluster_b = all_clusters[j]
                    
                    # Only correlate across different categories
                    if cluster_a["category"] != cluster_b["category"]:
                        correlations.append({
                            "source": {
                                "category": cluster_a["category"],
                                "name": cluster_a["cluster_name"],
                                "id": cluster_a["cluster_id"]
                            },
                            "target": {
                                "category": cluster_b["category"],
                                "name": cluster_b["cluster_name"],
                                "id": cluster_b["cluster_id"]
                            },
                            "similarity": float(sim_matrix[i][j])
                        })
        
        return correlations
