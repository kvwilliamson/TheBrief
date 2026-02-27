import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import os

logger = logging.getLogger(__name__)

def perform_semantic_clustering(videos, percentile=85, linkage='complete'):
    """
    Groups videos into narrative clusters based on an adaptive similarity threshold.
    
    Args:
        videos: List of video dictionaries, each containing an 'embedding' key.
        percentile: The percentile of similarity scores to use as the threshold.
        linkage: The linkage method for AgglomerativeClustering ('complete', 'average', 'single').
    """
    if not videos:
        return []
    
    embeddings = [v.get('embedding') for v in videos if v.get('embedding')]
    
    if len(embeddings) < 2:
        for i, v in enumerate(videos):
            v['cluster_id'] = 0 if v.get('embedding') else -1
            v['cluster_strength'] = 0.0
        return videos

    # Normalize embeddings
    embeddings = np.array(embeddings)
    
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Mask self-similarity
    mask = np.ones(similarities.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    scores = similarities[mask]
    
    if len(scores) > 0:
        adaptive_sim_threshold = np.percentile(scores, percentile)
        distance_threshold = 1.0 - adaptive_sim_threshold
    else:
        distance_threshold = 0.3 # Fallback
    
    # Agglomerative Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=max(0.01, distance_threshold), 
        metric='cosine', 
        linkage=linkage
    )
    
    labels = clustering.fit_predict(embeddings)
    
    # Assign labels and calculate high-fidelity diagnostics
    cluster_map = {}
    for i, label in enumerate(labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(i)
        
    strength_scores = {}
    coherence_scores = {}
    for label, indices in cluster_map.items():
        if len(indices) > 1:
            sub_sims = similarities[np.ix_(indices, indices)]
            mask = np.ones(sub_sims.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            mean_sim = np.mean(sub_sims[mask])
            coherence_scores[label] = float(mean_sim)
            strength_scores[label] = len(indices) * mean_sim
        else:
            coherence_scores[label] = 1.0
            strength_scores[label] = 1.0 # Base strength for singletons
    
    embedding_idx = 0
    for v in videos:
        if v.get('embedding'):
            label = int(labels[embedding_idx])
            size = len(cluster_map[label])
            v['cluster_id'] = label
            v['cluster_size'] = size
            v['cluster_coherence'] = coherence_scores[label]
            v['cluster_strength'] = float(strength_scores[label])
            v['is_singleton'] = size == 1
            # Coherence-based fracture detection
            v['is_fractured'] = size > 1 and coherence_scores[label] < 0.6
            embedding_idx += 1
        else:
            v['cluster_id'] = -1
            v['cluster_size'] = 0
            v['cluster_coherence'] = 0.0
            v['cluster_strength'] = 0.0
            v['is_singleton'] = True
            v['is_fractured'] = False
            
    # Instrumentation
    log_clustering_stats(len(videos), labels, distance_threshold, percentile, linkage, coherence_scores)
            
    return videos

def log_clustering_stats(video_count, labels, threshold, percentile, linkage, coherence_scores):
    """Logs detailed instrumentation for clustering quality audit."""
    os.makedirs("data", exist_ok=True)
    stats_path = "data/clustering_stats.json"
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = counts.tolist()
    singleton_count = cluster_sizes.count(1)
    max_size = max(cluster_sizes) if cluster_sizes else 0
    
    avg_coherence = float(np.mean(list(coherence_scores.values()))) if coherence_scores else 0.0
    
    stat = {
        "timestamp": datetime.now().isoformat(),
        "video_count": video_count,
        "cluster_count": len(unique_labels),
        "singleton_count": singleton_count,
        "max_cluster_size": max_size,
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0,
        "avg_coherence": avg_coherence,
        "threshold_dist": float(threshold),
        "percentile": percentile,
        "linkage": linkage,
        "distribution": cluster_sizes
    }
    
    # Append to daily log
    history = []
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                history = json.load(f)
        except:
            history = []
            
    history.append(stat)
    with open(stats_path, "w") as f:
        json.dump(history[-30:], f, indent=2) # Keep last 30 runs

from datetime import datetime
