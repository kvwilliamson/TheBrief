import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from pipeline.summarization import run_summarization
from pipeline.clustering import perform_semantic_clustering

class TestNarrativeRegression(unittest.TestCase):
    
    def test_domain_agnostic_clustering(self):
        """
        GIVEN diverse transcripts (Geopolitics, Tech, Finance)
        WHEN clustered
        THEN system must group by semantic similarity without category logic.
        """
        # Create distinct vectors (not just multiples)
        def make_vec(val, idx):
            vec = [0.01] * 768
            vec[idx] = val
            return vec

        mock_videos = [
            # Group A: Geopolitics (Middle East) - High weight on early indices
            {"id": "v1", "title": "Middle East Conflict", "embedding": make_vec(0.9, 0)},
            {"id": "v2", "title": "Regional Tensions Rise", "embedding": make_vec(0.85, 0)},
            # Group B: Technology (AI) - High weight on middle indices
            {"id": "v3", "title": "AI Breakout", "embedding": make_vec(0.9, 384)},
            {"id": "v4", "title": "LLM Efficiency", "embedding": make_vec(0.85, 384)},
            # Group C: Unknown Asset (Uranium Miners) - High weight on late indices
            {"id": "v5", "title": "Uranium Supply Squeeze", "embedding": make_vec(0.9, 700)},
        ]
        
        # Test adaptive clustering
        clustered = perform_semantic_clustering(mock_videos, percentile=50)
        
        cluster_ids = [v['cluster_id'] for v in clustered]
        self.assertEqual(len(set(cluster_ids)), 3, f"Should have 3 distinct clusters, got {set(cluster_ids)}")
        
        # Verify Group A clustered together
        self.assertEqual(clustered[0]['cluster_id'], clustered[1]['cluster_id'])
        # Verify Group B clustered together
        self.assertEqual(clustered[2]['cluster_id'], clustered[3]['cluster_id'])
        # Verify Group C is distinct
        self.assertNotEqual(clustered[0]['cluster_id'], clustered[2]['cluster_id'])
        self.assertNotEqual(clustered[0]['cluster_id'], clustered[4]['cluster_id'])

    @patch('pipeline.summarization.get_llm')
    @patch('pipeline.summarization.summarize_transcript')
    @patch('pipeline.summarization.get_embedding')
    @patch('pipeline.summarization.generate_cluster_label')
    @patch('pipeline.summarization.generate_meta_summary')
    def test_pipeline_resilience(self, mock_meta, mock_label, mock_embed, mock_summ, mock_llm):
        """
        Verify the full run_summarization pipeline works with mocked components.
        """
        mock_llm.return_value = MagicMock()
        mock_summ.return_value = {
            "episode_title": "Test Video",
            "channel": "Test Channel",
            "one_line_summary": "Test Summary",
            "signal_strength": 8,
            "themes": ["test"],
            "core_claims": ["claim 1"],
            "positioning_implication": "implication",
            "time_horizon": "medium",
            "shelf_life": "Medium",
            "podcast_date": "2024-01-01",
            "processing_date": "2024-01-01"
        }
        mock_embed.return_value = [0.1] * 768
        mock_label.return_value = {
            "cluster_name": "Test Cluster",
            "description": "Test Description",
            "positioning_bias": "neutral"
        }
        mock_meta.return_value = "Meta summary."
        
        # Mock queue data
        import json
        import os
        os.makedirs("data", exist_ok=True)
        with open("data/queue.json", "w") as f:
            json.dump([{"id": "1", "title": "V1", "url": "url1", "channel": "C1"}], f)
            
        try:
            # Fix Mock: TinyDB is imported inside the function
            with patch('pipeline.summarization.send_email_digest'), \
                 patch('tinydb.TinyDB'):
                run_summarization()
                
            # Verify file exists
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            self.assertTrue(os.path.exists(f"briefs/{date_str}.json"))
            
            with open(f"briefs/{date_str}.json", "r") as f:
                data = json.load(f)
                self.assertIn("meta_summary", data)
                self.assertIn("clusters", data)
                self.assertEqual(data["clusters"][0]["name"], "Test Cluster")
                
        finally:
            if os.path.exists("data/queue.json"):
                os.remove("data/queue.json")

if __name__ == '__main__':
    unittest.main()
