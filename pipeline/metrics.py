from typing import Dict, Any

class NarrativeMetricPlugin:
    """
    Base interface for future narrative metric modules.
    System allows metric injection without rewriting core logic.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def compute(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes a specific metric given category intelligence data.
        To be implemented by subclasses.
        """
        pass

class NarrativeVelocity(NarrativeMetricPlugin):
    """Stub for computing Delta convergence over time."""
    def compute(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation will analyze historical convergence trends
        return {"metric": "velocity", "value": 0.0, "status": "stub"}

class CrossCategoryCorrelation(NarrativeMetricPlugin):
    """Stub for computing overlaps Between categories."""
    def compute(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"metric": "correlation", "value": 0.0, "status": "stub"}

class VolatilityIndex(NarrativeMetricPlugin):
    """Stub for computing sentiment/bias volatility."""
    def compute(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"metric": "volatility", "value": 0.0, "status": "stub"}

class AnomalyDetection(NarrativeMetricPlugin):
    """Stub for outlying narrative detection."""
    def compute(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"metric": "anomaly", "detected": False, "status": "stub"}
