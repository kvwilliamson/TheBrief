import os
import json
from typing import Dict, Any

def load_intelligence_profiles() -> Dict[str, Any]:
    profile_path = os.path.join("data", "intelligence_profiles.json")
    if not os.path.exists(profile_path):
        # Fallback to a basic speculative profile if missing
        return {
            "Speculative": {"features": {"tradeability": True, "novelty": True}}
        }
    with open(profile_path, "r") as f:
        return json.load(f)

def get_profile_for_category(category: str) -> Dict[str, Any]:
    data = load_intelligence_profiles()
    profiles = data.get("profiles", {})
    mappings = data.get("mappings", {})
    
    profile_key = mappings.get(category, "Speculative")
    return profiles.get(profile_key, profiles.get("Speculative", {}))
