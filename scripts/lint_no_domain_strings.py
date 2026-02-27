import os
import json
import re
import sys

def load_categories():
    try:
        with open("channels.json", "r") as f:
            data = json.load(f)
            categories = set()
            for channel in data.get("channels", []):
                cat = channel.get("category")
                if cat:
                    categories.add(cat)
            return list(categories)
    except FileNotFoundError:
        return []

def load_framing_blacklist():
    try:
        with open("config.json", "r") as f:
            data = json.load(f)
            return data.get("clustering", {}).get("framing_blacklist", [])
    except FileNotFoundError:
        return []

def lint():
    categories = load_categories()
    framing_blacklist = load_framing_blacklist()
    
    print(f"🔍 Monitoring {len(categories)} categories and {len(framing_blacklist)} framing phrases for purity...")
    
    # Files to scan
    scan_dirs = ["pipeline"]
    scan_files = ["app.py", "main.py"]
    
    violation_found = False
    
    files_to_scan = []
    for d in scan_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".py"):
                    files_to_scan.append(os.path.join(root, f))
    
    for f in scan_files:
        if os.path.exists(f):
            files_to_scan.append(f)
            
    for file_path in files_to_scan:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # Ignore comments 
                strippable_line = line.split("#")[0] 
                
                # 1. Check for Category Hardcoding
                for cat in categories:
                    if f'"{cat}"' in strippable_line or f"'{cat}'" in strippable_line:
                        # Allow category names in testing/verification scripts
                        if "verify" in file_path or "test" in file_path:
                            continue
                        print(f"❌ Violation in {file_path}:{i+1} - Hardcoded category found: '{cat}'")
                        violation_found = True
                
                # 2. Check for Framing Blacklist (Recap language)
                for phrase in framing_blacklist:
                    if phrase in strippable_line: # Use substring check for phrases
                        # ALLOWANCE: In summarization.py, these phrases are allowed ONLY within the instruction prompt
                        # that tells the LLM to avoid them.
                        if "summarization.py" in file_path:
                            # Heuristic: Allow if the line contains 'STRICTLY FORBIDDEN' or 'DO NOT use'
                            if "STRICTLY FORBIDDEN" in line or "DO NOT use" in line or "template =" in line:
                                continue
                                
                        # Allow in config-loading logic
                        if ".get(\"framing_blacklist\"" in line:
                            continue
                            
                        print(f"❌ Violation in {file_path}:{i+1} - Hardcoded framing phrase found: '{phrase}'")
                        violation_found = True
                        
    if violation_found:
        print("\n🛑 LINT FAILED: Domain-purity or framing-recap violations found.")
        sys.exit(1)
    else:
        print("\n✅ LINT PASSED: Purity enforced.")
        sys.exit(0)

if __name__ == "__main__":
    lint()
