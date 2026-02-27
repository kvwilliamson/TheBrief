import os
import re
import sys

# BAN LISTS
BANNED_KEYWORDS = [
    "crypto", "bitcoin", "gold", "silver", "rates", "inflation", 
    "equity", "stock", "forex", "uranium", "commodities"
]

BANNED_PATTERNS = [
    r"if .*? == ['\"].*?['\"]", # Categorical branching
    r"case ['\"].*?['\"]",      # Switch-case categorical branching
]

# EXEMPTIONS (Infrastructure/UI logic that uses if/case on strings)
EXEMPTIONS = [
    "__name__", "model_choice", "state", "selected_cat", "rec_cat", "new_cat_choice", "return_code", "SUMMARY_MODEL_NAME"
]

def check_file(filepath):
    errors = []
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.splitlines()

        # Check for banned keywords in variable names or logic (ignoring comments)
        for i, line in enumerate(lines):
            # Strip comments
            code_line = line.split('#')[0]
            
            # 1. BANNED KEYWORDS (Keyword Dictionaries / Static Asset Lists)
            for kw in BANNED_KEYWORDS:
                if re.search(fr"\b{kw}\b", code_line, re.IGNORECASE):
                    # Check if the line contains an exemption
                    if not any(ex in code_line for ex in EXEMPTIONS):
                        errors.append(f"Line {i+1}: Hardcoded asset/keyword reference found: '{kw}'")

            # 2. IF/CASE BRANCHING on categories
            for pattern in BANNED_PATTERNS:
                if "if" in code_line or "case" in code_line:
                    if re.search(pattern, code_line):
                        # Filter out common false positives and exemptions
                        if not any(ex in code_line for ex in EXEMPTIONS):
                            errors.append(f"Line {i+1}: Hardcoded categorical branching detected: '{line.strip()}'")

    return errors

def main():
    target_files = [
        "pipeline/summarization.py",
        "pipeline/clustering.py",
        "app.py"
    ]
    
    all_errors = []
    for f in target_files:
        if os.path.exists(f):
            print(f"Scanning {f}...")
            errors = check_file(f)
            all_errors.extend([(f, e) for e in errors])

    if all_errors:
        print("\n❌ ARCHITECTURAL GUARDRAIL VIOLATIONS FOUND:")
        for f, e in all_errors:
            print(f"[{f}] {e}")
        sys.exit(1)
    else:
        print("\n✅ All architectural guardrails passed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
