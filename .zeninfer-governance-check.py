#!/usr/bin/env python3
"""Small repository guard: prevent stale status language and missing governance references."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
required = ["ECC_BUILD_GOVERNANCE.md", "BUILD_PLAN.md", "docs/REMAINING_WORK.md"]
missing = [p for p in required if not (ROOT / p).is_file()]
if missing:
    print("FAIL: missing governance files: " + ", ".join(missing))
    sys.exit(1)
policy = (ROOT / "ECC_BUILD_GOVERNANCE.md").read_text(encoding="utf-8")
backlog = (ROOT / "docs/REMAINING_WORK.md").read_text(encoding="utf-8")
checks = {
    "RED gate documented": "RED" in policy and "before implementation" in policy,
    "GREEN gate documented": "GREEN" in policy,
    "NOT RUN semantics documented": "NOT RUN" in policy,
    "no fake ECC hooks claim": "does not install or execute ECC hooks" in policy,
    "backlog includes known open work": "ZI-008" in backlog and "ZI-009" in backlog,
}
failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("PASS" if ok else "FAIL") + ": " + name)
if failed:
    sys.exit(1)
