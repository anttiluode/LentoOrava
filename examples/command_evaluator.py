"""Minimal subprocess evaluator for the PulseTriage CLI example."""

from __future__ import annotations

import json
import os


HARMFUL = {"flag-017": 1.7, "flag-083": 0.9, "flag-144": 1.3, "flag-231": 0.7}
rolled_back = set(json.loads(os.environ.get("PULSE_TRIAGE_ROLLBACK_JSON", "[]")))
print(90.0 + sum(HARMFUL.get(item, 0.0) for item in rolled_back))
