"""Phase 3 (STUB): language command -> skill for the tracking policy.

The idea: every retargeted clip becomes a named skill with a short text description
(skills/skills.json). A language model picks the skill (and later, parameters such as
direction or target object); the phase-2 tracking policy executes it.

Today this file only does keyword matching so the interface can be wired up and tested.
Swap `select_skill` for a VLM/LLM call once phase 2 produces a working tracker.
"""
from __future__ import annotations

import json
import re


def load_skills(path: str = "skills/skills.json") -> dict:
    with open(path) as f:
        return json.load(f)


def select_skill(command: str, skills: dict) -> str | None:
    """Return the skill whose description shares the most words with the command."""
    words = set(re.findall(r"[a-z]+", command.lower()))
    best, score = None, 0
    for name, meta in skills.items():
        s = len(words & set(re.findall(r"[a-z]+", (name + " " + meta["description"]).lower())))
        if s > score:
            best, score = name, s
    return best


if __name__ == "__main__":
    import sys
    print(select_skill(" ".join(sys.argv[1:]) or "wave your hand", load_skills()))
