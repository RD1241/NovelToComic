"""
scene_interpreter.py
--------------------
Lightweight scene classifier. Returns only flags — does NOT generate prompts.
This is a read-only module: it classifies, never controls.
"""
import re


ACTION_WORDS = frozenset([
    "attack", "attacks", "attacked", "attacking",
    "fight", "fights", "fighting", "fought",
    "jump", "jumps", "jumped", "jumping",
    "dash", "dashes", "dashed", "dashing",
    "strike", "strikes", "struck", "striking",
    "slam", "slams", "slammed", "slamming",
    "slash", "slashes", "slashed", "slashing",
    "smash", "smashes", "smashed", "smashing",
    "charge", "charges", "charged", "charging",
    "kick", "kicks", "kicked", "kicking",
    "punch", "punches", "punched", "punching",
    "clash", "clashes", "clashed",
    "battle", "battles", "battled",
    "combat", "fight",
])

DIALOGUE_WORDS = frozenset([
    "said", "says", "asked", "asks", "replied",
    "replied", "shouted", "whispered", "muttered",
    "told", "answered", "called", "cried",
])

CALM_WORDS = frozenset([
    "stood", "sat", "gazed", "looked", "watched",
    "waited", "breathed", "rested", "paused",
    "smiled", "thought", "remembered", "recalled",
    "walked", "strolled", "wandered",
])

# Grouped style keywords — aliases are matched first, style name returned
STYLE_KEYWORDS = {
    "realistic": ["realistic", "photo", "photorealistic", "cinematic"],
    "manga":     ["manga", "black and white", "ink"],
    "manhwa":    ["manhwa", "webtoon", "korean comic"],
    "anime":     ["anime", "illustration", "animated"],
}

_DEFAULT_STYLE = "anime"


def classify_scene(text: str) -> dict:
    """
    Classify a single scene/beat of text.

    Returns:
        {
            "is_action": bool,
            "is_dialogue": bool,
            "is_calm": bool,
        }
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    return {
        "is_action":   bool(words & ACTION_WORDS),
        "is_dialogue": bool(words & DIALOGUE_WORDS),
        "is_calm":     bool(words & CALM_WORDS) and not bool(words & ACTION_WORDS),
    }


def detect_style(text: str) -> str:
    """
    Extract a user-specified art style from the input text.
    Checks grouped keywords (including aliases like 'photo', 'webtoon').
    Fallback: DEFAULT_STYLE.
    """
    text_lower = text.lower()
    for style, keywords in STYLE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return style
    return _DEFAULT_STYLE


def compute_panel_count(text: str, scenes: list) -> int:
    """
    Determine the appropriate panel count based on narrative complexity.

    Rules:
      - Character intro / static scene    →  1 panel
      - Simple scene                      →  2 panels
      - Moderate progression              →  3-4 panels
      - Action-heavy / multi-beat text    →  4-6 panels
    """
    scene_count = len(scenes)
    if scene_count == 0:
        return 1

    word_count  = len(text.split())
    flags       = classify_scene(text)

    # Determine raw count based on complexity
    if flags["is_action"] and word_count >= 150:
        raw = min(scene_count, 6)
    elif flags["is_action"] or word_count >= 80:
        raw = min(scene_count, 4)
    elif word_count >= 40:
        raw = min(scene_count, 3)
    elif word_count >= 20:
        raw = min(scene_count, 2)
    else:
        raw = 1

    # Safety clamp: guarantee result is always within [1, 6]
    # Apply lower bound first (max), then upper bound (min)
    return min(max(raw, 1), 6)
