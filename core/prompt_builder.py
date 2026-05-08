"""
prompt_builder.py
-----------------
Minimal, prompt-driven. No forced camera angles, no PANEL_LOCKS.
Prompt structure: character description, environment, action, emotion, style
"""
from config import settings


# ---------------------------------------------------------------------------
# Negative prompt — kept lean, no aggressive over-filtering
# ---------------------------------------------------------------------------
BASE_NEGATIVE = (
    "bad anatomy, extra limbs, deformed hands, blurry, low quality, "
    "worst quality, watermark, text, signature, duplicate character"
)


class PromptBuilder:

    def __init__(self, style: str = None):
        # Style is injected at construction or overridden per call
        self.style = style or getattr(settings, "DEFAULT_STYLE", "anime")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gender_tag(self, character: dict) -> str:
        """Return gender enforcement tokens if the character has them."""
        return character.get("_gender_tag", "")

    def _gender_negative(self, character: dict) -> str:
        return character.get("_negative_gender", "")

    def _character_token(self, character: dict, is_focus: bool) -> str:
        """
        Build a compact character token.
        Priority: name + description + gender tag.
        """
        name  = str(character.get("name", "")).strip()
        desc  = str(character.get("description", "")).strip()
        gtag  = self._gender_tag(character)

        parts = [p for p in [name, desc, gtag] if p]
        token = ", ".join(parts)

        # Consistency anchor for non-first panels
        if is_focus and character.get("_is_continuation"):
            token += ", same face, same outfit, same hairstyle"

        return token

    def _style_token(self) -> str:
        style = self.style.lower()
        style_map = {
            "anime":     "anime illustration, vibrant colors",
            "manga":     "manga illustration, black and white, ink lines",
            "manhwa":    "manhwa illustration, soft colors, detailed",
            "realistic": "realistic illustration, photorealistic",
            "cinematic": "cinematic render, dramatic lighting",
        }
        return style_map.get(style, f"{style} illustration")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        scene: dict,
        memory_manager,
        is_continuation: bool = False,
        style: str = None,
    ) -> tuple[str, str]:
        """
        Build (positive_prompt, negative_prompt) for a single panel.

        Priority order:
          1. Character description (identity)
          2. Environment (scene lock)
          3. Action (what is happening)
          4. Emotion (mood)
          5. Style (art direction)

        No forced camera angles. No PANEL_LOCKS. Prompt IS the controller.
        """
        active_style = style or self.style

        parts = []
        neg_parts = [BASE_NEGATIVE]

        # ----- 1. CHARACTERS -------------------------------------------
        characters = scene.get("characters") or []
        focus_name = str(scene.get("focus_character", "")).lower()
        focus_char = None

        for char in characters:
            char_name = str(char.get("name", "")).lower()
            is_focus  = focus_name and focus_name in char_name

            # Pull cached description for consistency
            cached_desc = memory_manager.get_character(char.get("name", ""))
            if cached_desc:
                char = dict(char)
                char["description"] = cached_desc

            if is_continuation:
                char = dict(char)
                char["_is_continuation"] = True

            token = self._character_token(char, is_focus=is_focus)
            if token:
                parts.append(token)

            # Accumulate gender negatives
            neg = self._gender_negative(char)
            if neg:
                neg_parts.append(neg)

            if is_focus:
                focus_char = char

        # ----- 2. ENVIRONMENT ------------------------------------------
        env = str(
            scene.get("global_environment") or scene.get("environment", "")
        ).strip()
        if env:
            parts.append(env)

        # ----- 3. ACTION -----------------------------------------------
        action = str(scene.get("action", "")).strip()
        if action:
            parts.append(action)

        # ----- 4. EMOTION ----------------------------------------------
        emotion = str(scene.get("emotion", "")).strip()
        if emotion:
            parts.append(emotion)

        # ----- 5. STYLE ------------------------------------------------
        parts.append(self._style_token() if active_style == self.style else f"{active_style} illustration")

        positive = ", ".join(p for p in parts if p)
        negative = ", ".join(p for p in neg_parts if p)

        return positive, negative

    def apply_reference_conditioning_prompt(
        self, positive_prompt: str, negative_prompt: str
    ) -> tuple[str, str]:
        """
        Tighten identity for panels 2+ that use an IP-Adapter reference image.
        Adds minimal consistency anchors without forcing composition.
        """
        if "same face" not in positive_prompt:
            positive_prompt += ", same face, same outfit, consistent character"
        return positive_prompt, negative_prompt
