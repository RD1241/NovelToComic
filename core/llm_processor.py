import json
import re
import ollama
from config import settings
from core.scene_interpreter import classify_scene, compute_panel_count


class LLMProcessor:
    """
    Extracts scenes from novel text via local Ollama LLM.

    The LLM is ONLY asked to extract facts (environment, characters, action,
    emotion, dialogue). It does NOT decide shot types, panel roles, or
    cinematic structure — those are user / interpreter concerns.
    """

    ENVIRONMENT_KEYWORDS = (
        "cafe", "coffee shop", "restaurant", "kitchen", "bedroom", "living room",
        "office", "park", "garden", "library", "classroom", "train station",
        "forest", "castle", "dungeon", "city", "street", "alley", "rooftop",
        "temple", "cave", "desert", "battlefield", "arena", "village",
        "mountain", "river", "bridge", "throne room", "ruins", "tower",
        "courtyard", "market", "lab", "ship", "school", "palace",
        "industrial district", "facility", "walkway", "steel beam",
    )

    def __init__(self, model_name=None):
        self.model_name = model_name or settings.LLM_MODEL
        self.system_prompt = """You are a comic storyboard extractor. Read the novel text and output ONLY valid JSON. No markdown, no explanation.

Extract scenes into this exact structure:
{"global_environment": "<where overall story happens, max 8 words>", "scenes": [{"scene_id": 1, "environment": "<location>", "focus_character": "<main char name>", "characters": [{"name": "<name>", "character_role": "main_character|secondary_character|enemy_character", "description": "<max 10 words, include male/female>"}], "action": "<what physically happens>", "emotion": "<mood>", "dialogue": [{"speaker": "<name or Narrator>", "type": "speech|narration", "text": "<words>"}]}]}

Rules:
- Use as many scenes as the text has beats (do not force exactly 4).
- Keep descriptions under 10 words. Include gender: male/female.
- Guards/soldiers = secondary_character. Enemies/monsters = enemy_character. Hero = main_character.
- Output ONLY the JSON object. Nothing before or after it."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shorten(self, text: str, max_words: int = 8) -> str:
        words = re.findall(r"[A-Za-z0-9'-]+", str(text or ""))
        return " ".join(words[:max_words]).lower()

    def _extract_json(self, text: str) -> dict | None:
        """
        Extract the first VALID JSON object from text using bracket-depth tracking.
        If the first '{...}' block found is not valid JSON (e.g. curly braces in
        surrounding prose like "Here is the JSON {output}:"), scanning continues
        from the next '{' until a valid object is found or the text is exhausted.
        """
        search_from = 0
        while True:
            start = text.find('{', search_from)
            if start == -1:
                return None
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i, ch in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                return None   # unmatched brace — no hope
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                # This '{...}' block wasn't valid JSON; skip past it and try again
                search_from = start + 1

    def _extract_global_environment(self, text: str, parsed: dict) -> str:
        for scene in parsed.get("scenes", []):
            env = str(scene.get("environment", "")).strip()
            if env and env.lower() not in ("none", "unknown", ""):
                return env
        text_lower = text.lower()
        for keyword in self.ENVIRONMENT_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                return keyword
        return "cinematic scene"

    def _apply_gender_bias_fix(self, character: dict) -> dict:
        """
        Reduce model female-bias by explicitly tagging gender in description.
        """
        desc = str(character.get("description", "")).lower()
        name = str(character.get("name", "")).lower()
        if "male" in desc or "man" in desc or "boy" in desc:
            character["_gender_tag"] = "male character, masculine features"
            character["_negative_gender"] = "feminine face, female anatomy"
        elif "female" in desc or "woman" in desc or "girl" in desc:
            character["_gender_tag"] = "female character"
            character["_negative_gender"] = ""
        # If ambiguous, leave unset so prompt builder doesn't inject noise
        return character

    def _normalize_characters(self, scene: dict):
        """Clean up character list and apply gender bias fix."""
        seen = set()
        normalized = []
        for char in scene.get("characters", []) or []:
            char = dict(char)
            key = str(char.get("name", "")).lower()
            if key in seen:
                continue
            seen.add(key)
            char = self._apply_gender_bias_fix(char)
            normalized.append(char)
        scene["characters"] = normalized

    def _normalize_storyboard(self, parsed: dict, source_text: str) -> dict:
        """
        Lightweight post-processing:
        - Lock the global environment
        - Clean characters
        - Trim scene count to what the narrative supports
        """
        scenes = parsed.get("scenes") or []

        # Determine how many panels we actually need
        panel_count = compute_panel_count(source_text, scenes)
        scenes = scenes[:panel_count]
        parsed["scenes"] = scenes

        # Lock environment across all panels from first scene
        global_env = self._extract_global_environment(source_text, parsed)
        global_env = self._shorten(global_env, 8)
        parsed["global_environment"] = global_env

        for index, scene in enumerate(scenes, start=1):
            scene["scene_id"] = index
            # Lock environment to prevent drift
            scene["environment"] = global_env
            scene["global_environment"] = global_env
            # Clean characters
            self._normalize_characters(scene)
            # Inject flags from scene interpreter
            flags = classify_scene(
                str(scene.get("action", "")) + " " + source_text
            )
            scene["is_action"] = flags["is_action"]
            scene["is_dialogue"] = flags["is_dialogue"]
            scene["is_calm"] = flags["is_calm"]

        return parsed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_text(self, text: str) -> dict:
        for attempt in range(3):
            content = None
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": text},
                    ],
                    options={
                        "temperature": 0.3,
                        "num_ctx":     6144,   # llama3 supports 8192; 6144 gives ~5k tokens for output
                        "num_predict": 3000,   # cap runaway generation
                    },
                    keep_alive=settings.LLM_KEEP_ALIVE,
                )
                content = response["message"]["content"]

                # Strip markdown fences
                clean = re.sub(r"```(?:json)?\s*", "", content).replace("```", "").strip()

                # Use bracket-depth extraction — immune to trailing text after '}'  
                parsed = self._extract_json(clean)
                if parsed:
                    parsed = self._normalize_storyboard(parsed, text)
                    return parsed

                print(f"[LLM] No JSON found on attempt {attempt + 1}. Raw output snippet:\n{clean[:500]}")

            except Exception as exc:
                print(f"[LLM] Error on attempt {attempt + 1}: {exc}")
                if attempt == 2:
                    print(f"[LLM] Raw output: {content}")
                import time
                time.sleep(2)

        return {"scenes": []}
