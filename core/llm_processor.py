import json
import re
import time
import httpx
import ollama
from config import settings
from core.scene_interpreter import classify_scene, compute_panel_count

# Module-level client bound to the configured Ollama host
_ollama_client = ollama.Client(host=settings.OLLAMA_HOST)


def _wait_for_ollama(timeout: int = 30) -> bool:
    """
    Poll the Ollama HTTP endpoint until it responds or timeout expires.
    Returns True if ready, False if timed out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


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
- ALL scenes MUST be inside the single top-level "scenes" array. Never split scenes into multiple arrays or objects.
- Output ONLY the single JSON object. Nothing before or after it. No trailing commas."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _repair_json(self, text: str) -> str:
        """
        Fix llama3's most common malformed output: it sometimes splits the
        scenes array into multiple separate arrays separated by commas:

            "scenes": [{obj1}], [{obj2}]   ← BROKEN (two arrays)

        This repairs it to:

            "scenes": [{obj1}, {obj2}]     ← VALID (one array)

        The regex targets the exact boundary pattern: `}], [{`
        (end of a JSON object, close bracket, comma, open bracket, new object)
        and replaces it with `}, {` which simply continues the same array.
        """
        repaired = re.sub(r'\}\]\s*,\s*\[', '}, ', text)
        return repaired

    def _shorten(self, text: str, max_words: int = 8) -> str:
        words = re.findall(r"[A-Za-z0-9'-]+", str(text or ""))
        return " ".join(words[:max_words]).lower()

    def _extract_json(self, text: str) -> dict | None:
        """
        Extract a valid storyboard JSON object from LLM output.

        Handles two failure modes common with smaller Ollama models:

        1. Normal case — single well-formed JSON object somewhere in the text.

        2. Split-array bug — the LLM outputs the first scene inside the main
           object and the remaining scenes as a dangling array, e.g.:
               {"global_environment": "...", "scenes": [{scene_1}]},
               [{scene_2}, {scene_3}]
           We detect the trailing '[ ... ]' that follows the first object and
           splice those extra scenes back in before parsing.
        """
        search_from = 0

        while True:
            # ── Step 1: find the start of the next '{' candidate ──────────
            start = text.find('{', search_from)
            if start == -1:
                return None

            # ── Step 2: bracket-depth scan to find matching '}' ───────────
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
                return None  # unmatched brace — no hope

            candidate = text[start:end + 1]

            # ── Step 3: try to fix the split-array bug before parsing ─────
            #
            # Pattern:  <candidate>  ,  [ {scene}, {scene} ... ]
            # The bit after the closing '}' may look like:  , [{...}, ...]
            # We scan for it and, if found, merge into the candidate.
            candidate_fixed = self._try_merge_split_scenes(candidate, text[end + 1:])

            # ── Step 4: parse — prefer merged version, fall back to raw ──
            for blob in (candidate_fixed, candidate):
                if blob is None:
                    continue
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    pass

            # Neither worked; skip past this '{' and keep searching
            search_from = start + 1

    # ------------------------------------------------------------------

    def _try_merge_split_scenes(self, first_obj: str, remainder: str) -> str | None:
        """
        Detect the split-array LLM bug and repair it.

        The LLM sometimes emits:
            {"global_environment":"...","scenes":[{scene1}]},\n[{scene2},{scene3}]

        We look for a '[' in *remainder* (skipping whitespace / commas)
        and, if found, extract all scene objects from that array and inject
        them into the "scenes" list of *first_obj* before parsing.

        Returns the repaired JSON string, or None if the pattern isn't found.
        """
        # Skip whitespace and commas immediately after the closing '}'
        stripped = remainder.lstrip(', \t\r\n')
        if not stripped.startswith('['):
            return None  # no dangling array — nothing to do

        # Find the matching ']' for this array
        depth = 0
        in_string = False
        escape = False
        arr_end = -1
        for i, ch in enumerate(stripped):
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
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    arr_end = i
                    break

        if arr_end == -1:
            return None  # array is truncated — can't recover

        extra_array_str = stripped[:arr_end + 1]
        try:
            extra_scenes = json.loads(extra_array_str)
        except json.JSONDecodeError:
            return None  # dangling array itself is malformed

        if not isinstance(extra_scenes, list):
            return None

        # Parse the first object and splice in the extra scenes
        try:
            obj = json.loads(first_obj)
        except json.JSONDecodeError:
            return None

        existing = obj.get("scenes", [])
        if not isinstance(existing, list):
            existing = []
        obj["scenes"] = existing + extra_scenes

        print(f"[LLM] Split-array bug detected and repaired: merged {len(extra_scenes)} extra scene(s).")
        return json.dumps(obj)

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
        # Wait for Ollama to be reachable before first attempt
        if not _wait_for_ollama(timeout=30):
            print("[LLM] Ollama not reachable after 30s — aborting.")
            return {"scenes": []}

        for attempt in range(3):
            content = None
            try:
                response = _ollama_client.chat(
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

                # Repair llama3's broken split-array output BEFORE parsing
                clean = self._repair_json(clean)

                # Use bracket-depth extraction — immune to trailing text after '}'  
                parsed = self._extract_json(clean)
                if parsed and parsed.get("scenes"):
                    parsed = self._normalize_storyboard(parsed, text)
                    return parsed

                print(f"[LLM] Invalid JSON or empty scenes on attempt {attempt + 1}. Raw output snippet:\n{clean[:500]}")

            except Exception as exc:
                print(f"[LLM] Error on attempt {attempt + 1}: {exc}")
                if attempt == 2:
                    print(f"[LLM] Raw output: {content}")
                import time
                time.sleep(2)

        return {"scenes": []}
