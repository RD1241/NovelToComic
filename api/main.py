from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import json
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.llm_processor import LLMProcessor
from core.memory_manager import MemoryManager
from core.prompt_builder import PromptBuilder
from core.sd_generator import SDGenerator
from core.comic_renderer import ComicRenderer
from core.job_manager import JobManager
from core.cache_manager import CacheManager
from core.monitoring import DriftMonitor
from core.scene_interpreter import detect_style

app = FastAPI(title="Novel to Comic API", description="Enterprise-Grade AI Pipeline")

llm_processor = LLMProcessor()
memory_manager = MemoryManager()

sd_generator = SDGenerator()
comic_renderer = ComicRenderer()
job_manager = JobManager()
cache_manager = CacheManager()
drift_monitor = DriftMonitor()

# Single worker to protect GPU from OOM
job_executor = ThreadPoolExecutor(max_workers=1)

class NovelInput(BaseModel):
    text: str
    style: str = ""   # optional: anime | manga | manhwa | realistic | cinematic

def ok(message: str, data: dict = None) -> dict:
    """Standard success response."""
    return {"status": "success", "message": message, "data": data or {}}

def fail(message: str, detail: str = None) -> JSONResponse:
    """Standard failure response."""
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": message, "detail": detail or ""}
    )

def process_job_worker(job_id: str, text: str, style: str = "anime"):
    """Background worker that handles the heavy AI generation."""
    start_time = time.time()
    try:
        job_manager.update_job(job_id, status="processing")

        # Step 0: Clear memory to prevent trait poisoning from previous run
        memory_manager.clear_memory()

        # Step 1: LLM scene extraction (with retry already in llm_processor)
        job_manager.update_job(job_id, status="processing", progress="Extracting scenes with LLM...")
        scene_data = llm_processor.process_text(text)
        if not scene_data.get("scenes"):
            raise ValueError("Failed to extract scenes from text.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_output_dir = os.path.join(settings.OUTPUTS_DIR, timestamp)
        os.makedirs(job_output_dir, exist_ok=True)

        # Build a prompt_builder keyed to the detected style
        style_prompt_builder = PromptBuilder(style=style)

        output_images = []
        global_env = None
        character_reference_path = None
        scene_seed = settings.SD_DEFAULT_SEED
        base_latents = sd_generator.create_base_latents(scene_seed)
        total_panels = len(scene_data["scenes"])

        # Step 2: Generate each panel
        for i, scene in enumerate(scene_data["scenes"]):
            panel_num = i + 1
            job_manager.update_job(
                job_id, status="processing",
                progress=f"Drawing panel {panel_num}/{total_panels}..."
            )

            # Lock environment from Panel 1 to all panels
            if not global_env and scene.get("environment"):
                global_env = scene.get("environment")
            if global_env:
                scene["environment"] = global_env
                scene["global_environment"] = global_env
                scene["environment_locked"] = True

            memory_manager.process_scene_characters(scene)
            pos_prompt, neg_prompt = style_prompt_builder.build_prompt(
                scene, memory_manager, is_continuation=(i > 0), style=style
            )

            use_reference = i > 0 and character_reference_path and os.path.exists(character_reference_path)
            if use_reference:
                pos_prompt, neg_prompt = style_prompt_builder.apply_reference_conditioning_prompt(pos_prompt, neg_prompt)

            image_filename = f"scene_{scene.get('scene_id', i + 1)}.png"
            image_path = os.path.join(job_output_dir, image_filename)

            # Retry loop for SD generation
            success = False
            last_error = None
            for attempt in range(settings.MAX_RETRIES):
                try:
                    sd_generator.generate_image(
                        positive_prompt=pos_prompt,
                        negative_prompt=neg_prompt,
                        output_path=image_path,
                        seed=scene_seed + i,
                        reference_image_path=character_reference_path if use_reference else None,
                        reference_strength=settings.SD_REFERENCE_STRENGTH if use_reference else None,
                        base_latents=base_latents,
                        scene_id=scene.get("scene_id", i + 1),
                        panel_index=i,
                        style=style,
                        action=scene.get("action", ""),
                    )
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    print(f"[Warning] Panel {panel_num} attempt {attempt + 1} failed: {e}")
                    # Simplify prompt on retry
                    pos_prompt = pos_prompt.split(",")[0] + ", " + settings.MASTER_STYLE_TAG

            if not success:
                raise RuntimeError(f"Panel {panel_num} failed after {settings.MAX_RETRIES} attempts: {last_error}")

            # Draw speech bubbles
            if scene.get("dialogue"):
                try:
                    comic_renderer.draw_speech_bubble(image_path, scene["dialogue"], image_path)
                except Exception as e:
                    print(f"[Warning] Speech bubble failed for panel {panel_num}: {e}")

            output_images.append(image_path)

            # Extract character anchor from Panel 1 for IP-Adapter
            if i == 0:
                character_reference_path = os.path.join(job_output_dir, "character_ref.png")
                try:
                    sd_generator.extract_character_anchor(image_path, character_reference_path)
                except Exception as e:
                    print(f"[Warning] Character anchor extraction failed: {e}")
                    character_reference_path = None

        # Step 3: Assemble comic page
        job_manager.update_job(job_id, status="processing", progress="Assembling final comic page...")
        final_page_filename = "final_comic_page.png"
        final_page_path = os.path.join(job_output_dir, final_page_filename)
        comic_renderer.create_comic_page(output_images, final_page_path)

        # Save metadata
        try:
            with open(os.path.join(job_output_dir, "metadata.json"), "w") as f:
                json.dump(scene_data, f, indent=4)
        except Exception:
            pass

        # Step 4: Cleanup VRAM — unload model to free memory
        try:
            sd_generator.unload_model()
        except Exception as e:
            print(f"[Warning] Model unload failed: {e}")

        # Step 5: Build result and cache
        web_panels = [f"/outputs/{timestamp}/{os.path.basename(p)}" for p in output_images]
        web_final = f"/outputs/{timestamp}/{final_page_filename}"

        result_payload = {
            "message": "Comic generated successfully!",
            "total_scenes": len(scene_data["scenes"]),
            "final_page": web_final,
            "panels": web_panels,
            "scenes": scene_data["scenes"],
        }

        gen_time = time.time() - start_time
        drift_monitor.log_job_metrics(job_id, text, scene_data["scenes"], gen_time, True, output_images)
        cache_manager.set_cached_result(text, result_payload)
        job_manager.update_job(job_id, status="completed", result=json.dumps(result_payload))

    except Exception as e:
        gen_time = time.time() - start_time
        print(f"[Error] Job {job_id} failed: {e}")
        drift_monitor.log_job_metrics(job_id, text, [], gen_time, False, [])
        job_manager.update_job(job_id, status="failed", error=str(e))
        # Always free VRAM on failure
        try:
            sd_generator.unload_model()
        except Exception:
            pass


@app.post("/api/generate_comic")
def generate_comic(novel_input: NovelInput):
    text = (novel_input.text or "").strip()

    # Resolve style: user-specified > auto-detected from text > default
    raw_style  = (novel_input.style or "").strip().lower()
    style      = raw_style if raw_style in getattr(settings, "MODEL_MAP", {}) else detect_style(text)
    style      = style or getattr(settings, "DEFAULT_STYLE", "anime")

    # Input validation
    if len(text) < settings.MIN_INPUT_LENGTH:
        return fail(
            f"Input too short. Please provide at least {settings.MIN_INPUT_LENGTH} characters.",
            f"Got {len(text)} characters."
        )
    if len(text) > settings.MAX_INPUT_LENGTH:
        return fail(
            f"Input too long. Max {settings.MAX_INPUT_LENGTH} characters allowed.",
            f"Got {len(text)} characters."
        )

    # Cache check
    if getattr(settings, "ENABLE_CACHING", True):
        cached = cache_manager.get_cached_result(text)
        if cached:
            job_id = job_manager.create_job()
            job_manager.update_job(job_id, status="completed", result=json.dumps(cached))
            return {"status": "success", "message": "Returned from cache.", "job_id": job_id, "cached": True, "data": {"job_id": job_id}}

    # Queue new job
    job_id = job_manager.create_job()
    job_executor.submit(process_job_worker, job_id, text, style)
    return {"status": "success", "message": "Job queued successfully.", "job_id": job_id, "cached": False, "data": {"job_id": job_id}}


@app.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/health")
def get_health():
    return drift_monitor.get_system_health()


@app.get("/api/metrics")
def get_metrics():
    return drift_monitor.get_system_health()


os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.OUTPUTS_DIR), name="outputs")
app.mount("/", StaticFiles(directory=os.path.join(settings.BASE_DIR, "frontend"), html=True), name="frontend")
