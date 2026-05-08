"""
predownload_models.py
---------------------
Downloads ALL models used by NovelToComic directly to D:\AI_Models\HuggingFace.
Run ONCE. After this, the pipeline runs fully offline forever.

Usage:
    .\venv\Scripts\python.exe predownload_models.py
"""
import os
import sys

# ── Set HF_HOME BEFORE any huggingface import ─────────────────────────────────
HF_HOME = r"D:\AI_Models\HuggingFace"
os.environ["HF_HOME"]              = HF_HOME
os.environ["HF_HUB_CACHE"]        = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"]= HF_HOME
print(f"[Setup] HF_HOME = {HF_HOME}\n")

import torch
from huggingface_hub import snapshot_download
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
)

DTYPE = torch.float16   # fp16 saves ~50% disk vs fp32

# ── Models to download ────────────────────────────────────────────────────────
SD15_MODELS = [
    ("Lykon/dreamshaper-8",                   "anime / default"),
    ("Linaqruf/anything-v3-better-vae",       "manga / manhwa"),
    ("SG161222/Realistic_Vision_V5.1_noVAE",  "realistic"),
]

SDXL_MODELS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "cinematic (SDXL - large ~6 GB)"),
]

CONTROLNET_MODELS = [
    ("lllyasviel/control_v11p_sd15_openpose", "pose skeleton"),
]

# IP-Adapter: only specific files needed (not full repo)
IP_ADAPTER_REPO     = "h94/IP-Adapter"
IP_ADAPTER_PATTERNS = [
    "models/ip-adapter_sd15.bin",
    "models/ip-adapter_sd15_light.bin",
]

def separator(label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)

def download_sd15(model_id, label):
    separator(f"SD 1.5 Model: {model_id}  [{label}]")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            safety_checker=None,
            local_files_only=False,
        )
        del pipe   # free RAM — we only needed the download
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[OK] {model_id} downloaded to {HF_HOME}")
    except Exception as e:
        print(f"[FAIL] {model_id} — {e}")

def download_sdxl(model_id, label):
    separator(f"SDXL Model: {model_id}  [{label}]")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            use_safetensors=True,
            local_files_only=False,
        )
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[OK] {model_id} downloaded to {HF_HOME}")
    except Exception as e:
        print(f"[FAIL] {model_id} — {e}")

def download_controlnet(model_id, label):
    separator(f"ControlNet: {model_id}  [{label}]")
    try:
        cn = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            local_files_only=False,
        )
        del cn
        print(f"[OK] {model_id} downloaded to {HF_HOME}")
    except Exception as e:
        print(f"[FAIL] {model_id} — {e}")

def download_ip_adapter():
    separator(f"IP-Adapter: {IP_ADAPTER_REPO}")
    try:
        path = snapshot_download(
            repo_id=IP_ADAPTER_REPO,
            allow_patterns=IP_ADAPTER_PATTERNS,
            local_files_only=False,
        )
        print(f"[OK] IP-Adapter downloaded to {path}")
    except Exception as e:
        print(f"[FAIL] IP-Adapter — {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("NovelToComic — Model Pre-Downloader")
    print(f"All files will be saved to: {HF_HOME}\n")

    # --- SD 1.5 models (core pipeline) ---
    for model_id, label in SD15_MODELS:
        download_sd15(model_id, label)

    # --- ControlNet ---
    for model_id, label in CONTROLNET_MODELS:
        download_controlnet(model_id, label)

    # --- IP-Adapter (selective files only) ---
    download_ip_adapter()

    # --- SDXL (optional, large) ---
    skip_sdxl = "--skip-sdxl" in sys.argv
    if skip_sdxl:
        print("\n[Skipped] SDXL cinematic model (--skip-sdxl flag set)")
    else:
        print("\n[Note] Downloading SDXL (~6 GB). Use --skip-sdxl to skip.")
        for model_id, label in SDXL_MODELS:
            download_sdxl(model_id, label)

    # ── Final verification ────────────────────────────────────────────────────
    separator("Download Summary — D:\\AI_Models\\HuggingFace")
    total = sum(
        f.stat().st_size
        for f in __import__("pathlib").Path(HF_HOME).rglob("*")
        if f.is_file()
    )
    print(f"Total cached: {total / 1e9:.2f} GB in {HF_HOME}")

    # Permanent env var — survives terminal restarts
    import subprocess
    subprocess.run(
        ['setx', 'HF_HOME', HF_HOME],
        capture_output=True
    )
    subprocess.run(
        ['setx', 'HF_HUB_CACHE', HF_HOME],
        capture_output=True
    )
    print(f"\n[Setup] HF_HOME permanently set to {HF_HOME} (user env var)")
    print("[Done] All models ready. Pipeline will run fully offline.")
