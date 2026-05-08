import os, sys
os.environ['HF_HOME']      = 'D:\\AI_Models\\HuggingFace'
os.environ['HF_HUB_CACHE'] = 'D:\\AI_Models\\HuggingFace'
sys.path.insert(0, 'd:/Project_I/NovelToComic')

from config import settings
from pathlib import Path

print('=== MODEL_MAP ===')
for style, model_id in settings.MODEL_MAP.items():
    print(f'  {style:10} -> {model_id}')

hf_cache = Path(settings.HF_HOME)
cached = [d.name for d in hf_cache.iterdir() if d.is_dir() and d.name.startswith('models--')]

print('\n=== Models on D drive ===')
for m in sorted(cached):
    print(f'  {m}')

print('\n=== Coverage Check ===')
map_models = set(settings.MODEL_MAP.values())
all_ok = True
for model_id in sorted(map_models):
    folder = 'models--' + model_id.replace('/', '--')
    found  = folder in cached
    status = 'OK' if found else 'MISSING'
    print(f'  [{status}] {model_id}')
    if not found:
        all_ok = False

print('\n=== ControlNet ===')
cn_folder = 'models--' + settings.SD_CONTROLNET_MODEL_ID.replace('/', '--')
cn_found  = cn_folder in cached
print(f'  [{"OK" if cn_found else "MISSING"}] {settings.SD_CONTROLNET_MODEL_ID}')

print('\n=== IP-Adapter ===')
ip_folder = 'models--' + settings.SD_IP_ADAPTER_MODEL_ID.replace('/', '--')
ip_found  = ip_folder in cached
print(f'  [{"OK" if ip_found else "MISSING"}] {settings.SD_IP_ADAPTER_MODEL_ID}')

print(f'\nHF_HOME = {settings.HF_HOME}')
print('DEFAULT_STYLE =', settings.DEFAULT_STYLE)
status = 'ALL MODELS READY' if (all_ok and cn_found and ip_found) else 'SOME MODELS MISSING'
print(f'\n>>> {status}')
