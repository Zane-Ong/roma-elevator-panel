import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["TORCH_HOME"] = str(ROOT / "models")
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / "temp")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HOME"] = str(ROOT / "models" / "huggingface")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
for folder in (ROOT / "temp", ROOT / "results"):
    folder.mkdir(exist_ok=True)
