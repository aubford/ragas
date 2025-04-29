import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompts() -> dict[str, str]:
    """
    Loads all .md and .json prompt files in this directory and subdirectories.
    Returns a dict mapping file names (without extension) to their contents as strings.
    For .json files, the JSON is minified (no extra whitespace).
    """
    prompts = {}
    for file_path in PROMPTS_DIR.rglob("*.*"):
        key = str(file_path.relative_to(PROMPTS_DIR).with_suffix(""))
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                prompts[key] = json.dumps(obj, separators=(",", ":"))
        elif file_path.suffix == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                prompts[key] = f.read()
    return prompts
