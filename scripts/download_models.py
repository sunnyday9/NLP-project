
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add project root to path (optional, but good practice)
sys.path.append(str(Path(__file__).resolve().parent.parent))

def _load_dotenv(dotenv_path: Path) -> None:
    """
    Minimal .env loader (no external dependency).
    Supports lines like:
      HF_TOKEN=xxxxx
      # comment
      export HF_TOKEN="xxxxx"
    """
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1].strip()

        # Only set if not already present in the OS environment.
        os.environ.setdefault(key, val)


def _repo_id_to_cache_dirname(repo_id: str) -> str:
    # Hugging Face cache uses: models--{org}--{name}
    return "models--" + repo_id.replace("/", "--")


def _looks_like_model_cached(cache_dir: Path, repo_id: str) -> bool:
    """
    Heuristic cache check based on HF cache structure:
    cache_dir/models--{org}--{name}/snapshots/<snapshot>/
    """
    repo_cache = cache_dir / _repo_id_to_cache_dirname(repo_id) / "snapshots"
    if not repo_cache.exists():
        return False

    try:
        snapshots = [p for p in repo_cache.iterdir() if p.is_dir()]
    except OSError:
        return False

    if not snapshots:
        return False

    weight_markers = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "tf_model.h5",
        "flax_model.msgpack",
    )

    for snap in snapshots:
        for marker in weight_markers:
            if (snap / marker).exists():
                return True
        if any(snap.glob("model-*-of-*.safetensors")) or any(snap.glob("pytorch_model-*-of-*.bin")):
            return True

    return False


def download_models():
    models = [
        "Qwen/Qwen2.5-Math-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ]

    # Load token from .env (if present)
    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv(project_root / ".env")

    # Common env var names used by Hugging Face ecosystem
    hf_token = (
        os.getenv("HF_TOKEN")
    )
    token_kwargs = {"token": hf_token} if hf_token else {}
    
    # Save models in project-local cache directory
    cache_dir = Path("model_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to: {cache_dir.absolute()}")
    
    for model_name in models:
        print(f"\n--- Downloading {model_name} ---")
        # Avoid re-downloading if already cached (HF cache presence check).
        if _looks_like_model_cached(cache_dir, model_name):
            print("Detected existing cache. Skipping download.")
            continue

        # If cache seems incomplete, try local-only load first (still may work without network).
        if token_kwargs:
            local_kwargs = dict(token_kwargs)
        else:
            local_kwargs = {}

        try:
            AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
                local_files_only=True,
                **local_kwargs,
            )
            AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
                **local_kwargs,
            )
            print("Already cached locally. Skipping download.")
            continue
        except Exception:
            pass

        print("Downloading tokenizer...")
        AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            **token_kwargs,
        )

        print("Downloading model weights (this may take a while)...")
        AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **token_kwargs,
        )
        print(f"Finished downloading {model_name}")

if __name__ == "__main__":
    download_models()
