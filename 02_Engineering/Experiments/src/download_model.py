import sys
from huggingface_hub import snapshot_download

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
print(f"=== Starting Download of {MODEL_ID} ===")
try:
    path = snapshot_download(repo_id=MODEL_ID)
    print(f"\n[✓] Download Successfully Complete!")
    print(f"Local Path: {path}")
except Exception as e:
    print(f"\n[!] Download Failed: {e}", file=sys.stderr)
    sys.exit(1)

