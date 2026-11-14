import urllib.request
import json

url = "https://huggingface.co/api/models?author=mlx-community&search=26b&limit=15"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("=== MLX 26B Models ===")
        for d in data:
            print(d["id"])
except Exception as e:
    print(f"Error 26b: {e}")

url2 = "https://huggingface.co/api/models?author=mlx-community&search=Qwen2.5-32B&limit=10"
req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req2) as resp:
        data2 = json.loads(resp.read().decode())
        print("\n=== MLX Qwen2.5-32B Models ===")
        for d in data2:
            print(d["id"])
except Exception as e:
    print(f"Error 32b: {e}")

