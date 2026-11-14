#!/usr/bin/env python3
"""
client_test.py
Windows (Obsidian) から Mac mini (192.168.0.9:8080) 上の
MLX Steered Inference Server を叩くテストクライアント。
"""
import sys
import json
import time
import urllib.request

SERVER_URL = "http://192.168.0.9:8080"

def test_health():
    print(f"[1/3] Checking Server Health ({SERVER_URL}/health)...")
    req = urllib.request.Request(f"{SERVER_URL}/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("  Response:", json.dumps(data, indent=2, ensure_ascii=False))

def test_chat_non_stream():
    print(f"\n[2/3] Testing Non-Streaming Chat Completion...")
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "user", "content": "1+1の答えを一言で教えて。"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        content = res["choices"][0]["message"]["content"]
        usage = res.get("usage", {})
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Content:\n{content.strip()}")
        print(f"  Usage: {usage}")

def test_chat_stream():
    print(f"\n[3/3] Testing Streaming Chat Completion (SSE)...")
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "user", "content": "バットとボールの合計は110円。バットはボールより100円高い。ボールの値段は？簡潔に結論を述べて。"}
        ],
        "max_tokens": 250,
        "temperature": 0.7,
        "stream": True,
        "alpha": 1.0  # 第26層 Clarity ステアリング
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.time()
    print("  Streaming output: ", end="", flush=True)
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                except Exception:
                    pass
    print(f"\n  Elapsed: {time.time() - t0:.2f}s")

def main():
    try:
        test_health()
        test_chat_non_stream()
        test_chat_stream()
        print("\n[✓] All Client Tests Passed Successfully!")
    except Exception as e:
        print(f"\n[!] Test Failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

