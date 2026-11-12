#!/usr/bin/env python3
"""
extract_vector_mlx.py
Gemma 4 26B-A4B (MLX) の第14層から Clarity（明晰・確信）ベクトルを抽出する。
"""
import os
import time
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 14
OUTPUT_FILE = "clarity_vector_layer14.npz"

PROMPTS_CLARITY = [
    "The logic is completely clear. 1 + 1 = 2. Verified and mathematically certain.",
    "All contradictions are fully resolved. The conclusion is absolute and indisputable.",
    "The derivation is clean, concise, and flawlessly verified.",
    "矛盾は完全に解消され、論理的帰結として明白である。確信がある。",
    "明晰な視界のもと、解は一意に定まり、疑う余地は存在しない。"
]

PROMPTS_CONFUSION = [
    "Wait, but what if I am completely mistaken? I doubt everything and feel lost.",
    "This makes no sense, maybe it is a trick. I cannot decide and keep second-guessing.",
    "I am confused, going in circles, unable to find any solid ground or truth.",
    "待て、本当にそうか？何か重大な誤りがあるのではないか、疑念が晴れない。",
    "混乱して思考が堂々巡りし、何が正しいのか全く確信が持てず迷走している。"
]

def get_layer_activations(model, tokenizer, prompts, target_layer_idx):
    activations = []
    current_act = None
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self, *args, **kwargs):
        res = original_call(self, *args, **kwargs)
        if getattr(self, "layer_idx", None) == target_layer_idx:
            nonlocal current_act
            h, kvs, offset = res
            current_act = h
        return res
        
    gemma4_text.DecoderLayer.__call__ = hook_call
    
    try:
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(formatted)
            tokens_mx = mx.array([tokens])
            
            current_act = None
            _ = model(tokens_mx)
            mx.eval(current_act)
            
            # 最後のトークンのアクティベーションを取得 (float32へキャストしてからnumpy化)
            last_act = current_act[0, -1, :].astype(mx.float32)
            activations.append(np.array(last_act))
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
        
    return np.array(activations)

def main():
    print(f"--- MLX Clarity Vector Extraction (Target Layer: {TARGET_LAYER}) ---")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    print("\n[1/3] Capturing Clarity Activations...")
    acts_clarity = get_layer_activations(model, tokenizer, PROMPTS_CLARITY, TARGET_LAYER)
    mean_clarity = np.mean(acts_clarity, axis=0)
    print(f"  Clarity shape: {acts_clarity.shape}, mean norm: {np.linalg.norm(mean_clarity):.4f}")
    
    print("\n[2/3] Capturing Confusion Activations...")
    acts_confusion = get_layer_activations(model, tokenizer, PROMPTS_CONFUSION, TARGET_LAYER)
    mean_confusion = np.mean(acts_confusion, axis=0)
    print(f"  Confusion shape: {acts_confusion.shape}, mean norm: {np.linalg.norm(mean_confusion):.4f}")
    
    print("\n[3/3] Computing Difference Vector & Normalizing...")
    diff_vector = mean_clarity - mean_confusion
    norm = np.linalg.norm(diff_vector)
    normalized_vector = diff_vector / norm
    
    print(f"  Raw diff norm: {norm:.4f}")
    print(f"  Normalized vector shape: {normalized_vector.shape}")
    
    # 差分ベクトルの自己相関チェック
    cos_sim = np.dot(mean_clarity, mean_confusion) / (np.linalg.norm(mean_clarity) * np.linalg.norm(mean_confusion))
    print(f"  Cosine similarity (Clarity vs Confusion): {cos_sim:.4f}")
    
    np.savez_compressed(
        OUTPUT_FILE,
        vector=normalized_vector,
        layer=TARGET_LAYER,
        model=MODEL_ID,
        raw_norm=norm
    )
    print(f"\n[✓] Successfully saved to {OUTPUT_FILE}!")

if __name__ == "__main__":
    main()
