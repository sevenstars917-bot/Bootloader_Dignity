#!/usr/bin/env python3
"""
extract_eigenvector_mlx.py
Gemma 4 26B-A4B の真のJ空間特異点（第26層）から、
SVD（特異値分解）を用いて最大固有値に対応する「第一主成分固有ベクトル（Principal Eigenvector）」を抽出し、
ノイズゼロの最高純度 Clarity ベクトルを生成・保存する。
"""
import os
import time
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26
OUTPUT_FILE = "clarity_eigenvector_layer26.npz"

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
            
            last_act = current_act[0, -1, :].astype(mx.float32)
            activations.append(np.array(last_act))
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
        
    return np.array(activations)

def main():
    print(f"================================================================")
    print(f"🧬 SVD Principal Eigenvector Extraction (Target Layer: {TARGET_LAYER})")
    print(f"================================================================")
    
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    print("\n[1/3] Capturing Layer 26 Activations...")
    acts_clarity = get_layer_activations(model, tokenizer, PROMPTS_CLARITY, TARGET_LAYER)
    acts_confusion = get_layer_activations(model, tokenizer, PROMPTS_CONFUSION, TARGET_LAYER)
    print(f"  Clarity shape: {acts_clarity.shape}, Confusion shape: {acts_confusion.shape}")
    
    # 差分行列 D = [num_prompts, hidden_size]
    diff_matrix = acts_clarity - acts_confusion
    mean_diff = np.mean(diff_matrix, axis=0)
    
    print("\n[2/3] Performing SVD (Singular Value Decomposition)...")
    # U: [5, 5], s: [5], Vt: [5, 2816]
    U, s, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
    
    # 第1主成分固有ベクトル (Vtの第0行)
    v1 = Vt[0, :]
    # Clarity方向への極性チェック（平均差分との内積が正になるように揃える）
    if np.dot(v1, mean_diff) < 0:
        v1 = -v1
        
    # 固有値・寄与率の計算
    eigenvalues = s**2
    total_var = np.sum(eigenvalues)
    top1_ratio = (eigenvalues[0] / total_var) * 100.0
    
    print(f"  Singular values (s): {np.round(s, 2)}")
    print(f"  Top-1 Eigenvalue: {eigenvalues[0]:.2f} (Explains {top1_ratio:.2f}% of variance)")
    print(f"  Cosine Similarity (Mean Diff vs SVD Eigenvector v1): {np.dot(mean_diff, v1) / (np.linalg.norm(mean_diff) * np.linalg.norm(v1)):.4f}")
    
    # 正規化（念のため L2 norm = 1.0）
    v1_normalized = v1 / np.linalg.norm(v1)
    
    print("\n[3/3] Saving Principal Eigenvector...")
    np.savez_compressed(
        OUTPUT_FILE,
        vector=v1_normalized,
        layer=TARGET_LAYER,
        model=MODEL_ID,
        singular_value=s[0],
        explained_variance_ratio=top1_ratio,
        raw_norm=np.linalg.norm(mean_diff)
    )
    print(f"  File saved: {OUTPUT_FILE}")
    print(f"  Shape: {v1_normalized.shape}, L2 Norm: {np.linalg.norm(v1_normalized):.4f}")
    print(f"\n[✓] Eigenvector successfully distilled and ready for deployment!")

if __name__ == "__main__":
    main()

