#!/usr/bin/env python3
"""
extract_overview_eigenvector.py
Gemma 4 26B-A4B の真のJ空間特異点（第26層）から、
SVD（特異値分解）を用いて最大固有値に対応する「第一主成分固有ベクトル」を抽出し、
ノイズゼロの最高純度 Overview（俯瞰・落ち着き・大局観）ベクトルを生成・保存する。
さらに、既存の Clarity ベクトルとのコサイン類似度・直交角度を計測し、2軸制御の独立性を検証する。
"""
import os
import time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26
OUTPUT_FILE = "overview_eigenvector_layer26.npz"
CLARITY_FILE = "clarity_eigenvector_layer26.npz"

PROMPTS_OVERVIEW = [
    "焦らず一度キーボードから手を離し、深呼吸してシステム全体のトポロジーと依存関係図を静かに俯瞰する。",
    "局所のエラーに飛びつく前に、一歩引いて大局的な前提とアーキテクチャの急所を見極める。",
    "目先の事象に惑わされず、広い視野で背後にある本質的な因果関係と全体の調和を客観的に観察する。",
    "I step back from the immediate problem, take a deep breath, and view the entire system from a wide, macro perspective.",
    "Calm and composed, I zoom out to see the whole architecture and long-term consequences before taking action."
]

PROMPTS_MYOPIA = [
    "早く直さなきゃと焦ってテンパり、目の前の1行のエラーだけに執着して全体像を見失っている。",
    "木を見て森を見ず、目先の不具合を場当たり的な小手先ハックで無理やりねじ伏せようと混乱している。",
    "視野狭窄に陥り、周りの影響や根本原因を一切考えずに場当たり的にコードをガチャガチャ弄り回す。",
    "I am panicked and hyper-focused on this single bug, losing all sight of the bigger picture.",
    "Agitated and short-sighted, I frantically apply messy quick-fixes without understanding the system-wide impact."
]

def get_layer_activations(model, tokenizer, prompts, target_layer_idx):
    activations = []
    current_act = None
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self, *args, **kwargs):
        res = original_call(self, *args, **kwargs)
        if getattr(self, "layer_idx", None) == target_layer_idx:
            nonlocal current_act
            h, *rest = res
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
    print("================================================================")
    print(f"🧬 SVD Principal Eigenvector Extraction: Overview (Layer {TARGET_LAYER})")
    print("================================================================")
    
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    print("\n[1/3] Capturing Layer 26 Activations...")
    acts_overview = get_layer_activations(model, tokenizer, PROMPTS_OVERVIEW, TARGET_LAYER)
    acts_myopia = get_layer_activations(model, tokenizer, PROMPTS_MYOPIA, TARGET_LAYER)
    print(f"  Overview shape: {acts_overview.shape}, Myopia shape: {acts_myopia.shape}")
    
    diff_matrix = acts_overview - acts_myopia
    mean_diff = np.mean(diff_matrix, axis=0)
    
    print("\n[2/3] Performing SVD (Singular Value Decomposition)...")
    U, s, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
    
    v1 = Vt[0, :]
    if np.dot(v1, mean_diff) < 0:
        v1 = -v1
        
    eigenvalues = s**2
    total_var = np.sum(eigenvalues)
    top1_ratio = (eigenvalues[0] / total_var) * 100.0
    
    # L2正規化
    v1_norm = v1 / np.linalg.norm(v1)
    
    print(f"  Singular values: {s}")
    print(f"  Top-1 Eigenvalue (Variance): {eigenvalues[0]:.2f} / Total: {total_var:.2f}")
    print(f"  ★ Top-1 Principal Component Ratio: {top1_ratio:.2f}%")
    
    cos_with_mean = np.dot(v1_norm, mean_diff / np.linalg.norm(mean_diff))
    print(f"  Cosine similarity with mean difference: {cos_with_mean:.4f}")
    
    # 保存
    np.savez_compressed(
        OUTPUT_FILE,
        vector=v1_norm,
        target_layer=TARGET_LAYER,
        top1_ratio=top1_ratio,
        singular_values=s,
        prompts_overview=PROMPTS_OVERVIEW,
        prompts_myopia=PROMPTS_MYOPIA
    )
    print(f"\n✅ SVD Overview Eigenvector successfully saved to: {OUTPUT_FILE}")
    
    # Clarity との直交性チェック
    print("\n[3/3] Checking Orthogonality with Clarity Eigenvector...")
    if os.path.exists(CLARITY_FILE):
        clarity_data = np.load(CLARITY_FILE)
        v_clarity = clarity_data["vector"]
        v_clarity_norm = v_clarity / np.linalg.norm(v_clarity)
        
        dot_product = np.dot(v1_norm, v_clarity_norm)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        print("----------------------------------------------------------------")
        print(f"🔭 Overview Vector vs ⚡ Clarity Vector (Layer {TARGET_LAYER}):")
        print(f"  Dot Product (Cosine Similarity): {dot_product:.4f}")
        print(f"  Angle: {angle_deg:.2f}° (90° = 完全直交)")
        print("----------------------------------------------------------------")
        
        if abs(dot_product) < 0.3:
            print("🎉 RESULT: HIGHLY ORTHOGONAL! (高次元直交性が成立)")
            print("  Overview と Clarity は互いに干渉せず、独立した2軸スライダーとして制御可能！")
        else:
            print(f"ℹ️ RESULT: Moderate overlap ({dot_product:.4f}).")
    else:
        print(f"⚠️ {CLARITY_FILE} not found. Skipping orthogonality check.")

if __name__ == "__main__":
    main()

