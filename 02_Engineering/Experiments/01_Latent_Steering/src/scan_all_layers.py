#!/usr/bin/env python3
"""
scan_all_layers.py
Gemma 4 26B-A4B の全30層（Layer 0〜29）を走査し、
Clarity vs Confusion の分離度、コサイン距離、SVD固有値（第1主成分寄与率）を算出して
真の「J空間（最適ステアリング層）」を特定する。
"""
import time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
NUM_LAYERS = 30

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

def capture_all_layers_activations(model, tokenizer, prompts):
    """1回のフォワードで全30層の最終トークンアクティベーションを一括取得"""
    # [num_prompts, num_layers, hidden_size]
    all_acts = [[] for _ in range(NUM_LAYERS)]
    current_layer_acts = [None] * NUM_LAYERS
    
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self_layer, x, *args, **kwargs):
        res = original_call(self_layer, x, *args, **kwargs)
        idx = getattr(self_layer, "layer_idx", None)
        if idx is not None and 0 <= idx < NUM_LAYERS:
            h, *rest = res
            current_layer_acts[idx] = h
        return res
        
    gemma4_text.DecoderLayer.__call__ = hook_call
    
    try:
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(formatted)
            tokens_mx = mx.array([tokens])
            
            for i in range(NUM_LAYERS):
                current_layer_acts[i] = None
                
            _ = model(tokens_mx)
            
            for i in range(NUM_LAYERS):
                act_mx = current_layer_acts[i]
                mx.eval(act_mx)
                # 最後のトークンを float32 で保存
                last_act = np.array(act_mx[0, -1, :].astype(mx.float32))
                all_acts[i].append(last_act)
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
        
    # [NUM_LAYERS, num_prompts, hidden_size]
    return [np.array(acts) for acts in all_acts]

def main():
    print(f"================================================================")
    print(f"🔬 Gemma 4 26B-A4B: All {NUM_LAYERS} Layers J-Space Scanner")
    print(f"================================================================")
    
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    print("\nScanning Clarity activations across all layers...")
    acts_clarity = capture_all_layers_activations(model, tokenizer, PROMPTS_CLARITY)
    
    print("Scanning Confusion activations across all layers...")
    acts_confusion = capture_all_layers_activations(model, tokenizer, PROMPTS_CONFUSION)
    
    print("\nComputing Layer Metrics (Separation, Cosine Distance, SVD Top-1 Ratio)...")
    results = []
    
    for l in range(NUM_LAYERS):
        c_acts = acts_clarity[l]   # [5, 2816]
        f_acts = acts_confusion[l] # [5, 2816]
        
        c_mean = np.mean(c_acts, axis=0)
        f_mean = np.mean(f_acts, axis=0)
        
        diff = c_mean - f_mean
        diff_norm = np.linalg.norm(diff)
        base_norm = 0.5 * (np.linalg.norm(c_mean) + np.linalg.norm(f_mean))
        rel_diff = (diff_norm / base_norm) * 100.0 # パーセント相対分離度
        
        cos_sim = np.dot(c_mean, f_mean) / (np.linalg.norm(c_mean) * np.linalg.norm(f_mean))
        cos_dist = 1.0 - cos_sim
        
        # SVD (固有値・主成分分析)
        # 対照ペアの差分行列: shape [5, 2816]
        diff_matrix = c_acts - f_acts
        # 特異値分解
        _, s, _ = np.linalg.svd(diff_matrix, full_matrices=False)
        top1_variance_ratio = (s[0]**2) / np.sum(s**2) * 100.0 # 第1主成分の寄与率(%)
        
        results.append({
            "layer": l,
            "diff_norm": diff_norm,
            "base_norm": base_norm,
            "rel_diff": rel_diff,
            "cos_dist": cos_dist,
            "top1_s": s[0],
            "top1_ratio": top1_variance_ratio
        })
        
    print("\n-----------------------------------------------------------------------------------------")
    print(f"{'Layer':<6} | {'Diff Norm':<10} | {'Base Norm':<10} | {'Rel Sep (%)':<12} | {'Cos Dist':<10} | {'Top-1 Ratio':<12} | Notes")
    print("-----------------------------------------------------------------------------------------")
    
    # 相対分離度でソートしたランキング上位をマーク
    rel_ranks = np.argsort([r["rel_diff"] for r in results])[::-1]
    best_layer = rel_ranks[0]
    
    for r in results:
        l = r["layer"]
        marker = ""
        if l == best_layer:
            marker = "★ BEST J-SPACE (Max Relative Separation)"
        elif l == 14:
            marker = "(Current 14th Layer)"
        elif l in rel_ranks[:3]:
            marker = "★ Top 3"
            
        print(f"L{l:<4} | {r['diff_norm']:<10.2f} | {r['base_norm']:<10.2f} | {r['rel_diff']:<12.2f}% | {r['cos_dist']:<10.4f} | {r['top1_ratio']:<11.1f}% | {marker}")
        
    print("-----------------------------------------------------------------------------------------")
    print(f"\n🎯 Recommended Layer: Layer {best_layer} (Relative Separation: {results[best_layer]['rel_diff']:.2f}%, Top-1 SVD Ratio: {results[best_layer]['top1_ratio']:.1f}%)")

if __name__ == "__main__":
    main()

