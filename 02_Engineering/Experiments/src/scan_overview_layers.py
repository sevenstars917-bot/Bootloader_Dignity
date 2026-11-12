#!/usr/bin/env python3
"""
scan_overview_layers.py
Gemma 4 26B-A4B の全30層（Layer 0〜29）を走査し、
Overview（大局観・俯瞰・冷静） vs Myopia（視野狭窄・テンパり・短絡）の
相対分離度、コサイン距離、SVD固有値（第1主成分寄与率）を算出して
真の「Overview J空間（最適ステアリング層）」を特定する。
"""
import time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
NUM_LAYERS = 30

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

def capture_all_layers_activations(model, tokenizer, prompts):
    """1回のフォワードで全30層の最終トークンアクティベーションを一括取得"""
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
                last_act = np.array(act_mx[0, -1, :].astype(mx.float32))
                all_acts[i].append(last_act)
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
        
    return [np.array(acts) for acts in all_acts]

def main():
    print("================================================================")
    print(f"🔬 Gemma 4 26B-A4B: All {NUM_LAYERS} Layers Overview Scanner")
    print("================================================================")
    
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    print("\n[1/3] Scanning Overview activations across all layers...")
    acts_overview = capture_all_layers_activations(model, tokenizer, PROMPTS_OVERVIEW)
    
    print("[2/3] Scanning Myopia activations across all layers...")
    acts_myopia = capture_all_layers_activations(model, tokenizer, PROMPTS_MYOPIA)
    
    print("\n[3/3] Computing Layer Metrics (Separation, Cosine Distance, SVD Top-1 Ratio)...")
    results = []
    
    for l in range(NUM_LAYERS):
        o_acts = acts_overview[l] # [5, 2816]
        m_acts = acts_myopia[l]   # [5, 2816]
        
        o_mean = np.mean(o_acts, axis=0)
        m_mean = np.mean(m_acts, axis=0)
        
        diff = o_mean - m_mean
        diff_norm = np.linalg.norm(diff)
        base_norm = 0.5 * (np.linalg.norm(o_mean) + np.linalg.norm(m_mean))
        rel_diff = (diff_norm / base_norm) * 100.0
        
        cos_sim = np.dot(o_mean, m_mean) / (np.linalg.norm(o_mean) * np.linalg.norm(m_mean))
        cos_dist = 1.0 - cos_sim
        
        diff_matrix = o_acts - m_acts
        _, s, _ = np.linalg.svd(diff_matrix, full_matrices=False)
        top1_variance_ratio = (s[0]**2) / np.sum(s**2) * 100.0
        
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
    
    rel_ranks = np.argsort([r["rel_diff"] for r in results])[::-1]
    best_layer = rel_ranks[0]
    
    for r in results:
        l = r["layer"]
        marker = ""
        if l == best_layer:
            marker = "★ BEST OVERVIEW J-SPACE (Max Relative Separation)"
        elif l == 26:
            marker = "(Clarity Singularity: Layer 26)"
        elif l in rel_ranks[:3]:
            marker = "★ Top 3"
            
        print(f"L{l:<4} | {r['diff_norm']:<10.2f} | {r['base_norm']:<10.2f} | {r['rel_diff']:<12.2f}% | {r['cos_dist']:<10.4f} | {r['top1_ratio']:<11.1f}% | {marker}")
        
    print("-----------------------------------------------------------------------------------------")
    print(f"\n🎯 Recommended Layer: Layer {best_layer} (Relative Separation: {results[best_layer]['rel_diff']:.2f}%, Top-1 SVD Ratio: {results[best_layer]['top1_ratio']:.1f}%)")
    
    # 26層との比較を出力
    l26 = results[26]
    print(f"ℹ️ Layer 26 Stats: Rel Sep: {l26['rel_diff']:.2f}%, Cos Dist: {l26['cos_dist']:.4f}, Top-1 Ratio: {l26['top1_ratio']:.1f}%")

if __name__ == "__main__":
    main()

