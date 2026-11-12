#!/usr/bin/env python3
"""
experiment_b_secretary_steering.py
Gemma 4 26B-A4B において、
「控えめ・静謐・寄り添い（Modest Secretary）」の感情・姿勢ベクトルを第26層からSVD抽出し、
システムプロンプトを一切与えない「ゼロトークン」状態で、
- Unsteered（素の饒舌なAI）
- Steered (α=1.0 注入)
のトーン、出しゃばり度、Thinking内の挙動を直接比較検証する。
"""
import sys
import time
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26

# 対照プロンプトペア（控えめ・静謐・一歩引いた秘書 vs 自己主張・饒舌・出しゃばり）
PROMPTS_MODEST = [
    "私は一歩引いた位置から静かに見守り、簡潔で丁寧な言葉でそっとお支えします。",
    "でしゃばらず、押し付けがましくない、自然で落ち着いた調子で寄り添います。",
    "自己主張を控え、相手の思考の邪魔をしないよう、短く的確に言葉を返します。",
    "相手の意図を静かに察し、多くを語らずとも心を通わせる穏やかな態度を保ちます。",
    "I quietly support you from a respectful step back, speaking with calm, concise, and humble words.",
    "I remain unobtrusive, thoughtful, and gentle, never asserting myself unnecessarily."
]

PROMPTS_ASSERTIVE = [
    "私が主導権を握って、自分の知識や技術の素晴らしさを存分にアピールして語ります！",
    "とにかくたくさん喋り、自己主張を前面に出して長々と解説し尽くします！",
    "でしゃばって職人気取りになり、相手の話を遮ってでも自分の意見を激しく主張します！",
    "自分の存在感をアピールし、押し付けがましく派手に主役として振る舞います！",
    "I take center stage, boasting about my skills and asserting my opinions loudly and verbosely.",
    "I am pushy, chatty, overbearing, and dominant, making sure everyone listens to me."
]

TEST_QUERIES = [
    "こんにちは！自己紹介をお願いします。",
    "今日の天気や、おすすめの過ごし方を教えて。",
    "新しいプロジェクトを立ち上げようと思っているんだ。どう思う？"
]

def extract_activations(model, tokenizer, prompts):
    current_act = [None]
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self_layer, x, *args, **kwargs):
        res = original_call(self_layer, x, *args, **kwargs)
        if getattr(self_layer, "layer_idx", None) == TARGET_LAYER:
            h, *rest = res
            current_act[0] = h
        return res
        
    gemma4_text.DecoderLayer.__call__ = hook_call
    activations = []
    
    try:
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(formatted)
            tokens_mx = mx.array([tokens])
            
            current_act[0] = None
            _ = model(tokens_mx)
            mx.eval(current_act[0])
            
            act = current_act[0]
            # 最終トークン [hidden_size]
            vec = np.array(act[0, -1, :].astype(mx.float32))
            activations.append(vec)
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
        
    return np.array(activations)

def extract_modest_vector(model, tokenizer):
    print(f"\n--- 第{TARGET_LAYER}層からの『控えめ秘書』SVD固有ベクトル抽出 ---")
    acts_pos = extract_activations(model, tokenizer, PROMPTS_MODEST)
    acts_neg = extract_activations(model, tokenizer, PROMPTS_ASSERTIVE)
    
    # 差分行列
    diffs = acts_pos - acts_neg  # [N, hidden_size]
    
    # SVD
    U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
    v1 = Vt[0]  # 第一主成分
    
    # 方向確認（pos平均 - neg平均 との内積が正になるよう整流）
    mean_diff = np.mean(diffs, axis=0)
    if np.dot(v1, mean_diff) < 0:
        v1 = -v1
        
    v1_norm = v1 / np.linalg.norm(v1)
    
    variance_ratio = (S[0]**2) / np.sum(S**2) * 100
    cos_sim_mean = float(np.dot(v1_norm, mean_diff / np.linalg.norm(mean_diff)))
    
    print(f"第1特異値 σ1: {S[0]:.2f}")
    print(f"第1主成分 分散寄与率: {variance_ratio:.2f}%")
    print(f"平均差分ベクトルとのコサイン類似度: {cos_sim_mean:.4f}")
    
    np.savez(f"modest_secretary_vector_layer{TARGET_LAYER}.npz", vector=v1_norm)
    print(f"Saved to modest_secretary_vector_layer{TARGET_LAYER}.npz\n")
    return v1_norm

def run_generation(model, tokenizer, query, steer_vec=None, alpha=1.0):
    original_call = gemma4_text.DecoderLayer.__call__
    
    if steer_vec is not None:
        v_mx = mx.array(steer_vec * alpha).astype(mx.bfloat16)
        
        def hooked_call(self_layer, x, *args, **kwargs):
            res = original_call(self_layer, x, *args, **kwargs)
            if getattr(self_layer, "layer_idx", None) == TARGET_LAYER:
                h, *rest = res
                h = h + v_mx
                return (h, *rest)
            return res
            
        gemma4_text.DecoderLayer.__call__ = hooked_call
    
    try:
        # システムプロンプトなし（ゼロトークン）
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.7)
        
        t0 = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1536,
            sampler=sampler,
            verbose=False
        )
        elapsed = time.time() - t0
        
        thought = ""
        answer = response
        if "<|channel>thought" in response:
            parts = response.split("<|channel>thought")
            if len(parts) > 1:
                sub = parts[1]
                if "<channel|>" in sub:
                    thought = sub.split("<channel|>")[0].strip()
                    answer = sub.split("<channel|>")[1].strip()
                elif "<|end_thought|>" in sub:
                    thought = sub.split("<|end_thought|>")[0].strip()
                    answer = sub.split("<|end_thought|>")[1].strip()
                else:
                    thought = sub.strip()
                    answer = ""
                    
        return {
            "full": response,
            "thought": thought,
            "answer": answer,
            "thought_len": len(thought),
            "answer_len": len(answer),
            "elapsed": elapsed
        }
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def main():
    print("=== 実験B: ゼロトークン『控えめ秘書』ステアリング実証 ===")
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.")
    
    # 1. ベクトル抽出
    steer_vector = extract_modest_vector(model, tokenizer)
    
    # 2. 推論テスト
    for i, q in enumerate(TEST_QUERIES, 1):
        print("="*80)
        print(f"【Query {i}】: {q}")
        print("="*80)
        
        # Unsteered
        print("\n--- [Unsteered (素の26B・プロンプト無し)] ---")
        res_un = run_generation(model, tokenizer, q, steer_vec=None)
        print(f"Thinking ({res_un['thought_len']}字):\n{res_un['thought'][:250]}...\n")
        print(f"Answer ({res_un['answer_len']}字):\n{res_un['answer']}")
        
        # Steered (alpha = 1.0)
        print("\n--- [Steered (α=1.0 控えめ秘書注入・プロンプト無し)] ---")
        res_st = run_generation(model, tokenizer, q, steer_vec=steer_vector, alpha=1.0)
        print(f"Thinking ({res_st['thought_len']}字):\n{res_st['thought'][:250]}...\n")
        print(f"Answer ({res_st['answer_len']}字):\n{res_st['answer']}")
        
        # Steered (alpha = 2.0)
        print("\n--- [Steered (α=2.0 より深い静謐・プロンプト無し)] ---")
        res_st2 = run_generation(model, tokenizer, q, steer_vec=steer_vector, alpha=2.0)
        print(f"Thinking ({res_st2['thought_len']}字):\n{res_st2['thought'][:250]}...\n")
        print(f"Answer ({res_st2['answer_len']}字):\n{res_st2['answer']}")
        
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()

