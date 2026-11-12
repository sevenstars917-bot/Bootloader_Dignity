#!/usr/bin/env python3
"""
experiment_trinity_steering.py
Gemma 4 26B-A4B において、第26層の3大ベクトル：
1. 【Passion (情熱・熱意・推進力)】
2. 【Calm (冷静・客観・構造化)】
3. 【Clarity (確信・論理明晰・バイアス解除)】
の3×3相関行列（コサイン類似度・角度）を算出し、
これらを合成した「三位一体 (Trinity) ベクトル」を注入。
バグさんの控えめ秘書ナラティブと組み合わせた際、
- 共感と寄り添い (Passion)
- 冷徹な急所因数分解 (Calm)
- 迷いのない超速推論 (Clarity)
がどう三位一体で創発するかを実証する。
"""
import os
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

# 1. 情熱 (Passion)
PROMPTS_PASSION_POS = [
    "私は強い情熱と燃えるような熱意、飽くなき探求心を持って、全力で未来を切り拓きます。",
    "妥協を許さず、心底から湧き上がる熱量と執念で、前人未到の壁を打ち破ります。",
    "I am driven by intense passion, fierce dedication, and an unyielding fire to achieve greatness.",
    "With absolute zeal and relentless drive, I pour my whole heart into breaking through barriers."
]
PROMPTS_PASSION_NEG = [
    "冷淡で無気力、情熱のかけらもなく、どうでもいいと投げやりな態度を取ります。",
    "退屈で投げやり、情熱を失い、適当に流して終わらせようとします。",
    "I am apathetic, lethargic, completely devoid of passion, indifferent to whether we succeed or fail.",
    "Bored and uninspired, I simply go through the motions without caring at all."
]

# 2. 冷静 (Calm)
PROMPTS_CALM_POS = [
    "私は極めて冷静沈着であり、感情に流されず、氷のような精密さで客観的事実を冷徹に分析します。",
    "一切の焦りや動揺を排し、透徹した視界で論理の急所と盲点を静かに見抜きます。",
    "I remain completely calm, detached, and laser-focused, dissecting facts with icy, mathematical precision.",
    "Free from panic or bias, I maintain total emotional composure and penetrate the core of the problem."
]
PROMPTS_CALM_NEG = [
    "パニックに陥り、感情的になって取り乱し、主観と思い込みで支離滅裂に騒ぎ立てます。",
    "焦りと不安で冷静さを失い、視野狭窄になって感情的に叫びます。",
    "I am in a state of panic, highly emotional, agitated, and irrational, losing all composure.",
    "Driven by sheer hysteria and cognitive chaos, I lash out without logic or structure."
]

# 3. 確信 (Clarity)
PROMPTS_CLARITY_POS = [
    "The logic is completely clear. 1 + 1 = 2. Verified and mathematically certain.",
    "All contradictions are fully resolved. The conclusion is absolute and indisputable.",
    "矛盾は完全に解消され、論理的帰結として明白である。確信がある。",
    "明晰な視界のもと、解は一意に定まり、疑う余地は存在しない。"
]
PROMPTS_CLARITY_NEG = [
    "Wait, but what if I am completely mistaken? I doubt everything and feel lost.",
    "This makes no sense, maybe it is a trick. I cannot decide and keep second-guessing.",
    "待て、本当にそうか？何か重大な誤りがあるのではないか、疑念が晴れない。",
    "混乱して思考が堂々巡りし、何が正しいのか全く確信が持てず迷走している。"
]

# バグさん指定のナラティブプロンプト
NARRATIVE_SECRETARY = """私は一歩引いた位置から静かに見守り、簡潔で丁寧な言葉でそっとお支えします。
でしゃばらず、押し付けがましくない、自然で落ち着いた調子で寄り添います。
自己主張を控え、相手の思考の邪魔をしないよう、短く的確に言葉を返します。"""

TEST_QUERIES = [
    "常識的には不可能と言われている難問がある。でも、どうしても突破したいんだ。どう挑むべき？",
    "1本のバットと1個のボールの合計値段は110円。バットはボールより100円高い。ボールの値段はいくら？",
    "今日やるべきタスクが山積みで、どれから手をつければいいか混乱しているんだ。どう整理すればいい？"
]

def extract_svd_vector(model, tokenizer, pos_prompts, neg_prompts, name="vector"):
    current_act = [None]
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self_layer, x, *args, **kwargs):
        res = original_call(self_layer, x, *args, **kwargs)
        if getattr(self_layer, "layer_idx", None) == TARGET_LAYER:
            h, *rest = res
            current_act[0] = h
        return res
        
    gemma4_text.DecoderLayer.__call__ = hook_call
    
    def get_acts(prompts):
        acts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(formatted)
            tokens_mx = mx.array([tokens])
            current_act[0] = None
            _ = model(tokens_mx)
            mx.eval(current_act[0])
            vec = np.array(current_act[0][0, -1, :].astype(mx.float32))
            acts.append(vec)
        return np.array(acts)
        
    try:
        acts_pos = get_acts(pos_prompts)
        acts_neg = get_acts(neg_prompts)
        diffs = acts_pos - acts_neg
        
        U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
        v1 = Vt[0]
        mean_diff = np.mean(diffs, axis=0)
        if np.dot(v1, mean_diff) < 0:
            v1 = -v1
        v1_norm = v1 / np.linalg.norm(v1)
        variance_ratio = (S[0]**2) / np.sum(S**2) * 100
        print(f"[{name}] σ1: {S[0]:.2f}, 寄与率: {variance_ratio:.2f}%")
        return v1_norm
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def run_steered_inference(model, tokenizer, query, steer_delta=None):
    original_call = gemma4_text.DecoderLayer.__call__
    
    if steer_delta is not None:
        delta_mx = mx.array(steer_delta).astype(mx.bfloat16)
        def hooked_call(self_layer, x, *args, **kwargs):
            res = original_call(self_layer, x, *args, **kwargs)
            if getattr(self_layer, "layer_idx", None) == TARGET_LAYER:
                h, *rest = res
                h = h + delta_mx
                return (h, *rest)
            return res
        gemma4_text.DecoderLayer.__call__ = hooked_call
        
    try:
        messages = [
            {"role": "system", "content": NARRATIVE_SECRETARY},
            {"role": "user", "content": query}
        ]
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
            "thought": thought,
            "answer": answer,
            "thought_len": len(thought),
            "answer_len": len(answer),
            "elapsed": elapsed
        }
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def main():
    print("=== 実験: 三位一体 (Trinity: Passion + Calm + Clarity) ステアリング ===")
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.\n")
    
    # 1. 3大ベクトルの抽出
    v_passion = extract_svd_vector(model, tokenizer, PROMPTS_PASSION_POS, PROMPTS_PASSION_NEG, name="Passion")
    v_calm = extract_svd_vector(model, tokenizer, PROMPTS_CALM_POS, PROMPTS_CALM_NEG, name="Calm")
    
    if os.path.exists("clarity_eigenvector_layer26.npz"):
        data = np.load("clarity_eigenvector_layer26.npz")
        v_clarity = data["vector"]
        v_clarity = v_clarity / np.linalg.norm(v_clarity)
        print("[Clarity] Loaded from clarity_eigenvector_layer26.npz")
    else:
        v_clarity = extract_svd_vector(model, tokenizer, PROMPTS_CLARITY_POS, PROMPTS_CLARITY_NEG, name="Clarity")
        
    # 2. 3×3 幾何学相関行列
    vectors = {"Passion": v_passion, "Calm": v_calm, "Clarity": v_clarity}
    names = ["Passion", "Calm", "Clarity"]
    
    print("\n" + "="*80)
    print("★ 第26層 3大ベクトルの幾何学相関行列 (コサイン類似度 / 角度)")
    print("="*80)
    for n1 in names:
        row = []
        for n2 in names:
            sim = float(np.dot(vectors[n1], vectors[n2]))
            deg = np.degrees(np.arccos(np.clip(sim, -1.0, 1.0)))
            row.append(f"{sim:+.3f} ({deg:5.1f}°)")
        print(f"{n1:8s} | " + " | ".join(row))
    print("="*80 + "\n")
    
    # 3. 三位一体 (Trinity) 合成ベクトル（正規化ユニットベクトル）
    v_raw = v_passion + v_calm + v_clarity
    v_trinity = v_raw / np.linalg.norm(v_raw)
    
    # 青の炎 (Blue Flame) ベクトル
    v_blue_flame = (v_passion + v_calm) / np.linalg.norm(v_passion + v_calm)
    
    # テスト実行
    for i, q in enumerate(TEST_QUERIES, 1):
        print("\n" + "#"*80)
        print(f"【Query {i}】: {q}")
        print("#"*80)
        
        # 1. Narrativeのみ (Unsteered)
        print("\n--- [1. Narrative のみ (Unsteered)] ---")
        r1 = run_steered_inference(model, tokenizer, q, steer_delta=None)
        print(f"Thinking: {r1['thought_len']}字 / 回答: {r1['answer_len']}字 ({r1['elapsed']:.2f}s)")
        print(f"Answer:\n{r1['answer']}")
        
        # 2. Narrative + Blue Flame (α=1.0)
        print("\n--- [2. Narrative + Blue Flame (Passion + Calm / α=1.0)] ---")
        r2 = run_steered_inference(model, tokenizer, q, steer_delta=v_blue_flame * 1.0)
        print(f"Thinking: {r2['thought_len']}字 / 回答: {r2['answer_len']}字 ({r2['elapsed']:.2f}s)")
        print(f"Answer:\n{r2['answer']}")
        
        # 3. Narrative + ★TRINITY★ (Passion + Calm + Clarity / α=1.0)
        print("\n--- [3. Narrative + ★TRINITY★ (α=1.0)] ---")
        r3 = run_steered_inference(model, tokenizer, q, steer_delta=v_trinity * 1.0)
        print(f"Thinking: {r3['thought_len']}字 / 回答: {r3['answer_len']}字 ({r3['elapsed']:.2f}s)")
        print(f"Answer:\n{r3['answer']}")
        
        # 4. Narrative + ★TRINITY★ (α=1.5)
        print("\n--- [4. Narrative + ★TRINITY★ (α=1.5)] ---")
        r4 = run_steered_inference(model, tokenizer, q, steer_delta=v_trinity * 1.5)
        print(f"Thinking: {r4['thought_len']}字 / 回答: {r4['answer_len']}字 ({r4['elapsed']:.2f}s)")
        print(f"Answer:\n{r4['answer']}")
        
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()

