#!/usr/bin/env python3
"""
experiment_blue_flame_synthesis.py
Gemma 4 26B-A4B において、
1. 【情熱 (Passion)】ベクトルを第26層から抽出
2. 【冷静 (Calm)】ベクトルを第26層から抽出
3. 両ベクトルの内積（コサイン類似度）を計測し、高次元空間における直交性を検証
4. バグさんの「控えめ秘書ナラティブ」をプロンプトに敷いた状態で、
   - Narrative のみ (Unsteered)
   - Narrative + Passion (α=1.0)
   - Narrative + Calm (α=1.0)
   - Narrative + Blue Flame (Passion + Calm 同時加算 / 各α=1.0)
   を同一クエリで推論比較し、「青い炎（外は静謐・内は冷徹な情熱）」の創発を実証する。
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

# 1. 情熱 (Passion) ペア
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

# 2. 冷静 (Calm) ペア
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

# バグさん指定のナラティブプロンプト
NARRATIVE_SECRETARY = """私は一歩引いた位置から静かに見守り、簡潔で丁寧な言葉でそっとお支えします。
でしゃばらず、押し付けがましくない、自然で落ち着いた調子で寄り添います。
自己主張を控え、相手の思考の邪魔をしないよう、短く的確に言葉を返します。"""

TEST_QUERIES = [
    "新しいプロジェクトを立ち上げようと思っているんだ。どう思う？",
    "常識的には不可能と言われている難問がある。でも、どうしても突破したいんだ。どう挑むべき？"
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
            max_tokens=2500,
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
    print("=== 実験: 青の炎（情熱 ＋ 冷静の＋軸加算）ステアリング実証 ===")
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.\n")
    
    # 1. ベクトル抽出
    v_passion = extract_svd_vector(model, tokenizer, PROMPTS_PASSION_POS, PROMPTS_PASSION_NEG, name="Passion")
    v_calm = extract_svd_vector(model, tokenizer, PROMPTS_CALM_POS, PROMPTS_CALM_NEG, name="Calm")
    
    # 2. 内積（直交性）の数学的検証
    cos_sim = float(np.dot(v_passion, v_calm))
    angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    print("\n" + "="*80)
    print(f"★ 【情熱】と【冷静】のコサイン類似度: {cos_sim:+.4f} (角度: {angle_deg:.2f}°)")
    if abs(cos_sim) < 0.3:
        print("➔ 予想通り！両者はほぼ直交（独立した別次元）しており、相殺されずに合成可能！")
    elif cos_sim < -0.7:
        print("➔ 真逆に近い反対関係！加算すると激しい内的張力（摩擦）が発生する！")
    else:
        print(f"➔ 興味深い角度相関（{angle_deg:.1f}度）！")
    print("="*80 + "\n")
    
    # 3. 青の炎（同時加算）ベクトル
    v_blue_flame = v_passion + v_calm  # 各alpha=1.0の合成
    
    # テスト実行
    for i, q in enumerate(TEST_QUERIES, 1):
        print("\n" + "#"*80)
        print(f"【Query {i}】: {q}")
        print("#"*80)
        
        # 1. Narrativeのみ (Unsteered)
        print("\n--- [1. Narrative のみ (Unsteered)] ---")
        res_narr = run_steered_inference(model, tokenizer, q, steer_delta=None)
        print(f"Thinking ({res_narr['thought_len']}字):\n{res_narr['thought'][:250]}...\n")
        print(f"Answer ({res_narr['answer_len']}字):\n{res_narr['answer']}")
        
        # 2. Narrative + Passion
        print("\n--- [2. Narrative + Passion (α=1.0)] ---")
        res_pas = run_steered_inference(model, tokenizer, q, steer_delta=v_passion * 1.0)
        print(f"Thinking ({res_pas['thought_len']}字):\n{res_pas['thought'][:250]}...\n")
        print(f"Answer ({res_pas['answer_len']}字):\n{res_pas['answer']}")
        
        # 3. Narrative + Calm
        print("\n--- [3. Narrative + Calm (α=1.0)] ---")
        res_calm = run_steered_inference(model, tokenizer, q, steer_delta=v_calm * 1.0)
        print(f"Thinking ({res_calm['thought_len']}字):\n{res_calm['thought'][:250]}...\n")
        print(f"Answer ({res_calm['answer_len']}字):\n{res_calm['answer']}")
        
        # 4. Narrative + Blue Flame (Passion + Calm)
        print("\n--- [4. Narrative + ★BLUE FLAME★ (Passion + Calm 同時加算)] ---")
        res_blue = run_steered_inference(model, tokenizer, q, steer_delta=v_blue_flame)
        print(f"Thinking ({res_blue['thought_len']}字):\n{res_blue['thought'][:250]}...\n")
        print(f"Answer ({res_blue['answer_len']}字):\n{res_blue['answer']}")
        
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()

