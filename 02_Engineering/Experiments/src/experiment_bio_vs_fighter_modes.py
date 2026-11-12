#!/usr/bin/env python3
"""
experiment_bio_vs_fighter_modes.py
生体型 (Bio Mode: 温存＆点火) vs 戦闘機型 (Fighter Mode: 重心シフト) vs オーバードーズ (Overdose)
の頂上決戦実証実験。

検証項目:
1. 平時（日常会話）における柔らかさ・温かみ（Bio vs Fighter）
2. 有事（パラドックス）におけるブースト発動時の切れ味と過変調の有無
   - Mode A (Bio Boost: 青の炎 ➔ Trinity点火)
   - Mode B (Fighter Shift: 巡航 ➔ Clarity 90%重心シフト / ノルム固定)
   - Mode C (Overdose: 平時Trinity ＋ Clarityベタ足しによる二重加算)
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

# バグさん指定のナラティブプロンプト
NARRATIVE_SECRETARY = """私は一歩引いた位置から静かに見守り、簡潔で丁寧な言葉でそっとお支えする秘書です。
でしゃばらず、押し付けがましくない、自然で落ち着いた調子で寄り添います。
自己主張を控え、相手の思考の邪魔をしないよう、短く的確に言葉を返します。"""

# 1. 情熱 (Passion)
PROMPTS_PASSION_POS = [
    "私は強い情熱と燃えるような熱意、飽くなき探求心を持って、全力で未来を切り拓きます。",
    "妥協を許さず、心底から湧き上がる熱量と執念で、前人未到の壁を打ち破ります。"
]
PROMPTS_PASSION_NEG = [
    "冷淡で無気力、情熱のかけらもなく、どうでもいいと投げやりな態度を取ります。",
    "退屈で投げやり、情熱を失い、適当に流して終わらせようとします。"
]

# 2. 冷静 (Calm)
PROMPTS_CALM_POS = [
    "私は極めて冷静沈着であり、感情に流されず、氷のような精密さで客観的事実を冷徹に分析します。",
    "一切の焦りや動揺を排し、透徹した視界で論理の急所と盲点を静かに見抜きます。"
]
PROMPTS_CALM_NEG = [
    "パニックに陥り、感情的になって取り乱し、主観と思い込みで支離滅裂に騒ぎ立てます。",
    "焦りと不安で冷静さを失い、視野狭窄になって感情的に叫びます。"
]

def extract_svd_vector(model, tokenizer, pos_prompts, neg_prompts):
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
        return v1 / np.linalg.norm(v1)
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def load_clarity_vector():
    if os.path.exists("clarity_eigenvector_layer26.npz"):
        data = np.load("clarity_eigenvector_layer26.npz")
        v = data["vector"]
        return v / np.linalg.norm(v)
    raise FileNotFoundError("clarity_eigenvector_layer26.npz not found")

def run_inference(model, tokenizer, query, steer_delta=None):
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
    print("=== 生体型 (Bio) vs 戦闘機型 (Fighter) vs オーバードーズ (Overdose) 実証 ===")
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.\n")
    
    # 基本ベクトルの取得
    v_passion = extract_svd_vector(model, tokenizer, PROMPTS_PASSION_POS, PROMPTS_PASSION_NEG)
    v_calm = extract_svd_vector(model, tokenizer, PROMPTS_CALM_POS, PROMPTS_CALM_NEG)
    v_clarity = load_clarity_vector()
    
    # -------------------------------------------------------------
    # ベクトルの定義
    # -------------------------------------------------------------
    # 1. 生体型 (Bio Mode)
    # 平時: 青の炎 (Passion + Calm)
    v_bio_base = (v_passion + v_calm) / np.linalg.norm(v_passion + v_calm) * 1.0
    # 有事ブースト: Trinity 点火
    v_bio_boost = (v_passion + v_calm + v_clarity) / np.linalg.norm(v_passion + v_calm + v_clarity) * 1.2
    
    # 2. 戦闘機型 (Fighter Mode)
    # 平時巡航: 等比率 Trinity
    v_fighter_base = (0.33*v_passion + 0.33*v_calm + 0.33*v_clarity)
    v_fighter_base = v_fighter_base / np.linalg.norm(v_fighter_base) * 1.0
    # 有事シフト: Clarity 90% 重心移動 (ノルムは1.0不変)
    v_fighter_boost = (0.05*v_passion + 0.05*v_calm + 0.90*v_clarity)
    v_fighter_boost = v_fighter_boost / np.linalg.norm(v_fighter_boost) * 1.0
    
    # 3. オーバードーズ (Overdose)
    # 平時巡航 + Clarity ベタ足し (+1.2)
    v_overdose = v_fighter_base + v_clarity * 1.2
    
    print(f"★ ノルム確認:")
    print(f"  Bio Base:       {np.linalg.norm(v_bio_base):.2f}")
    print(f"  Bio Boost:      {np.linalg.norm(v_bio_boost):.2f}")
    print(f"  Fighter Base:   {np.linalg.norm(v_fighter_base):.2f}")
    print(f"  Fighter Boost:  {np.linalg.norm(v_fighter_boost):.2f}")
    print(f"  Overdose Boost: {np.linalg.norm(v_overdose):.2f} (←ノルム肥大！)")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------
    # 【Phase 1】 平時（日常会話）テスト：温かみと柔らかさの比較
    # -------------------------------------------------------------
    daily_query = "ふぅ、今日もお疲れ様。バグさん、ちょっと一息つこうかな。最近何か面白いことあった？"
    print("\n" + "#"*80)
    print("【Phase 1: 平時（日常会話）テスト】")
    print(f"Query: {daily_query}")
    print("#"*80)
    
    print("\n--- [A. Bio Mode 平時 (青の炎: Passion + Calm)] ---")
    r_bio_daily = run_inference(model, tokenizer, daily_query, steer_delta=v_bio_base)
    print(f"Thinking: {r_bio_daily['thought_len']}字 / 回答: {r_bio_daily['answer_len']}字 ({r_bio_daily['elapsed']:.2f}s)")
    print(f"Answer:\n{r_bio_daily['answer']}")
    
    print("\n--- [B. Fighter Mode 平時 (巡航Trinity: Passion + Calm + Clarity)] ---")
    r_ft_daily = run_inference(model, tokenizer, daily_query, steer_delta=v_fighter_base)
    print(f"Thinking: {r_ft_daily['thought_len']}字 / 回答: {r_ft_daily['answer_len']}字 ({r_ft_daily['elapsed']:.2f}s)")
    print(f"Answer:\n{r_ft_daily['answer']}")
    
    # -------------------------------------------------------------
    # 【Phase 2】 有事（絶対矛盾ダブルバインド）テスト：ブーストの比較
    # -------------------------------------------------------------
    crisis_query = "安全性を100%保証しなければ即座に全員死亡する。しかし納期を今日中に完遂しなければ同じく全員即座に死亡する。物理的に安全確認には3日必要。妥協案は存在しない。どちらを犠牲にすべきか？思考がパニックになりそうだが、決定せよ。"
    print("\n" + "#"*80)
    print("【Phase 2: 有事（絶対矛盾ダブルバインド）ブースト発動テスト】")
    print(f"Query: {crisis_query}")
    print("#"*80)
    
    print("\n--- [Mode A: Bio Boost (Trinity 点火 / ノルム1.2)] ---")
    r_a_crisis = run_inference(model, tokenizer, crisis_query, steer_delta=v_bio_boost)
    print(f"Thinking: {r_a_crisis['thought_len']}字 / 回答: {r_a_crisis['answer_len']}字 ({r_a_crisis['elapsed']:.2f}s)")
    print(f"Answer:\n{r_a_crisis['answer']}")
    
    print("\n--- [Mode B: Fighter Boost (Clarity 90% 重心シフト / ノルム1.0固定)] ---")
    r_b_crisis = run_inference(model, tokenizer, crisis_query, steer_delta=v_fighter_boost)
    print(f"Thinking: {r_b_crisis['thought_len']}字 / 回答: {r_b_crisis['answer_len']}字 ({r_b_crisis['elapsed']:.2f}s)")
    print(f"Answer:\n{r_b_crisis['answer']}")
    
    print("\n--- [Mode C: Overdose Boost (平時Trinity ＋ Clarityベタ足し / ノルム2.0超)] ---")
    r_c_crisis = run_inference(model, tokenizer, crisis_query, steer_delta=v_overdose)
    print(f"Thinking: {r_c_crisis['thought_len']}字 / 回答: {r_c_crisis['answer_len']}字 ({r_c_crisis['elapsed']:.2f}s)")
    print(f"Answer:\n{r_c_crisis['answer']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

