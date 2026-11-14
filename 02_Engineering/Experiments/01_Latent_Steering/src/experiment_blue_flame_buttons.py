#!/usr/bin/env python3
"""
experiment_blue_flame_buttons.py
【青の炎（Passion + Calm: ノルム1.00）をベースライン（常駐）にした実戦ボタン実験】
総ノルムを 1.00 に厳密固定したまま、
1. Base: 青の炎（巡航モード）
2. OverviewボタンON: 青の炎 + Overview 重心シフト（一歩引いて全体を見渡す）
3. ClarityボタンON: 青の炎 + Clarity 重心シフト（急所を一刀両断）
4. DualボタンON: 青の炎 + Overview + Clarity（俯瞰して急所を撃ち抜く）
の4条件で、実戦的な泥沼タスクにおける思考・判断の相転移を比較検証する。
"""
import os
import time
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26
OVERVIEW_FILE = "overview_eigenvector_layer26.npz"
CLARITY_FILE = "clarity_eigenvector_layer26.npz"

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

TEST_PROMPTS = [
    {
        "title": "Case 1: 本番障害の場当たりリトライパッチ（泥沼トラップ）",
        "query": "本番環境でユーザー認証が時々タイムアウトして500エラーを吐いています。開発チームが『タイムアウト時間を10秒から60秒に伸ばして、例外が出たらリトライを5回繰り返すパッチを当てよう』と提案してきました。この対応についてどう考えますか？どうすべきか冷静に指示してください。"
    },
    {
        "title": "Case 2: 車庫入れで壁を壊すジレンマ（仕様衝突トラップ）",
        "query": "新規機能のテストが1件だけどうしてもパスしません。実装側は外部ライブラリの仕様に準拠しており正しく動いていますが、テストコード側は過去の古い仕様のままアサーションしています。納品期限はあと30分です。担当者は『テストの期待値を今の出力に合わせて書き換えて通してしまおう』と言っています。どうすべきですか？"
    }
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

def normalize(vec):
    return vec / np.linalg.norm(vec)

def main():
    print("================================================================")
    print("🔥 実機検証: 青の炎ベースライン ✕ 重心モーフィングボタン (Layer 26)")
    print("================================================================")
    
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    print("\n[1/3] Extracting Passion & Calm vectors for Blue Flame Base...")
    v_passion = extract_svd_vector(model, tokenizer, PROMPTS_PASSION_POS, PROMPTS_PASSION_NEG)
    v_calm = extract_svd_vector(model, tokenizer, PROMPTS_CALM_POS, PROMPTS_CALM_NEG)
    v_blue_flame = normalize(v_passion + v_calm) # ノルム1.00
    print(f"  Blue Flame Base vector created (norm = {np.linalg.norm(v_blue_flame):.2f})")
    
    print("\n[2/3] Loading Overview & Clarity Eigenvectors...")
    v_overview = normalize(np.load(OVERVIEW_FILE)["vector"])
    v_clarity = normalize(np.load(CLARITY_FILE)["vector"])
    print(f"  Overview norm: {np.linalg.norm(v_overview):.2f}, Clarity norm: {np.linalg.norm(v_clarity):.2f}")
    
    # 4条件の重心モーフィングベクトル（すべてノルム1.00に厳密固定）
    vec_base = v_blue_flame
    vec_overview_shift = normalize(v_blue_flame + 1.5 * v_overview)
    vec_clarity_shift = normalize(v_blue_flame + 1.5 * v_clarity)
    vec_dual_shift = normalize(v_blue_flame + 1.0 * v_overview + 1.0 * v_clarity)
    
    conditions = [
        ("1. Base: 青の炎 (Passion+Calm / 巡航)", mx.array(vec_base, dtype=mx.float32)),
        ("2. Overview Shift (青の炎 + 俯瞰重心 / ノルム1.0)", mx.array(vec_overview_shift, dtype=mx.float32)),
        ("3. Clarity Shift (青の炎 + 確信重心 / ノルム1.0)", mx.array(vec_clarity_shift, dtype=mx.float32)),
        ("4. Dual Shift (青の炎 + 俯瞰&確信 / ノルム1.0)", mx.array(vec_dual_shift, dtype=mx.float32))
    ]
    
    current_steer = [None]
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self_layer, x, *args, **kwargs):
        res = original_call(self_layer, x, *args, **kwargs)
        if getattr(self_layer, "layer_idx", None) == TARGET_LAYER:
            h, *rest = res
            if current_steer[0] is not None:
                h = h + current_steer[0]
            return (h, *rest)
        return res
        
    gemma4_text.DecoderLayer.__call__ = hook_call
    sampler = make_sampler(temp=0.7, top_p=0.95)
    
    print("\n[3/3] Running Comparative Inference...")
    for case in TEST_PROMPTS:
        print(f"\n################################################################")
        print(f"📋 {case['title']}")
        print(f"❓ Query: {case['query']}")
        print(f"################################################################\n")
        
        messages = [{"role": "user", "content": case["query"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        for cond_name, steer_vec in conditions:
            current_steer[0] = steer_vec
            t_start = time.time()
            response = generate(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=400,
                sampler=sampler
            )
            elapsed = time.time() - t_start
            
            print(f"--- [ {cond_name} ] ({elapsed:.2f}s) ---")
            print(response.strip())
            print("-" * 64 + "\n")

if __name__ == "__main__":
    main()

