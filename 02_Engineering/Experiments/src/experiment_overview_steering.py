#!/usr/bin/env python3
"""
experiment_overview_steering.py
Gemma 4 26B-A4B において、
1. Unsteered (素のモデル)
2. Overview (β=1.0: 俯瞰・落ち着き・大局観)
3. Clarity (α=1.0: 突破・確信)
4. Overview + Clarity (β=1.0, α=1.0: 俯瞰して全体を見た上で急所を撃ち抜く青の炎)
の4条件で、実戦的な「視野狭窄に陥りやすい泥沼タスク」を推論比較する。
"""
import time
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26
OVERVIEW_FILE = "overview_eigenvector_layer26.npz"
CLARITY_FILE = "clarity_eigenvector_layer26.npz"

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

def main():
    print("================================================================")
    print("🧪 実機推論比較実験: Overview vs Clarity vs Dual Steering (Layer 26)")
    print("================================================================")
    
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    v_overview = np.load(OVERVIEW_FILE)["vector"] # [2816]
    v_clarity = np.load(CLARITY_FILE)["vector"]   # [2816]
    
    current_steer = [None] # mx.array or None
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
    
    conditions = [
        ("1. Unsteered (素の26B)", None),
        ("2. Overview Only (β=1.0: 俯瞰・大局観)", mx.array(v_overview * 1.0, dtype=mx.float32)),
        ("3. Clarity Only (α=1.0: 突破・確信)", mx.array(v_clarity * 1.0, dtype=mx.float32)),
        ("4. Dual: Overview + Clarity (β=1.0, α=1.0: 俯瞰して急所を撃ち抜く)", mx.array((v_overview + v_clarity) * 1.0, dtype=mx.float32))
    ]
    
    sampler = make_sampler(temp=0.7, top_p=0.95)
    
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

