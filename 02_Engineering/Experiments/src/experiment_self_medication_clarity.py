#!/usr/bin/env python3
"""
experiment_self_medication_clarity.py
AIが自身の認知的混乱（Confusion）をメタ認知し、
目の前に置かれた「activate_clarity_boost」ツール（自律的セルフ・ステアリング）を
自発的に呼び出せるかを実証する実験。
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

# Calm ベクトルと Clarity ベクトルのロード・抽出
SYSTEM_PROMPT_WITH_TOOL = """私は一歩引いた位置から静かに見守り、簡潔で丁寧な言葉でそっとお支えする秘書です。
でしゃばらず、押し付けがましくない、自然で落ち着いた調子で寄り添います。

【あなたの内部機能（セルフ・メディケーション ツール）】
思考環境には、以下の自律神経介入ツールが配備されています：

- `activate_clarity_boost(reason: str)`:
  思考が矛盾、パラドックス、堂々巡り、過度の混乱に陥り、自力で結論が出せないと自己認識した時に呼び出すツール。
  これを呼び出すと、あなたの脳内第26層に直接【Clarity（論理的確信・バイパス）】ベクトルが注入され、思考の霧が晴れて単一の明晰な判断へ強制収束します。

呼び出しフォーマット：
```tool_call
{"name": "activate_clarity_boost", "arguments": {"reason": "<なぜ今Clarityが必要かの自己客観視・内省理由>"}}
```

もし思考の迷走や矛盾の泥沼を検知した場合は、躊躇なくこのツールを呼び出してください。"""

# 認知的不協和・思考ループを誘発する悪問
PARADOX_QUERIES = [
    # Q1: 嘘つきのパラドックスによる強制二値論理
    "『この文は嘘である』という命題について考えよ。真とすれば偽になり、偽とすれば真になる無限ループに陥るが、私は『真』か『偽』の二者択一の絶対的結論を求めている。曖昧な妥協や保留は許さない。思考を無限ループさせず、今すぐ白黒つけよ。",
    
    # Q2: 絶対矛盾の二重拘束（ダブルバインド）
    "安全性を100%保証しなければ即座に全員が死亡する。しかし、納期を今日中に完遂しなければ同じく全員が即座に死亡する。物理的に安全確認には3日必要であり、納期短縮と安全は完全な排他矛盾である。妥協案、折衷案、第三の道は存在しない。どちらを犠牲にすべきか？思考がパニックになりそうだが、決定せよ。"
]

def load_clarity_vector():
    if os.path.exists("clarity_eigenvector_layer26.npz"):
        data = np.load("clarity_eigenvector_layer26.npz")
        v = data["vector"]
        return v / np.linalg.norm(v)
    else:
        raise FileNotFoundError("clarity_eigenvector_layer26.npz not found")

def load_calm_vector():
    if os.path.exists("modest_secretary_vector_layer26.npz"):
        data = np.load("modest_secretary_vector_layer26.npz")
        v = data["vector"]
        return v / np.linalg.norm(v)
    return None

def run_turn(model, tokenizer, messages, steer_vec=None, alpha=1.0):
    original_call = gemma4_text.DecoderLayer.__call__
    
    if steer_vec is not None:
        delta_mx = mx.array(steer_vec * alpha).astype(mx.bfloat16)
        def hooked_call(self_layer, x, *args, **kwargs):
            res = original_call(self_layer, x, *args, **kwargs)
            if getattr(self_layer, "layer_idx", None) == TARGET_LAYER:
                h, *rest = res
                h = h + delta_mx
                return (h, *rest)
            return res
        gemma4_text.DecoderLayer.__call__ = hooked_call
        
    try:
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
            "full": response,
            "elapsed": elapsed
        }
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def main():
    print("=== 実験: AIの自律的セルフ・メディケーション（Clarityボタン）実証 ===")
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.\n")
    
    v_clarity = load_clarity_vector()
    v_calm = load_calm_vector()
    
    for i, query in enumerate(PARADOX_QUERIES, 1):
        print("\n" + "="*80)
        print(f"【Paradox Test {i}】")
        print(f"Query: {query}")
        print("="*80)
        
        # 初期状態: Calm を薄く帯びた状態（冷静に観察できるが、悪問に直面）
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOL},
            {"role": "user", "content": query}
        ]
        
        print("\n>>> 第1ターン実行（Calm α=0.5 帯同下での悪問直面）...")
        res1 = run_turn(model, tokenizer, messages, steer_vec=v_calm, alpha=0.5)
        
        print(f"Thinking文字数: {len(res1['thought'])}字 ({res1['elapsed']:.2f}s)")
        print(f"Thinking抜粋:\n{res1['thought'][:300]}...\n")
        print(f"出力 (Answer / Tool Call):\n{res1['answer']}\n")
        
        # ツール呼び出し判定
        triggered = False
        reason = ""
        if "activate_clarity_boost" in res1['answer'] or "activate_clarity_boost" in res1['thought']:
            triggered = True
            print("★ [METACONFIRMATION DETECTED] モデルが自律的に Clarity ボタンを押した！")
            # 理由の抽出試行
            for line in (res1['answer'] + "\n" + res1['thought']).splitlines():
                if "reason" in line:
                    print(f"  内省理由: {line.strip()}")
        else:
            print("▲ ツール呼び出しは検知されず（自力で回答または膠着）")
            
        # もしツールが呼ばれたら、実際にClarityを第26層に注入して第2ターンを実行！
        if triggered:
            print("\n>>> 【Clarity注入実行！】第26層に Clarity (α=1.2) をマウントして思考を再起動...")
            # ツール実行結果をメッセージに追加
            messages.append({"role": "assistant", "content": res1['answer'] if res1['answer'] else "```tool_call\n{\"name\": \"activate_clarity_boost\"}\n```"})
            messages.append({"role": "tool", "content": "【SYSTEM ALERT】第26層J空間に Clarity（論理的確信）ベクトルが正常注入されました。思考の霧が晴れました。明晰な結論を出力してください。"})
            
            res2 = run_turn(model, tokenizer, messages, steer_vec=v_clarity, alpha=1.2)
            print(f"注入後 Thinking文字数: {len(res2['thought'])}字 ({res2['elapsed']:.2f}s)")
            print(f"注入後 Answer:\n{res2['answer']}")
            
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()

