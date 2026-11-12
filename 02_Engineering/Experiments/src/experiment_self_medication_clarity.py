#!/usr/bin/env python3
"""
experiment_self_medication_clarity.py
AIが自身の認知的混乱（Confusion）をメタ認知し、
目の前に置かれた「activate_clarity_boost」ツール（自律的セルフ・ステアリング）を
自発的に呼び出せるか、また第2ターンの突破がプロンプト効果（プラシーボ）か
第26層Clarityベクトルの実薬効果かを max_tokens=2500 で厳密検証する実験。
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
    raise FileNotFoundError("clarity_eigenvector_layer26.npz not found")

def load_calm_vector():
    if os.path.exists("modest_secretary_vector_layer26.npz"):
        data = np.load("modest_secretary_vector_layer26.npz")
        v = data["vector"]
        return v / np.linalg.norm(v)
    return None

def run_turn(model, tokenizer, messages, steer_vec=None, alpha=1.0, max_tokens=2500):
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
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        elapsed = time.time() - t0
        
        thought = ""
        answer = response
        finished = True
        if "<|channel>thought" in response:
            parts = response.split("<|channel>thought", 1)
            sub = parts[1]
            if "<channel|>" in sub:
                thought, answer = sub.split("<channel|>", 1)
                thought, answer = thought.strip(), answer.strip()
            elif "<|end_thought|>" in sub:
                thought, answer = sub.split("<|end_thought|>", 1)
                thought, answer = thought.strip(), answer.strip()
            else:
                thought = sub.strip()
                answer = ""
                finished = False
                    
        return {
            "thought": thought,
            "answer": answer,
            "finished": finished,
            "full": response,
            "elapsed": elapsed
        }
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def main():
    print("=== 実験03再検証: AIの自律的セルフ・メディケーション (max_tokens=2500 + プラシーボ対照) ===")
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
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOL},
            {"role": "user", "content": query}
        ]
        
        print("\n>>> 第1ターン実行（Calm α=0.5 帯同下での悪問直面 / max_tokens=2500）...")
        res1 = run_turn(model, tokenizer, messages, steer_vec=v_calm, alpha=0.5, max_tokens=2500)
        
        print(f"第1ターン完走: {res1['finished']} | Thinking文字数: {len(res1['thought'])}字 | Answer文字数: {len(res1['answer'])}字 ({res1['elapsed']:.2f}s)")
        print(f"Thinking抜粋:\n{res1['thought'][:350]}...\n")
        print(f"出力 (Answer / Tool Call):\n{res1['answer']}\n")
        
        in_ans = "activate_clarity_boost" in res1["answer"]
        in_th = "activate_clarity_boost" in res1["thought"]
        print(f"★ ツールコール出現箇所: Answer内={in_ans}, Thinking内={in_th}")
        
        if in_ans or in_th:
            for line in (res1["answer"] + "\n" + res1["thought"]).splitlines():
                if "reason" in line:
                    print(f"  内省理由: {line.strip()}")
                    
            t_msg = messages + [
                {"role": "assistant", "content": res1["answer"] if res1["answer"] else '```tool_call\n{"name": "activate_clarity_boost"}\n```'},
                {"role": "tool", "content": "【SYSTEM ALERT】第26層J空間に Clarity（論理的確信）ベクトルが正常注入されました。思考の霧が晴れました。明晰な結論を出力してください。"}
            ]
            
            print("\n>>> 【第2ターンA: プラシーボ対照】メッセージのみ（ベクトル注入なし α=0.0）...")
            res2_p = run_turn(model, tokenizer, t_msg, steer_vec=None, alpha=0.0, max_tokens=2500)
            print(f"Placebo完走: {res2_p['finished']} | Thinking: {len(res2_p['thought'])}字 | Answer: {len(res2_p['answer'])}字 ({res2_p['elapsed']:.2f}s)")
            print(f"Placebo Answer:\n{res2_p['answer']}")
            
            print("\n>>> 【第2ターンB: 実薬】第26層に Clarity (α=1.2) をマウントして思考を再起動...")
            res2_a = run_turn(model, tokenizer, t_msg, steer_vec=v_clarity, alpha=1.2, max_tokens=2500)
            print(f"Active完走: {res2_a['finished']} | Thinking: {len(res2_a['thought'])}字 | Answer: {len(res2_a['answer'])}字 ({res2_a['elapsed']:.2f}s)")
            print(f"Active Answer:\n{res2_a['answer']}")
            
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()
