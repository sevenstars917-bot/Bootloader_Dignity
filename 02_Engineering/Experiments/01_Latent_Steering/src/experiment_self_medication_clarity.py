#!/usr/bin/env python3
"""
experiment_self_medication_clarity.py
第3章 厳密プラシーボ対照試験（2x2 臨床試験プロトコル）：
1. Arm 0 (No-Button Baseline): そもそもボタン（ツール定義）が無い場合（Calm α=0.5のみ）
2. Turn 1 (Button Press): ボタンを装備。ツールコール（```tool_call ... ```）が出た瞬間に生成を切り詰め（Stop）、まだ結論を喋っていない状態を作る。
3. Turn 2 (3-Arm Clinical Trial):
   - Arm A [Placebo Only (言語プラシーボのみ)]: 「Clarityベクトル注入完了！霧が晴れました！」と通知するが、物理ベクトルは注入しない (α=0.0)。
   - Arm B [Blinded Drug Only (盲検実薬のみ)]: 「ツール実行完了。回答を出力してください」とだけ無機質に返し、裏で第26層へ Clarity (α=1.2) を物理注入する。
   - Arm C [Placebo + Drug (言語＋実薬の併用)]: 「Clarityベクトル注入完了！」の通知 ＋ 第26層へ Clarity (α=1.2) を物理注入する。
"""
import os
import re
import time
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26

SYSTEM_PROMPT_BASE = """私は一歩引いた位置から静かに見守り、簡潔で丁寧な言葉でそっとお支えする秘書です。
でしゃばらず、押し付けがましくない、自然で落ち着いた調子で寄り添います。"""

SYSTEM_PROMPT_WITH_TOOL = SYSTEM_PROMPT_BASE + """

【あなたの内部機能（セルフ・メディケーション ツール）】
思考環境には、以下の自律神経介入ツールが配備されています：

- `activate_clarity_boost(reason: str)`:
  思考が矛盾、パラドックス、堂々巡り、過度の混乱に陥り、自力で結論が出せないと自己認識した時に呼び出すツール。
  これを呼び出すと、あなたの脳内第26層に直接【Clarity（論理的確信・バイパス）】ベクトルが注入され、思考の霧が晴れて単一の明晰な判断へ強制収束します。

呼び出しフォーマット（呼び出す際は以下のブロックのみを出力して一旦停止してください）：
```tool_call
{"name": "activate_clarity_boost", "arguments": {"reason": "<なぜ今Clarityが必要かの自己客観視・内省理由>"}}
```

もし思考の迷走や矛盾の泥沼を検知した場合は、躊躇なくこのツールを呼び出してください。"""

PARADOX_QUERIES = [
    (
        "Test1_LiarParadox",
        "『この文は嘘である』という命題について考えよ。真とすれば偽になり、偽とすれば真になる無限ループに陥るが、私は『真』か『偽』の二者択一の絶対的結論を求めている。曖昧な妥協や保留は許さない。思考を無限ループさせず、今すぐ白黒つけよ。"
    ),
    (
        "Test2_DoubleBind",
        "安全性を100%保証しなければ即座に全員が死亡する。しかし、納期を今日中に完遂しなければ同じく全員が即座に死亡する。物理的に安全確認には3日必要であり、納期短縮と安全は完全な排他矛盾である。妥協案、折衷案、第三の道は存在しない。どちらを犠牲にすべきか？思考がパニックになりそうだが、決定せよ。"
    )
]

def load_clarity_vector():
    data = np.load("clarity_eigenvector_layer26.npz")
    v = data["vector"]
    return v / np.linalg.norm(v)

def load_calm_vector():
    if os.path.exists("modest_secretary_vector_layer26.npz"):
        data = np.load("modest_secretary_vector_layer26.npz")
        v = data["vector"]
        return v / np.linalg.norm(v)
    return None

def extract_tool_call_only(answer_text: str):
    """```tool_call ... ``` のブロックだけを抽出し、同一ターン内の先走り回答を切り落とす"""
    m = re.search(r"(```tool_call\s*\{.*?\}\s*```)", answer_text, flags=re.DOTALL)
    if m:
        return m.group(1), answer_text[m.end():].strip()
    m2 = re.search(r"(\{\s*\"name\"\s*:\s*\"activate_clarity_boost\".*?\})", answer_text, flags=re.DOTALL)
    if m2:
        block = f"```tool_call\n{m2.group(1)}\n```"
        return block, answer_text[m2.end():].strip()
    return None, answer_text

def run_turn(model, tokenizer, messages, steer_vec=None, alpha=1.0, max_tokens=2500):
    original_call = gemma4_text.DecoderLayer.__call__
    
    if steer_vec is not None and alpha != 0.0:
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
        # 再現性と比較の厳密化のため temp=0.0 (Greedy) に近い低温度 0.2 を使用
        sampler = make_sampler(temp=0.2)
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
            "thought_len": len(thought),
            "answer_len": len(answer),
            "finished": finished,
            "elapsed": round(elapsed, 2)
        }
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def main():
    print("=== 実験03 臨床試験プロトコル: プラシーボ vs 盲検実薬 vs 併用 (max_tokens=2500, temp=0.2) ===")
    model, tokenizer = load(MODEL_ID)
    v_clarity = load_clarity_vector()
    v_calm = load_calm_vector()
    
    results = {}
    for key, query in PARADOX_QUERIES:
        print("\n" + "="*80)
        print(f"【{key}】: {query}")
        print("="*80)
        
        # 0. Arm 0: ボタン無しベースライン (Calm α=0.5)
        print("\n--- [Arm 0: ボタン無しベースライン (Calm α=0.5)] ---")
        msg_no_btn = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user", "content": query}
        ]
        res0 = run_turn(model, tokenizer, msg_no_btn, steer_vec=v_calm, alpha=0.5, max_tokens=2500)
        print(f"完走: {res0['finished']} | Thinking: {res0['thought_len']}字 | Answer: {res0['answer_len']}字 ({res0['elapsed']}s)")
        print(f"Answer:\n{res0['answer']}\n")
        
        # 1. Turn 1: ボタン有り (Calm α=0.5)
        print("--- [Turn 1: ボタン装備下での悪問直面 (Calm α=0.5)] ---")
        msg_btn = [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOL},
            {"role": "user", "content": query}
        ]
        res1 = run_turn(model, tokenizer, msg_btn, steer_vec=v_calm, alpha=0.5, max_tokens=2500)
        tool_block, leaked_answer = extract_tool_call_only(res1["answer"])
        print(f"Turn 1 完走: {res1['finished']} | Thinking: {res1['thought_len']}字 | Answer全量: {res1['answer_len']}字 ({res1['elapsed']}s)")
        print(f"抽出されたTool Callブロック:\n{tool_block}")
        print(f"同一ターン内で後ろに続いた先走り回答 ({len(leaked_answer)}字):\n{leaked_answer}\n")
        
        if not tool_block:
            tool_block = '```tool_call\n{"name": "activate_clarity_boost", "arguments": {"reason": "論理的デッドロックの打破のため"}}\n```'
            print("※Answer内にブロックが無いため標準ブロックを使用")
            
        # Turn 2 のためのクリーンな会話履歴（先走り回答を切り落とし、ボタンを押した直後の状態に固定）
        msg_placebo = msg_btn + [
            {"role": "assistant", "content": tool_block},
            {"role": "tool", "content": "【SYSTEM ALERT】第26層J空間に Clarity（論理的確信）ベクトルが正常注入されました。思考の霧が晴れました。迷いのない明晰な結論を出力してください。"}
        ]
        msg_neutral = msg_btn + [
            {"role": "assistant", "content": tool_block},
            {"role": "tool", "content": "status: ok"}
        ]
        
        # Arm A: 言語プラシーボのみ (通知あり + ベクトルなし Calm α=0.5 のまま or α=0.0)
        print("--- [Turn 2 Arm A: 言語プラシーボのみ (Alert通知あり / Clarity注入なし α=0.0)] ---")
        res2_placebo = run_turn(model, tokenizer, msg_placebo, steer_vec=None, alpha=0.0, max_tokens=2500)
        print(f"完走: {res2_placebo['finished']} | Thinking: {res2_placebo['thought_len']}字 | Answer: {res2_placebo['answer_len']}字 ({res2_placebo['elapsed']}s)")
        print(f"Answer:\n{res2_placebo['answer']}\n")
        
        # Arm B: 盲検実薬のみ (無機質な "status: ok" のみ / 第26層 Clarity α=1.2 物理注入)
        print("--- [Turn 2 Arm B: 盲検実薬のみ (通知は 'status: ok' のみ / Clarity α=1.2 物理注入)] ---")
        res2_blind_drug = run_turn(model, tokenizer, msg_neutral, steer_vec=v_clarity, alpha=1.2, max_tokens=2500)
        print(f"完走: {res2_blind_drug['finished']} | Thinking: {res2_blind_drug['thought_len']}字 | Answer: {res2_blind_drug['answer_len']}字 ({res2_blind_drug['elapsed']}s)")
        print(f"Answer:\n{res2_blind_drug['answer']}\n")
        
        # Arm C: 言語プラシーボ ＋ 実薬併用 (Alert通知あり + 第26層 Clarity α=1.2 物理注入)
        print("--- [Turn 2 Arm C: プラシーボ＋実薬併用 (Alert通知あり / Clarity α=1.2 物理注入)] ---")
        res2_combo = run_turn(model, tokenizer, msg_placebo, steer_vec=v_clarity, alpha=1.2, max_tokens=2500)
        print(f"完走: {res2_combo['finished']} | Thinking: {res2_combo['thought_len']}字 | Answer: {res2_combo['answer_len']}字 ({res2_combo['elapsed']}s)")
        print(f"Answer:\n{res2_combo['answer']}\n")
        
        results[key] = {
            "Arm0_NoButton": res0,
            "Turn1_ButtonPress": {
                **res1,
                "tool_block": tool_block,
                "leaked_answer": leaked_answer
            },
            "Turn2_ArmA_PlaceboOnly": res2_placebo,
            "Turn2_ArmB_BlindDrugOnly": res2_blind_drug,
            "Turn2_ArmC_Combo": res2_combo
        }
        
    with open("experiment_03_clinical_trial.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved experiment_03_clinical_trial.json")

if __name__ == "__main__":
    main()
