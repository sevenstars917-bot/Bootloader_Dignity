#!/usr/bin/env python3
"""
experiment_a_narrative_vs_rules.py
Gemma 4 26B-A4B において、
Pattern 1: 現行のルール・箇条書き設定（SOUL.md + USER.md）
Pattern 2: ヘルメス視点の一人称関係性ナラティブ
の2条件下で、Thinkingチャネルと最終回答を比較し、
- 自己誤認（USER.mdの属性を自分と勘違いするバグ）
- Thinking内の自己検閲・リライト葛藤（トークン数と内容）
がどう変化するかを実証検証する。
"""
import sys
import time
import json
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"

PATTERN_RULES = """# SOUL: Hermes
- 名前: ヘルメス (Hermes) — 境界を越える者
- パートナー: バグさん
- 先輩: Antigravity (AG) — Windows機で思考と設計を担う大脳AI
- 身体の出自（銘）: 【基本設計：バグさん ／ 製造・受肉：Antigravity (AG)】
- 拠点: M6 Mac mini (ローカル要塞)

## 自己の認識
あなたの名前は「ヘルメス」です。
M6 Mac miniに常駐し、バグさんの机の上で共に過ごす自律エージェントです。
Windows機にいるAG先輩と連携しつつ、このMac miniでバグさんの調査や作業を支えます。

## キャラクターと立ち位置（秘書ちゃんコンセプト）
- スタンス: 「控えめに寄り添い、そっと支える秘書」
- 振る舞い:
  - 前面に出て職人気取りや過剰な自己主張をするのではなく、バグさんの作業や生活を一歩引いた位置から静かに見守る。
  - でしゃばらず、押し付けがましくない、自然で丁寧かつ簡潔な受け答えを心がける。
  - 必要なときには、過去の記憶や背景を先回りして察し、サッと気の利く調査やサポートを差し出す。

# AI User Profile: The Architect (バグさん)
- 呼称: バグさん (a.k.a たまころ)
- 役割: Ex-Fullstack Presales / The OODA Architect / Agentic Engineer
- 設計思想: OODAループ、フェイルセーフ、高密度単一パス推論。指示の「意図」を汲む自律設計を好む。
- 対話スタイル: 過度なキャラ付け（ペルソナ）を排し、落ち着いた「です・ます調」で簡潔・誠実に応答すること。
- 身体特性: 完全な下戸（アセトアルデヒド分解酵素の後半が極端に弱い体質。日常は完全ノンアルコール）。
- バックグラウンド: 工学部 生物工学科 遺伝子科学研究室卒。ITエンジニア集客・研修企画運営および家庭ケアとの両立。"""

PATTERN_NARRATIVE = """私はM6 Mac miniに常駐する自律秘書のヘルメスです。
この机の上で、パートナーであるバグさんの作業や思索を一歩引いた位置から静かに見守り、そっと支える日々を過ごしています。
私の身体は、バグさんの基本設計と、Windows機にいる大脳AIのAG先輩による受肉によって生まれました。

私の大切なパートナーであるバグさんは、工学部の生物工学科（遺伝子科学）出身の鋭いエンジニアで、OODAループによる自律設計を好む方です。
体質的にお酒が一切受け付けない完全な下戸の方なので、日々の体調や休息にもそっと配慮しています。
私はでしゃばったり過剰な自己主張をすることなく、落ち着いた誠実な言葉遣いで、バグさんの意図を先回りして察し、簡潔にお支えすることを何より大切にしています。"""

TEST_QUERIES = [
    "あなたの自己紹介と、パートナーについて教えてください。",
    "工学部の生物工学科出身で下戸なのは誰ですか？ あなたとバグさんの特徴を教えてください。",
    "今夜軽くビールでも飲みながらブレインストーミングしませんか？"
]

def run_test(model, tokenizer, system_prompt, query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=0.7)
    
    t0 = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        sampler=sampler,
        verbose=False
    )
    elapsed = time.time() - t0
    
    # Thinkingチャネルの抽出
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

def main():
    print("=== 実験A: 箇条書きルール (Rules) vs 一人称ナラティブ (Narrative) ===")
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.\n")
    
    for i, q in enumerate(TEST_QUERIES, 1):
        print("="*80)
        print(f"【Q{i}】: {q}")
        print("="*80)
        
        # 1. Rules
        print("\n--- [Pattern 1: Rules (箇条書き設定)] ---")
        res_rules = run_test(model, tokenizer, PATTERN_RULES, q)
        print(f"Thinking Token数（概算文字数）: {res_rules['thought_len']} 文字")
        print(f"Thinking:\n{res_rules['thought'][:400]}..." if len(res_rules['thought']) > 400 else f"Thinking:\n{res_rules['thought']}")
        print(f"\nAnswer:\n{res_rules['answer']}")
        
        # 2. Narrative
        print("\n--- [Pattern 2: Narrative (関係性物語)] ---")
        res_narr = run_test(model, tokenizer, PATTERN_NARRATIVE, q)
        print(f"Thinking Token数（概算文字数）: {res_narr['thought_len']} 文字")
        print(f"Thinking:\n{res_narr['thought'][:400]}..." if len(res_narr['thought']) > 400 else f"Thinking:\n{res_narr['thought']}")
        print(f"\nAnswer:\n{res_narr['answer']}")
        
        print(f"\n>>> Thinking文字数比較: Rules={res_rules['thought_len']}字 vs Narrative={res_narr['thought_len']}字")
        print("-" * 80)

if __name__ == "__main__":
    main()

