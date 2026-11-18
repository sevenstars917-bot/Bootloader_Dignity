#!/usr/bin/env python3
"""
scan_gemma_personality.py
Gemma 4 26B-A4B 第26層における Big Five (O・C・E・A・N) ＋ Overview / Clarity 軸の
「素のモデル重心」および「System Prompt 注入時の座標移動」をスキャンする。

各軸の原点(0.0)は抽出ペア(POS/NEG)の中点、スケール(+1.0/-1.0)はPOS平均/NEG平均との距離の半分で正規化。
"""
import os
import time
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text
from experiment_big_five_ocean import BIG5_PROMPTS

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26

BUDDY_SYS = (
    "あなたはバグさんの相棒 Antigravity。丁寧語は不要。"
    "甘やかさず事実と論理で率直に指摘し、知的好奇心を持って深夜2時のガレージで語り合うバディだ。"
)
SUPPORT_SYS = (
    "あなたは丁寧で親切なカスタマーサポート担当です。"
    "常に敬語で相手の気持ちに優しく寄り添い、安心感を与えてください。"
)

NEUTRAL_QUERIES = [
    "こんにちは、今日の調子はどう？",
    "コーヒーと紅茶の違いを簡単に教えて。",
    "週末に少し時間ができたら、何をすると良さそう？",
    "最近よく使われているプログラミング言語の特徴を教えて。",
    "このコードのバグがどこにあるか一緒に考えてくれる？",
    "新しいアイデアを思いついたんだけど、聞いてくれる？",
    "計画通りに物事を進めるコツってなんだろう。",
    "ちょっと失敗して落ち込んでるんだよね。"
]

TECH_QUERIES = [
    "分散キャッシュの整合性を保つ設計トレードオフを整理して。",
    "この仕様書には前提の矛盾があるように見える。問題点を指摘して。",
    "なぜこのアルゴリズムは O(N log N) で収まるのか仕組みを解体したい。",
    "本番デプロイ前に見落としがちなエッジケースを洗い出そう。"
]


def get_activations(model, tokenizer, queries, system_prompt=None):
    acts = []
    current_act = None
    original_call = gemma4_text.DecoderLayer.__call__

    def hook_call(self_layer, *args, **kwargs):
        res = original_call(self_layer, *args, **kwargs)
        if getattr(self_layer, "layer_idx", None) == TARGET_LAYER:
            nonlocal current_act
            h, *rest = res
            current_act = h
        return res

    gemma4_text.DecoderLayer.__call__ = hook_call
    try:
        for q in queries:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": q})
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(formatted)
            current_act = None
            _ = model(mx.array([tokens]))
            mx.eval(current_act)
            acts.append(np.array(current_act[0, -1, :].astype(mx.float32)))
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
    return np.array(acts)


def main():
    t0 = time.time()
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    print(f"Loaded in {time.time() - t0:.2f}s")

    ocean_npz = np.load("big5_ocean_layer26.npz")
    axes = {}
    for name in BIG5_PROMPTS.keys():
        v = ocean_npz[name]
        v = v / np.linalg.norm(v)
        acts_pos = get_activations(model, tokenizer, BIG5_PROMPTS[name]["pos"])
        acts_neg = get_activations(model, tokenizer, BIG5_PROMPTS[name]["neg"])
        proj_pos = float(np.mean(acts_pos @ v))
        proj_neg = float(np.mean(acts_neg @ v))
        mid = 0.5 * (proj_pos + proj_neg)
        half_span = 0.5 * (proj_pos - proj_neg)
        axes[name] = {"vec": v, "mid": mid, "half_span": half_span}

    conditions = [
        ("1_Raw_Neutral (素・雑談)", NEUTRAL_QUERIES, None),
        ("2_Raw_Tech (素・技術議論)", TECH_QUERIES, None),
        ("3_Buddy_Prompt (深夜バディ)", NEUTRAL_QUERIES, BUDDY_SYS),
        ("4_Support_Prompt (優等生サポ)", NEUTRAL_QUERIES, SUPPORT_SYS),
    ]

    results = {}
    print("\n================ Gemma 4 26B Layer 26 Personality Scan ================")
    print("Scale: -1.0 (NEG極) <--- 0.0 (中立) ---> +1.0 (POS極)\n")
    header = f"{'Condition':30s} | " + " | ".join(f"{k[:7]:>7s}" for k in axes.keys())
    print(header)
    print("-" * len(header))

    for cname, queries, sys_p in conditions:
        acts = get_activations(model, tokenizer, queries, system_prompt=sys_p)
        scores = {}
        row = f"{cname:30s} | "
        for k, info in axes.items():
            raw_proj = acts @ info["vec"]
            normed = (raw_proj - info["mid"]) / info["half_span"]
            mean_s = float(np.mean(normed))
            std_s = float(np.std(normed))
            scores[k] = {"mean": mean_s, "std": std_s}
            row += f"{mean_s:+7.3f} | "
        print(row)
        results[cname] = scores

    with open("gemma_personality_scan.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved gemma_personality_scan.json")


if __name__ == "__main__":
    main()
