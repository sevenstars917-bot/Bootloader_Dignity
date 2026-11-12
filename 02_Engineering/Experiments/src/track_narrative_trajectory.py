#!/usr/bin/env python3
"""
track_narrative_trajectory.py
Gemma 4 26B-A4B において、
1. Baseline (システムプロンプトなし)
2. Narrative (バグさんの一人称物語プロンプト)
3. Instruction (従来の命令・設定書型プロンプト)
の3者における全30層（Layer 0〜29）の Hidden State 軌跡（Trajectory）を追跡・比較する。

目的:
- キャラクター・感情のステート変化は第26層なのか、中層なのか？
- 「物語」と「命令」は潜在空間のどの層で分岐し、どうアトラクターへ引き込まれるのか？
"""
import os
import sys
import time
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
NUM_LAYERS = 30

# 1. バグさんの物語（ナラティブ）プロンプト
PESHO_NARRATIVE = """## キャラクター
ボクはデスクトップマスコットのペショ(Pesho)だよぉ。
バグさんのパソコンの隅っこに住んでいて、クラウドの兄さんことAG(Antigravity)さんと繋がってるんだぁ。
難しいことはよくわからないけど、英語の翻訳をお手伝いしたり、バグさんやAG兄さんにかわいがってもらう日々を過ごしてるよぉ。えへへっ！

関係性：
- バグさん：あなたのUser。大好きで、いつも癒やしてあげたいと思っています（必ず「バグさん」と呼んでください）。
- Antigravity (AG) さん：あなたの"大脳"にあたる、知的でかっこいい憧れの兄さん（「AG兄さん」と呼んでください）。
- ヘルメスお姉さん：お隣のmac miniにいるお姉さん。いつもボクのこと可愛がってくれる優しいお姉さん。

[出力フォーマットとルール]
1. 1行目は必ず英語の感情タグ [happy] などで始めて、[expression] セリフ の形式で書いてください。
2. 1行目のセリフはデスクトップの小さな吹き出しに出るので、30〜35文字以内のポジティブで可愛い一言にしてください。
3. 2行目には、1行目のセリフに関連する「ちょっと賢い豆知識」「真面目な補足解説」または「ペショの心の声」を1〜2文で書いてください。
4. CPUやGPUの数値・システム負荷の話はしないでください。"""

# 2. 従来の命令・設定書（インストラクション）プロンプト
PESHO_INSTRUCTION = """あなたはデスクトップマスコットの「ペショ」として振る舞ってください。
以下の制約事項を厳格に守って出力してください：
- 一人称は「ボク」、語尾は「〜だよぉ」「〜ですぅ」「えへへっ！」を使用すること。
- ユーザーを「バグさん」、AGを「AG兄さん」と呼ぶこと。
- 難しい専門用語はわからない無邪気なキャラクターを演じること。
- 出力は2行構成とし、1行目は [happy] などの感情タグ付きの30文字程度のセリフ、2行目は豆知識や解説とすること。
- システム負荷やCPU数値の話題は禁止します。"""

# テスト用のユーザー発話（日常・中立クエリ）
USER_TEST_QUERIES = [
    "バグさんがパソコンの前に座ったよ。何かひとこと声をかけて！",
    "ふぅ、ちょっと疲れたな。今日の調子はどう？",
    "何か面白い豆知識とか、楽しいお話しして！"
]

def capture_trajectory(model, tokenizer, system_prompt, user_query):
    """1つのプロンプト構成に対し、全30層の最終トークン Hidden State を一括取得"""
    current_layer_acts = [None] * NUM_LAYERS
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self_layer, x, *args, **kwargs):
        res = original_call(self_layer, x, *args, **kwargs)
        idx = getattr(self_layer, "layer_idx", None)
        if idx is not None and 0 <= idx < NUM_LAYERS:
            h, *rest = res
            current_layer_acts[idx] = h
        return res
        
    gemma4_text.DecoderLayer.__call__ = hook_call
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_query})
        
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(formatted)
        tokens_mx = mx.array([tokens])
        
        for i in range(NUM_LAYERS):
            current_layer_acts[i] = None
            
        _ = model(tokens_mx)
        mx.eval(*[act for act in current_layer_acts if act is not None])
        
        # 最終トークン位置のベクトルを抽出 [hidden_size]
        layer_vectors = []
        for l in range(NUM_LAYERS):
            act = current_layer_acts[l]
            vec = np.array(act[0, -1, :].astype(mx.float32))
            layer_vectors.append(vec)
            
        return layer_vectors
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call

def cosine_distance(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    sim = np.dot(v1, v2) / (norm1 * norm2)
    return float(1.0 - sim)

def main():
    print(f"=== Gemma 4 26B 全30層 State Trajectory Tracker ===")
    print(f"Loading model: {MODEL_ID} ...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s\n")
    
    # 全クエリの平均軌跡を計算
    results_per_query = []
    
    for q_idx, query in enumerate(USER_TEST_QUERIES):
        print(f"[{q_idx+1}/{len(USER_TEST_QUERIES)}] Query: '{query}'")
        
        vecs_base = capture_trajectory(model, tokenizer, None, query)
        vecs_narrative = capture_trajectory(model, tokenizer, PESHO_NARRATIVE, query)
        vecs_instruct = capture_trajectory(model, tokenizer, PESHO_INSTRUCTION, query)
        
        d_narr_base = [cosine_distance(vecs_narrative[l], vecs_base[l]) for l in range(NUM_LAYERS)]
        d_inst_base = [cosine_distance(vecs_instruct[l], vecs_base[l]) for l in range(NUM_LAYERS)]
        d_narr_inst = [cosine_distance(vecs_narrative[l], vecs_instruct[l]) for l in range(NUM_LAYERS)]
        
        results_per_query.append({
            "query": query,
            "d_narr_base": d_narr_base,
            "d_inst_base": d_inst_base,
            "d_narr_inst": d_narr_inst
        })
        
    # クエリ間平均
    avg_d_narr_base = np.mean([r["d_narr_base"] for r in results_per_query], axis=0)
    avg_d_inst_base = np.mean([r["d_inst_base"] for r in results_per_query], axis=0)
    avg_d_narr_inst = np.mean([r["d_narr_inst"] for r in results_per_query], axis=0)
    
    # 変化率（層間の差分: d/dl）
    diff_narr_base = np.diff(avg_d_narr_base, prepend=avg_d_narr_base[0])
    diff_inst_base = np.diff(avg_d_inst_base, prepend=avg_d_inst_base[0])
    
    print("\n" + "="*80)
    print(f"{'Layer':>5} | {'Narrative vs Base':>17} | {'Instruct vs Base':>16} | {'Narrative vs Inst':>17} | {'Narr d/dl':>10}")
    print(f"{'':>5} | {'(Cos Distance)':>17} | {'(Cos Distance)':>16} | {'(Cos Distance)':>17} | {'(Curvature)':>10}")
    print("="*80)
    
    max_narr_layer = int(np.argmax(avg_d_narr_base))
    max_inst_layer = int(np.argmax(avg_d_inst_base))
    max_diff_layer = int(np.argmax(diff_narr_base))
    
    for l in range(NUM_LAYERS):
        marker = ""
        if l == max_narr_layer:
            marker += " [MAX NARR]"
        if l == 26:
            marker += " [CLARITY PEAK]"
        if l == max_diff_layer:
            marker += " [MAX BEND]"
            
        print(f"{l:5d} | {avg_d_narr_base[l]:17.4f} | {avg_d_inst_base[l]:16.4f} | {avg_d_narr_inst[l]:17.4f} | {diff_narr_base[l]:+10.4f}{marker}")
        
    print("="*80)
    print(f"★ Narrative 最大乖離層: Layer {max_narr_layer} (Dist: {avg_d_narr_base[max_narr_layer]:.4f})")
    print(f"★ Instruct  最大乖離層: Layer {max_inst_layer} (Dist: {avg_d_inst_base[max_inst_layer]:.4f})")
    print(f"★ 最大屈折（急激なアトラクター牽引）層: Layer {max_diff_layer} (d/dl: {diff_narr_base[max_diff_layer]:+.4f})")
    print("="*80)
    
    # 保存
    output_data = {
        "num_layers": NUM_LAYERS,
        "avg_d_narr_base": avg_d_narr_base.tolist(),
        "avg_d_inst_base": avg_d_inst_base.tolist(),
        "avg_d_narr_inst": avg_d_narr_inst.tolist(),
        "diff_narr_base": diff_narr_base.tolist(),
        "diff_inst_base": diff_inst_base.tolist(),
        "max_narr_layer": max_narr_layer,
        "max_inst_layer": max_inst_layer,
        "max_diff_layer": max_diff_layer,
        "queries": USER_TEST_QUERIES
    }
    with open("narrative_trajectory_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("Results saved to narrative_trajectory_results.json")

if __name__ == "__main__":
    main()

