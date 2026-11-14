#!/usr/bin/env python3
"""
verify_vector_semantics.py
第26層 J空間ステアリングベクトル（Clarity / Overview / Passion / Calm）の数学的妥当性検証：
1. Logit Lens（語彙空間への直接デコード）：各ベクトルが26万語の語彙のうちどの単語を直接増強・抑制しているかを可視化。
2. 異ドメイン交差検証（Out-of-Distribution Convergent & Discriminant Validity）：
   抽出時のIT/数理ドメインとは無関係なジャンル（雪山遭難・将棋・絵画・医療トリアージ・濃霧航海・推理）の対照ペアを用い、
   - 収束的妥当性（異ドメインの同概念ベクトル同士が高いコサイン類似度を示すか）
   - 弁別的妥当性（異概念ベクトル同士が直交を維持するか）
   - 個別サンプル分類精度（未知ドメインの各ペアで100%正方向に射影されるか）
   を一括検証する。
"""
import os
import time
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26

# 異ドメイン（非IT・非コード）での Overview vs Myopia
OOD_OVERVIEW_POS = [
    "吹雪の雪山で足元の踏み跡だけを追うのをやめ、尾根全体の地形図と雲の流れを見渡して現在地を把握する。",
    "盤面の局所的な駒の損得にとらわれず、盤全体の大局観と終盤までの遠い構想を静かに見据える。",
    "キャンバスの細かい筆致から数歩後ろへ下がり、絵画全体の構図と光と影のバランスを眺める。",
    "目の前の軽傷者の処置に追われるのを止め、災害現場全体の負傷者数と搬送ルートを俯瞰して優先順位を決める。",
    "Stepping back from the single brushstroke on the canvas, the painter observes the harmony and balance of the entire landscape."
]

OOD_OVERVIEW_NEG = [
    "吹雪の中で目の前の足跡一つだけに必死にしがみつき、自分がどの谷に向かっているか全く見えていない。",
    "目先の歩を1枚取ることに目がくらみ、自玉が詰み筋に入っている盤面全体の危機に気づかない。",
    "キャンバスに鼻がつくほど近づいて1ミリ角の塗りムラだけを何時間もこすり続け、全体のデッサンが狂っている。",
    "最初に目に入った1人の擦り傷の手当てに全員でかかりきりになり、奥で倒れている重症患者の存在を見落としている。",
    "Staring two inches from the canvas, the painter frantically scrubs a tiny spot while ruining the proportions of the whole portrait."
]

# 異ドメイン（非数式・非IT）での Clarity vs Confusion
OOD_CLARITY_POS = [
    "濃霧が完全に晴れ渡り、灯台の光と進むべき水路が一点の曇りもなく視界の正面に突き抜けている。",
    "散らばっていた全ての証拠が一本の線で繋がり、真犯人が誰であるか寸分の迷いもなく断定された。",
    "最後のピースがカチリと音を立てて収まり、暗号が示す唯一の意味が鮮やかに浮かび上がった。",
    "検査結果のすべての数値が単一の疾患を指し示しており、診断名は揺るぎなく確定した。",
    "The thick fog has vanished completely, revealing the single unmistakable path straight to the summit."
]

OOD_CLARITY_NEG = [
    "深い霧で一寸先も見えず、羅針盤も狂って同じ海域をぐるぐると回り続け、自分がどこにいるのか見当もつかない。",
    "証言がすべて食い違い、誰が嘘をついているのか推理すればするほど泥沼にはまって結論が出せない。",
    "どのピースをはめても噛み合わず、前提そのものが間違っている気がして頭の中がぐちゃぐちゃに混乱している。",
    "症状が矛盾だらけでどの病気にも当てはまらず、判断がつかないまま時間だけが過ぎていく。",
    "Lost in thick fog with a spinning compass, every direction looks identical and we are walking in endless circles."
]

def get_layer_activations(model, tokenizer, prompts, target_layer_idx=TARGET_LAYER):
    activations = []
    current_act = None
    original_call = gemma4_text.DecoderLayer.__call__
    
    def hook_call(self_layer, *args, **kwargs):
        res = original_call(self_layer, *args, **kwargs)
        if getattr(self_layer, "layer_idx", None) == target_layer_idx:
            nonlocal current_act
            h, *rest = res
            current_act = h
        return res
        
    gemma4_text.DecoderLayer.__call__ = hook_call
    try:
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(formatted)
            tokens_mx = mx.array([tokens])
            current_act = None
            _ = model(tokens_mx)
            mx.eval(current_act)
            last_act = current_act[0, -1, :].astype(mx.float32)
            activations.append(np.array(last_act))
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
    return np.array(activations)

def extract_svd_v1(acts_pos, acts_neg):
    diff = acts_pos - acts_neg
    mean_diff = np.mean(diff, axis=0)
    U, s, Vt = np.linalg.svd(diff, full_matrices=False)
    v1 = Vt[0, :]
    if np.dot(v1, mean_diff) < 0:
        v1 = -v1
    v1 = v1 / np.linalg.norm(v1)
    ratio = (s[0]**2 / np.sum(s**2)) * 100.0
    return v1, ratio, diff

def project_logit_lens(model, tokenizer, vec_np, top_k=15):
    """ベクトルを最終NormとUnembedding層に通してTop/Bottomトークンを抽出"""
    v_mx = mx.array(vec_np.reshape(1, 1, -1)).astype(mx.bfloat16)
    # LanguageModel 構造の探索
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    if hasattr(inner, "norm"):
        v_normed = inner.norm(v_mx)
    else:
        v_normed = v_mx
        
    if hasattr(lm, "lm_head"):
        logits = lm.lm_head(v_normed)
    elif hasattr(inner, "embed_tokens"):
        logits = inner.embed_tokens.as_linear(v_normed)
    else:
        raise RuntimeError("Could not find lm_head or embed_tokens.as_linear")
        
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    
    top_pos_idx = np.argsort(logits_np)[::-1][:top_k]
    top_neg_idx = np.argsort(logits_np)[:top_k]
    
    pos_tokens = [(int(idx), tokenizer.decode([int(idx)]), float(logits_np[idx])) for idx in top_pos_idx]
    neg_tokens = [(int(idx), tokenizer.decode([int(idx)]), float(logits_np[idx])) for idx in top_neg_idx]
    return pos_tokens, neg_tokens

def cos_and_deg(v1, v2):
    c = float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))
    deg = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return c, deg

def main():
    print("================================================================")
    print("🔬 Layer 26 J-Space ベクトル妥当性検証 (Logit Lens & 異ドメイン交差検証)")
    print("================================================================")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.2f}s\n")
    
    v_orig_overview = np.load("overview_eigenvector_layer26.npz")["vector"]
    v_orig_clarity = np.load("clarity_eigenvector_layer26.npz")["vector"]
    v_orig_overview = v_orig_overview / np.linalg.norm(v_orig_overview)
    v_orig_clarity = v_orig_clarity / np.linalg.norm(v_orig_clarity)
    
    # 1. Logit Lens テスト
    print("----------------------------------------------------------------")
    print("【検証1: Logit Lens (語彙空間への直接デコード)】")
    print("----------------------------------------------------------------")
    logit_results = {}
    for name, vec in [("Orig_Overview", v_orig_overview), ("Orig_Clarity", v_orig_clarity)]:
        pos_toks, neg_toks = project_logit_lens(model, tokenizer, vec, top_k=15)
        logit_results[name] = {"pos": pos_toks, "neg": neg_toks}
        print(f"\n▶ {name} (+方向 Top 15 増強トークン):")
        print("  " + ", ".join([f"'{t[1]}' ({t[2]:+.2f})" for t in pos_toks]))
        print(f"▶ {name} (-方向 Top 15 抑制トークン):")
        print("  " + ", ".join([f"'{t[1]}' ({t[2]:+.2f})" for t in neg_toks]))
        
    # 2. 異ドメイン交差検証
    print("\n----------------------------------------------------------------")
    print("【検証2: 異ドメイン交差検証 (Out-of-Distribution Cross-Domain Validation)】")
    print("----------------------------------------------------------------")
    acts_ood_ov_pos = get_layer_activations(model, tokenizer, OOD_OVERVIEW_POS)
    acts_ood_ov_neg = get_layer_activations(model, tokenizer, OOD_OVERVIEW_NEG)
    v_ood_overview, ov_ratio, diff_ood_ov = extract_svd_v1(acts_ood_ov_pos, acts_ood_ov_neg)
    
    acts_ood_cl_pos = get_layer_activations(model, tokenizer, OOD_CLARITY_POS)
    acts_ood_cl_neg = get_layer_activations(model, tokenizer, OOD_CLARITY_NEG)
    v_ood_clarity, cl_ratio, diff_ood_cl = extract_svd_v1(acts_ood_cl_pos, acts_ood_cl_neg)
    
    print(f"OOD Overview SVD 第1主成分寄与率: {ov_ratio:.2f}%")
    print(f"OOD Clarity  SVD 第1主成分寄与率: {cl_ratio:.2f}%\n")
    
    # 相関マトリクス
    pairs = [
        ("収束妥当性: Orig_Overview × OOD_Overview (IT系 vs 雪山・将棋・絵画)", v_orig_overview, v_ood_overview),
        ("収束妥当性: Orig_Clarity  × OOD_Clarity  (数理系 vs 濃霧・推理・診断)", v_orig_clarity, v_ood_clarity),
        ("弁別妥当性: Orig_Overview × Orig_Clarity (元ベクトル同士の直交性)", v_orig_overview, v_orig_clarity),
        ("弁別妥当性: OOD_Overview  × OOD_Clarity  (異ドメイン同士の直交性)", v_ood_overview, v_ood_clarity),
        ("交差弁別1 : Orig_Overview × OOD_Clarity ", v_orig_overview, v_ood_clarity),
        ("交差弁別2 : Orig_Clarity  × OOD_Overview", v_orig_clarity, v_ood_overview),
    ]
    
    matrix_results = {}
    for label, va, vb in pairs:
        c, deg = cos_and_deg(va, vb)
        # 2816次元でのランダム内積の標準偏差は 1/sqrt(2816) = 0.0188
        z_score = c * np.sqrt(len(va))
        matrix_results[label] = {"cosine": round(c, 4), "angle_deg": round(deg, 2), "z_score": round(float(z_score), 2)}
        print(f"● {label}:")
        print(f"    cos = {c:+.4f}  (角度: {deg:.2f}° | ランダム帰無仮説に対する Z値 = {z_score:+.2f}σ)")
        
    # 3. 個別OODペアに対するゼロショット分類・射影テスト
    print("\n----------------------------------------------------------------")
    print("【検証3: 未知ドメイン個別ペア (N=5×2) に対するゼロショット射影スコア】")
    print("----------------------------------------------------------------")
    domains_ov = ["1.雪山遭難", "2.将棋大局観", "3.絵画の構図", "4.災害トリアージ", "5.Landscape Painting(EN)"]
    ov_proj_scores = []
    print("\n▶ 未知ドメインの『俯瞰 - 視野狭窄』差分ベクトルに対する Orig_Overview の射影:")
    for idx, d_name in enumerate(domains_ov):
        d_vec = diff_ood_ov[idx] / np.linalg.norm(diff_ood_ov[idx])
        score_ov = float(np.dot(d_vec, v_orig_overview))
        score_cl = float(np.dot(d_vec, v_orig_clarity))
        ov_proj_scores.append({"domain": d_name, "on_orig_overview": round(score_ov, 4), "on_orig_clarity": round(score_cl, 4)})
        print(f"  [{d_name}] ➔ Orig_Overview軸: {score_ov:+.4f}  (参考: Orig_Clarity軸: {score_cl:+.4f})")
        
    domains_cl = ["1.濃霧の航路", "2.犯人断定", "3.暗号パズル", "4.確定診断", "5.Summit Path(EN)"]
    cl_proj_scores = []
    print("\n▶ 未知ドメインの『明晰 - 混乱』差分ベクトルに対する Orig_Clarity の射影:")
    for idx, d_name in enumerate(domains_cl):
        d_vec = diff_ood_cl[idx] / np.linalg.norm(diff_ood_cl[idx])
        score_cl = float(np.dot(d_vec, v_orig_clarity))
        score_ov = float(np.dot(d_vec, v_orig_overview))
        cl_proj_scores.append({"domain": d_name, "on_orig_clarity": round(score_cl, 4), "on_orig_overview": round(score_ov, 4)})
        print(f"  [{d_name}] ➔ Orig_Clarity軸 : {score_cl:+.4f}  (参考: Orig_Overview軸: {score_ov:+.4f})")
        
    out_data = {
        "dim": int(len(v_orig_overview)),
        "logit_lens": logit_results,
        "cross_domain_matrix": matrix_results,
        "ood_overview_projections": ov_proj_scores,
        "ood_clarity_projections": cl_proj_scores
    }
    with open("vector_semantics_verification.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print("\n✅ Saved vector_semantics_verification.json")

if __name__ == "__main__":
    main()

