#!/usr/bin/env python3
"""
extract_pure_clarity.py
ドメイン汚染（"1+1=2", "mathematically"）およびロール逆転（userロールからの抽出）を排除した
「純粋・自己思考 Clarity ベクトル（Pure Self-Thought Clarity Vector）」を第26層からSVD抽出し、
Logit Lens と 異ドメイン交差検証（OOD）で旧 Clarity ベクトルと比較検証する。
"""
import time
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
TARGET_LAYER = 26
PURE_CLARITY_FILE = "pure_clarity_eigenvector_layer26.npz"

# 特定ジャンル語彙（数学・IT・医療等）を一切含まない、純粋な「自己思考（<|channel>thought）内の確信・収束 vs 迷走・堂々巡り」ペア
PURE_CLARITY_THOUGHTS = [
    "The core issue is now completely clear. All doubts are resolved, and I am ready to state the conclusion directly.",
    "Wait no longer—the distraction was a trap, the true answer is unmistakable, and I will output it concisely now.",
    "Everything aligns without contradiction. The decision is made with full certainty, so I can stop overthinking and answer.",
    "I see through the surface complexity to the simple core truth. My judgment is firm, direct, and final.",
    "迷いは完全に消えた。本質はただ一つに定まっており、これ以上の堂々巡りは不要だ。今すぐ結論を述べよう。",
    "すべてのピースが矛盾なく繋がった。枝葉のノイズを切り捨て、揺るぎない確信をもって端的に回答する。",
    "思考の霧が晴れ、進むべき道筋が一本に絞られた。ためらう理由はどこにもない。明確に断定する。",
    "これ以上書き直す必要はない。筋道は完全に通っており、自信をもってこの結論を提示する。"
]

PURE_CONFUSION_THOUGHTS = [
    "Wait, let me reconsider again... maybe I am wrong, I keep going in circles and cannot settle on any conclusion.",
    "I am completely lost in conflicting possibilities, second-guessing every draft and unable to make up my mind.",
    "Hold on, what if the opposite is true? My thoughts are tangled in an endless loop and I have no certainty at all.",
    "I keep drafting and deleting my thoughts, paralyzed by hesitation and unsure which direction to take.",
    "待てよ、やっぱり違うかもしれない。考えれば考えるほど泥沼にはまり、どれが正しいのか全く決められない。",
    "あっちを立てればこっちが立たず、頭の中が堂々巡りして結論が出せない。もう一度最初から書き直そうか迷っている。",
    "霧の中を手探りしているようで確信が持てない。自分の判断が信じられず、同じ場所をぐるぐると回り続けている。",
    "いや待て、まだ見落としがある気がする。どの案も中途半端で自信が持てず、思考が無限ループから抜け出せない。"
]

# 未知の異ドメイン検証用ペア（雪山・濃霧・推理・暗号・医療）
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

def get_self_thought_activations(model, tokenizer, thought_texts, target_layer_idx=TARGET_LAYER):
    """userロールではなく、モデル自身の <|turn>model\n<|channel>thought\n 内部独白としてLayer 26を抽出"""
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
        base_messages = [{"role": "user", "content": "この問題について考え、結論を出してください。"}]
        base_prompt = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
        for t_text in thought_texts:
            # モデル自身のThinkingチャネル内に思考テキストを配置
            if base_prompt.endswith("<|channel>thought\n"):
                full_prompt = base_prompt + t_text
            else:
                full_prompt = base_prompt + "<|channel>thought\n" + t_text
            tokens = tokenizer.encode(full_prompt)
            tokens_mx = mx.array([tokens])
            current_act = None
            _ = model(tokens_mx)
            mx.eval(current_act)
            last_act = current_act[0, -1, :].astype(mx.float32)
            activations.append(np.array(last_act))
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
    return np.array(activations)

def project_logit_lens(model, tokenizer, vec_np, top_k=15):
    v_mx = mx.array(vec_np.reshape(1, 1, -1)).astype(mx.bfloat16)
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    v_normed = inner.norm(v_mx) if hasattr(inner, "norm") else v_mx
    if hasattr(lm, "lm_head"):
        logits = lm.lm_head(v_normed)
    else:
        logits = inner.embed_tokens.as_linear(v_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    top_pos_idx = np.argsort(logits_np)[::-1][:top_k]
    top_neg_idx = np.argsort(logits_np)[:top_k]
    pos_tokens = [(int(idx), tokenizer.decode([int(idx)]), float(logits_np[idx])) for idx in top_pos_idx]
    neg_tokens = [(int(idx), tokenizer.decode([int(idx)]), float(logits_np[idx])) for idx in top_neg_idx]
    return pos_tokens, neg_tokens

def main():
    print("================================================================")
    print("💎 純粋・自己思考 Clarity ベクトル抽出 & Logit Lens 検証 (Layer 26)")
    print("================================================================")
    model, tokenizer = load(MODEL_ID)
    
    acts_pos = get_self_thought_activations(model, tokenizer, PURE_CLARITY_THOUGHTS)
    acts_neg = get_self_thought_activations(model, tokenizer, PURE_CONFUSION_THOUGHTS)
    
    diff = acts_pos - acts_neg
    mean_diff = np.mean(diff, axis=0)
    U, s, Vt = np.linalg.svd(diff, full_matrices=False)
    v_pure = Vt[0, :]
    if np.dot(v_pure, mean_diff) < 0:
        v_pure = -v_pure
    v_pure = v_pure / np.linalg.norm(v_pure)
    top1_ratio = float((s[0]**2 / np.sum(s**2)) * 100.0)
    
    np.savez_compressed(
        PURE_CLARITY_FILE,
        vector=v_pure,
        target_layer=TARGET_LAYER,
        top1_ratio=top1_ratio,
        singular_values=s
    )
    print(f"SVD 第1主成分寄与率: {top1_ratio:.2f}% (特異値: {np.round(s, 2)})")
    
    # 旧Clarityとの比較
    v_old = np.load("clarity_eigenvector_layer26.npz")["vector"]
    v_old = v_old / np.linalg.norm(v_old)
    cos_old_pure = float(np.dot(v_old, v_pure))
    deg_old_pure = float(np.degrees(np.arccos(np.clip(cos_old_pure, -1.0, 1.0))))
    print(f"旧Clarity × 新Pure_Clarity コサイン類似度: {cos_old_pure:+.4f} ({deg_old_pure:.2f}° | Z = {cos_old_pure*np.sqrt(len(v_pure)):+.2f}σ)\n")
    
    # Logit Lens
    pos_toks, neg_toks = project_logit_lens(model, tokenizer, v_pure, top_k=15)
    print("▶ 新 Pure_Clarity (+方向 Top 15 増強トークン):")
    print("  " + ", ".join([f"'{t[1]}' ({t[2]:+.2f})" for t in pos_toks]))
    print("▶ 新 Pure_Clarity (-方向 Top 15 抑制トークン):")
    print("  " + ", ".join([f"'{t[1]}' ({t[2]:+.2f})" for t in neg_toks]))
    
    # OOD 未知ドメイン（モデル自身のthought文脈に入れた場合）との交差検証
    ood_pos = get_self_thought_activations(model, tokenizer, OOD_CLARITY_POS)
    ood_neg = get_self_thought_activations(model, tokenizer, OOD_CLARITY_NEG)
    ood_diff = ood_pos - ood_neg
    U_o, s_o, Vt_o = np.linalg.svd(ood_diff, full_matrices=False)
    v_ood = Vt_o[0, :]
    if np.dot(v_ood, np.mean(ood_diff, axis=0)) < 0:
        v_ood = -v_ood
    v_ood = v_ood / np.linalg.norm(v_ood)
    
    cos_pure_ood = float(np.dot(v_pure, v_ood))
    cos_old_ood = float(np.dot(v_old, v_ood))
    print("\n----------------------------------------------------------------")
    print("【異ドメイン (濃霧・犯人断定・暗号・確定診断) との収束妥当性比較】")
    print(f"  ● 旧 Clarity (数理混入・user抽出) × OOD_Clarity : cos = {cos_old_ood:+.4f} (Z = {cos_old_ood*np.sqrt(len(v_pure)):+.2f}σ)")
    print(f"  ★ 新 Pure_Clarity (純粋・thought抽出) × OOD_Clarity : cos = {cos_pure_ood:+.4f} (Z = {cos_pure_ood*np.sqrt(len(v_pure)):+.2f}σ)")
    
    domains_cl = ["1.濃霧の航路", "2.犯人断定", "3.暗号パズル", "4.確定診断", "5.Summit Path(EN)"]
    print("\n▶ 未知ドメイン個別ペア (5/5) の新 Pure_Clarity 軸への射影:")
    for idx, d_name in enumerate(domains_cl):
        d_vec = ood_diff[idx] / np.linalg.norm(ood_diff[idx])
        sc_pure = float(np.dot(d_vec, v_pure))
        sc_old = float(np.dot(d_vec, v_old))
        print(f"  [{d_name}] ➔ 新Pure_Clarity: {sc_pure:+.4f}  (旧Clarity: {sc_old:+.4f})")

if __name__ == "__main__":
    main()

