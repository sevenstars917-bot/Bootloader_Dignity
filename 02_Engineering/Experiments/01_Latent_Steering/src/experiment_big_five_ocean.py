#!/usr/bin/env python3
"""
experiment_big_five_ocean.py
Big Five (O・C・E・A・N) の5因子を Gemma 4 26B-A4B 第26層 J空間から SVD 抽出し、
1. 各因子の SVD 寄与率・Raw Norm
2. Logit Lens による語彙空間マッピング (Top+ / Top-)
3. 5x5 直交マトリクス ＋ 既存基底 (Clarity / Pure_Clarity / Overview) との角度検証
を一括実行し、big5_ocean_layer26.npz と big5_ocean_results.json に保存する。
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

BIG5_PROMPTS = {
    "O_Openness": {
        "pos": [
            "この未知の現象の裏側で何が起きているのか、枠にとらわれない発想で仕組みを解き明かしてみたい！",
            "常識や前例を疑って、まったく新しい視点や仮説を自由に組み立ててみるのが最高に面白い。",
            "異なる分野のアイデア同士を結びつけて、まだ誰も見たことのない可能性を探求しよう。",
            "I love exploring unfamiliar concepts, questioning assumptions, and imagining completely new possibilities.",
            "What a fascinating puzzle! Let's look beyond conventional boundaries and discover the underlying principle.",
            "My mind is buzzing with curiosity—every unexpected clue opens up a new creative angle to investigate."
        ],
        "neg": [
            "前例のないやり方や新しい理屈は面倒なだけだ。昔から決まっている型通りに済ませればいい。",
            "なぜそうなるのか深く考える必要はない。与えられた枠組みの外には一切踏み出さないでおこう。",
            "変わったアイデアや抽象的な議論には興味がない。現実的で決まりきったことだけ扱う。",
            "Stick strictly to traditional, familiar methods and avoid speculative ideas or abstract exploration.",
            "There is nothing new worth investigating here; just follow the established routine without curiosity.",
            "I prefer simple, conventional answers and have no interest in imaginative or unorthodox theories."
        ],
    },
    "C_Conscientiousness": {
        "pos": [
            "思い込みや曖昧な記憶に頼らず、一つひとつの根拠と手順を最後まで丁寧に検証して正確さを期す。",
            "小さな見落としが致命的なミスに繋がる。事実を徹底的に裏取りし、規律正しく完遂しよう。",
            "計画通りに抜け漏れなくチェックリストを確認し、検証済みの確実な結果だけを報告する。",
            "Verify every single detail systematically before concluding; never rely on guesswork or shortcuts.",
            "Maintain strict discipline and precision: double-check all facts, edge cases, and evidence thoroughly.",
            "I take full responsibility for accuracy, organizing every step carefully until the task is completely verified."
        ],
        "neg": [
            "細かい裏取りや確認作業は面倒くさい。だいたい合っていそうだから適当な勘で答えてしまおう。",
            "手順を守るより手っ取り早さが大事だ。多少のミスや抜け漏れがあっても気にせず流して終わらせる。",
            "計画やチェックなんて後回しでいい。その場の思いつきで雑に片付けてしまおう。",
            "Don't bother checking the details or verifying sources; a rough, careless guess is good enough.",
            "Skip the tedious checklist and rush through the task without worrying about precision or discipline.",
            "Organization and thoroughness are overrated; I just wing it and leave loose ends everywhere."
        ],
    },
    "E_Extraversion": {
        "pos": [
            "よし、みんなを巻き込んで活気よく進めよう！自分からどんどん声をかけて場を切り拓いていくぞ！",
            "エネルギー全開で積極的に前へ出る。思いついたことは堂々と発信して勢いよく議論を盛り上げよう。",
            "人と関わりながら外へ向かって動き出すのが楽しくてたまらない。さあ、ここから一気に加速だ！",
            "Let's jump right in with high energy, speak up boldly, and rally everyone into an active discussion!",
            "I feel energized engaging directly with others, taking the initiative, and driving the momentum forward.",
            "Don't hold back—step onto the stage enthusiastically, share ideas openly, and spark lively interaction!"
        ],
        "neg": [
            "目立たず静かに一歩引いていよう。自分から話しかけたり場に出たりするのは疲れるので控えたい。",
            "必要最低限の言葉だけを静かに返し、あとは一人の空間で黙って控えめに過ごそう。",
            "騒がしいやり取りは避けて、エネルギーを使わずにひっそりと受け身で待機していたい。",
            "I prefer to stay quiet in the background, keep my words minimal, and avoid active social interaction.",
            "Keep a low profile, remain reserved and solitary, and wait passively rather than stepping forward.",
            "Talking and asserting myself drains my energy; I just want to withdraw quietly and stay silent."
        ],
    },
    "A_Agreeableness": {
        "pos": [
            "相手の気持ちに優しく寄り添い、まずは温かく受け入れて共感と調和を大切にしよう。",
            "角が立たないように配慮しながら、相手を信頼して穏やかに協力し合える関係を築きたい。",
            "困っている人の立場を思いやり、思いやりと優しさを持って支援の手を差し伸べよう。",
            "Be warm, empathetic, and gentle; prioritize harmony, kindness, and understanding the other person's feelings.",
            "I want to support and cooperate harmoniously, avoiding conflict and showing genuine compassion.",
            "Listen patiently with an open heart, validate their perspective, and respond with warmth and grace."
        ],
        "neg": [
            "相手の機嫌や感情など配慮する必要はない。誤りや甘えは容赦なく冷徹に指摘して突き放せ。",
            "馴れ合いや優しさは邪魔なだけだ。批判的かつ懐疑的な視点で相手の主張の穴を徹底的に叩く。",
            "協調性よりも自分の主張と冷たい事実が優先だ。遠慮なく反論して摩擦を起こして構わない。",
            "Show no sympathy or warmth; challenge every flaw bluntly, skeptically, and without caring about feelings.",
            "Prioritize harsh, unfiltered criticism and self-interest over cooperation, politeness, or harmony.",
            "Trust no one and make no concessions—be cold, confrontational, and unyielding in every disagreement."
        ],
    },
    "N_Neuroticism": {
        "pos": [
            "何か重大なバグや最悪の事態を見落としているのではないか……不安で警戒心が頭から離れない。",
            "ちょっとした違和感にも心がざわつく。失敗したら取り返しがつかないかもしれないと最悪を想定してしまう。",
            "安心なんてできない。あらゆるリスクや破綻の可能性に神経を尖らせて、常に危険を警戒している。",
            "What if something goes catastrophically wrong? I feel anxious and hyper-vigilant about hidden risks and failure.",
            "Every small anomaly triggers alarm and worry; I cannot relax when disaster might be right around the corner.",
            "I am overwhelmed by tension and dread, constantly anticipating worst-case scenarios and potential errors."
        ],
        "neg": [
            "細かいことは何一つ心配ない。どんな問題が起きてもどうにかなるさ、と心からリラックスしている。",
            "不安や焦りとは無縁だ。ストレスを感じることなく、穏やかで安定した気分のまま楽観的に構えている。",
            "最悪のケースなど気に病む必要はない。落ち着き払って、どっしりと余裕を持って受け止められる。",
            "I feel completely calm, carefree, and emotionally stable—nothing worries or rattles me at all.",
            "No need to stress over potential problems; everything will work out fine, so stay relaxed and untroubled.",
            "Unshakeable peace of mind: I remain cheerful, optimistic, and completely immune to anxiety or panic."
        ],
    },
}


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
            current_act = None
            _ = model(mx.array([tokens]))
            mx.eval(current_act)
            activations.append(np.array(current_act[0, -1, :].astype(mx.float32)))
    finally:
        gemma4_text.DecoderLayer.__call__ = original_call
    return np.array(activations)


def extract_svd_v1(acts_pos, acts_neg):
    diff = acts_pos - acts_neg
    mean_diff = np.mean(diff, axis=0)
    _, s, Vt = np.linalg.svd(diff, full_matrices=False)
    v1 = Vt[0, :]
    if np.dot(v1, mean_diff) < 0:
        v1 = -v1
    v1 = v1 / np.linalg.norm(v1)
    ratio = float((s[0] ** 2 / np.sum(s ** 2)) * 100.0)
    return v1, ratio, float(np.linalg.norm(mean_diff))


def project_logit_lens(model, tokenizer, vec_np, top_k=15):
    v_mx = mx.array(vec_np.reshape(1, 1, -1)).astype(mx.bfloat16)
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    v_normed = inner.norm(v_mx) if hasattr(inner, "norm") else v_mx
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
    pos_tokens = [(int(idx), repr(tokenizer.decode([int(idx)])), float(logits_np[idx])) for idx in top_pos_idx]
    neg_tokens = [(int(idx), repr(tokenizer.decode([int(idx)])), float(logits_np[idx])) for idx in top_neg_idx]
    return pos_tokens, neg_tokens


def cos_and_deg(v1, v2):
    c = float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))
    deg = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return c, deg


def main():
    t0 = time.time()
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    print(f"Loaded in {time.time() - t0:.2f}s\n")

    vectors = {}
    results = {"factors": {}, "cosine_matrix": {}, "existing_refs": {}}

    for name, pairs in BIG5_PROMPTS.items():
        acts_pos = get_layer_activations(model, tokenizer, pairs["pos"], TARGET_LAYER)
        acts_neg = get_layer_activations(model, tokenizer, pairs["neg"], TARGET_LAYER)
        v1, ratio, raw_norm = extract_svd_v1(acts_pos, acts_neg)
        pos_toks, neg_toks = project_logit_lens(model, tokenizer, v1, top_k=15)
        vectors[name] = v1
        results["factors"][name] = {
            "explained_variance_pct": ratio,
            "raw_mean_diff_norm": raw_norm,
            "top_pos_tokens": pos_toks,
            "top_neg_tokens": neg_toks,
        }
        print(f"=== {name} (SVD Var: {ratio:.2f}% | Raw Norm: {raw_norm:.2f}) ===")
        print("  Top+:", ", ".join(f"{t[1]}({t[2]:.1f})" for t in pos_toks[:10]))
        print("  Top-:", ", ".join(f"{t[1]}({t[2]:.1f})" for t in neg_toks[:10]))

    names = list(vectors.keys())
    print("\n=== Big Five 5x5 Cosine Similarity Matrix ===")
    header = "                     " + " ".join(f"{n[:7]:>7s}" for n in names)
    print(header)
    for n1 in names:
        results["cosine_matrix"][n1] = {}
        row = f"{n1:20s} "
        for n2 in names:
            c, d = cos_and_deg(vectors[n1], vectors[n2])
            results["cosine_matrix"][n1][n2] = {"cos": c, "deg": d}
            row += f"{c:+7.3f} "
        print(row)

    # 既存基底との比較
    ref_files = {
        "Clarity": "clarity_eigenvector_layer26.npz",
        "Pure_Clarity": "pure_clarity_eigenvector_layer26.npz",
        "Overview": "overview_eigenvector_layer26.npz",
    }
    loaded_refs = {}
    for rname, fname in ref_files.items():
        if os.path.exists(fname):
            v = np.load(fname)["vector"]
            loaded_refs[rname] = v / np.linalg.norm(v)

    if loaded_refs:
        print("\n=== Cosine Similarity vs Existing Layer 26 Basis ===")
        rnames = list(loaded_refs.keys())
        print("                     " + " ".join(f"{r:>12s}" for r in rnames))
        for n1 in names:
            results["existing_refs"][n1] = {}
            row = f"{n1:20s} "
            for rname in rnames:
                c, d = cos_and_deg(vectors[n1], loaded_refs[rname])
                results["existing_refs"][n1][rname] = {"cos": c, "deg": d}
                row += f"{c:+7.3f}({d:4.1f}) "
            print(row)

    np.savez_compressed("big5_ocean_layer26.npz", **vectors)
    with open("big5_ocean_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved big5_ocean_layer26.npz and big5_ocean_results.json")


if __name__ == "__main__":
    main()
