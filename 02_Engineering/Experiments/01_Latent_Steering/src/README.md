# Experiments Reproduction Source Code (再現実験スクリプト集)

本ディレクトリは、Apple Silicon (M6 / 32GB) 上の MLX ネイティブ環境において実施された一連のステアリング・自律神経実験の**完全な再現スクリプト一式**です。

---

## 🛠️ 動作環境 (Prerequisites)
* **Hardware**: Apple Silicon Mac (M1〜M6 / Unified Memory 24GB以上推奨)
* **OS**: macOS 15.0+ (実験時は macOS 27.0)
* **Python**: Python 3.12+
* **Dependencies**:
  ```bash
  pip install mlx==0.32.4 mlx-lm==0.32.2 numpy
  ```
* **Base Model**: `mlx-community/gemma-4-26b-a4b-it-4bit` (Hugging Face / 自動ダウンロード)

---

## 📂 スクリプト一覧と対応レポート

| スクリプト | 概要 / 役割 | 対応実験レポート |
| :--- | :--- | :--- |
| **`scan_all_layers.py`** | 全30層走査によるJ空間特異点（第26層 / 分離度81%）の特定 | [全層J空間スキャンとSVD主成分ステアリングの実証](../実験レポート_Gemma4_26B_全層J空間スキャンとSVD主成分ステアリングの実証.md) |
| **`extract_eigenvector_mlx.py`** | 第26層 Clarity 差分行列からのSVD第一主成分ベクトル抽出 | 同上 |
| **`benchmark_n_tests.py`** | $n=5$ マルチタスクベンチマーク（バットとボール問題、リンダ問題等） | 同上 |
| **`track_narrative_trajectory.py`** | 全30層における Hidden State 状態軌跡追跡（物語 vs 命令の直交分岐） | [ナラティブと中間層ステアリングの機能境界](../実験レポート_大言語モデルにおけるナラティブと中間層ステアリングの機能境界.md) |
| **`experiment_a_narrative_vs_rules.py`** | 箇条書き設定 (Rules) vs 一人称物語 (Narrative) の思考チャネル比較 | 同上 |
| **`experiment_b_secretary_steering.py`** | 第26層 ゼロトークンステアリング検証 | 同上 |
| **`experiment_c_layer18_steering.py`** | 第18層 単独ステアリング検証（消去法） | 同上 |
| **`experiment_blue_flame_synthesis.py`** | 情熱 (Passion) ＋ 冷静 (Calm) の幾何学角度測定（75.8°直交）と青の炎合成 | 同上 |
| **`experiment_trinity_steering.py`** | 情熱 ＋ 冷静 ＋ 確信の3×3相関行列と三位一体（Trinity）神速即答実証 | 同上 |
| **`experiment_self_medication_clarity.py`** | 認知的デッドロック直面時の自律的ツール呼び出し（人工アドレナリン注射）実証 | [自律的セルフ・ステアリングの実証](../実験レポート_大言語モデルにおける自律的セルフステアリング実証_人工アドレナリンによる能動的覚醒.md) |
| **`experiment_bio_vs_fighter_modes.py`** | ノルム過変調死 (Overdose) vs ノルム1.00固定重心シフト (Fighter Boost) 比較 | [過変調境界と重心シフトの実測検証](../実験レポート_自律神経ステアリングの過変調境界と重心シフト_実測検証.md) |
| **`server.py`** | 第26層ステアリングを常時マウントした OpenAI 互換 FastAPI サーバー | 実戦配備用サーバー |

---

## 🚀 再現手順の一例

```bash
# 1. 依存ライブラリのインストール
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm numpy

# 2. 全層軌跡スキャンの再現
python track_narrative_trajectory.py

# 3. 三位一体（Trinity）直交基底の角度測定と即答テスト
python experiment_trinity_steering.py

# 4. 自律神経セルフ・メディケーション（Clarityボタン）の再現
python experiment_self_medication_clarity.py

# 5. 過変調死 vs 重心シフトの比較再現
python experiment_bio_vs_fighter_modes.py
```

