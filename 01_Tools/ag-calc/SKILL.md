---
name: ag-calc
description: 四則演算、浮動小数点計算、三角関数、べき乗、累乗根、並列抵抗、SI接頭辞（k/M/u等）、PTZカメラ画角・サッカード幾何、日付差分、エポック秒、X文字数判定（280pt）を求められた際、またはAIが内部的に決定論的な数値を算出する必要が生じた際、絶対に使い捨てのPythonスクリプトを自作せず、問答無用で本スキルの ag_calc.py を実行せよ。
---

# AG Scientific Co-Processor (`ag-calc`)

LLM（推論エンジン）のための外付け決定論的数理コプロセッサ CLI。
AI特有の「計算ミス（確率的トークン予測のハルシネーション）」や「使い捨てスクリプト生成（write_file/scratch.py）によるコンテキスト汚染」を物理的に根絶するための専用関数電卓。

---

## 🛑 行動規約 (Strict Axioms)

1. **スクリプト自作の厳禁**:
   - 算術計算、三角関数、単位換算、時間差分、画角計算のために `scratch.py` や `test.py` を書いてはならない。
   - 必ず本スキル付属の `scripts/ag_calc.py` をコマンド1発で実行せよ。
2. **外部依存ゼロ**:
   - Python標準ライブラリのみで駆動。ミリ秒単位で決定論的数値を返す。
3. **Ansメモリーと変数チェーン**:
   - 直前の計算結果は自動的に `ans` に保持される。
   - `r = 10k` のように変数代入を行えば、次回のコマンド実行時にもその変数を数式内で利用可能。

---

## 🛠️ コマンド実行チートシート

ターミナル（シェル）からスクリプトの相対パスまたは配置パスを指定して実行せよ：

```bash
# 0. ヘルプ・実例チートシートの表示（困ったらこれを叩け）
python scripts/ag_calc.py

# 1. 数理・SI単位（k, M, G, m, u/μ, n, p）
python scripts/ag_calc.py math "560k * 34.5u"
python scripts/ag_calc.py math "1 / (2 * π * 10k * 0.1u)"
python scripts/ag_calc.py math "parallel(100, 200, 300)"
python scripts/ag_calc.py math "root(125, 3)"
python scripts/ag_calc.py math "10^(-1.5)"
python scripts/ag_calc.py math --stat "[10, 20, 30, 40]"

# 2. メモリー・変数チェーン（Ans & Store）
python scripts/ag_calc.py math "ans / 2"
python scripts/ag_calc.py math "r = 10k"
python scripts/ag_calc.py mem
python scripts/ag_calc.py mem --clear

# 3. Active Vision & PTZサッカード幾何
python scripts/ag_calc.py vision fov --fov 60 --res 1920 --pos 1420
python scripts/ag_calc.py vision crop --res 1920x1080 --norm 0.2,0.3,0.8,0.7

# 4. テキスト・X判定・ハッシュ
python scripts/ag_calc.py text x "投稿ドラフト..."
python scripts/ag_calc.py text hash "文字列"

# 5. 時間・カレンダー
python scripts/ag_calc.py time diff "2026-07-10" "now"
python scripts/ag_calc.py time now
```

---

## 🚪 Exit Constraint (出力制約)

- 計算結果はターミナル標準出力の数値をそのまま回答に反映せよ。
- 一時ファイルは絶対に生成せず、計算履歴はスクリプト内包の `.ag_calc_state.json` に委ねよ。

