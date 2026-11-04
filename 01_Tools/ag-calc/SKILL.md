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
   - `r1 = 10k` のように変数代入を行えば、次回のコマンド実行時にもその変数を数式内で利用可能（定数pi/eへの上書きは保護）。

---

## 🛠️ コマンド実行チートシート

ターミナル（シェル）からスクリプトの相対パスまたは配置パスを指定して実行せよ：

```bash
# 0. ヘルプ・実例チートシートの表示（困ったらこれを叩け）
python scripts/ag_calc.py

# 1. 数理・SI単位（T, G, M, k, m, u/μ/µ, n, p）＆ 工学・時間単位（F, H, Ω, V, A, W, Hz, s/ms/us/ns）
python scripts/ag_calc.py math "560k * 34.5u"
python scripts/ag_calc.py math "4.7µF * 1kΩ"
python scripts/ag_calc.py math "10V / 2A"
python scripts/ag_calc.py math "1s / 50ms"
python scripts/ag_calc.py math ".5k + 10n"
python scripts/ag_calc.py math "round(2.5)"               # 算術四捨五入（ROUND_HALF_UP -> 3）
python scripts/ag_calc.py math "hypot(3, 4)"             # 三平方の定理
python scripts/ag_calc.py math "gcd(24, 36)"             # 最大公約数 / lcm(4, 6)
python scripts/ag_calc.py math "clamp(15, 0, 10)"        # リミッター (10)
python scripts/ag_calc.py math --explain "1 / (2 * π * 10k * 0.1u)" # SI展開検算モード
python scripts/ag_calc.py math "parallel(100, 200, 300)" # 並列合成抵抗
python scripts/ag_calc.py math "root(-8, 3)"             # 実数奇数乗根対応 (-2)
python scripts/ag_calc.py math "10^(-1.5)"               # べき乗 (^ または **)
python scripts/ag_calc.py math --stat "-1, 2, 3, 4"      # 平均・分散・中央値

# 2. メモリー・変数チェーン（Ans & Store ※定数pi/e等は上書き保護済み）
python scripts/ag_calc.py math "ans / 2"
python scripts/ag_calc.py math "r1 = 10k"
python scripts/ag_calc.py mem
python scripts/ag_calc.py mem --clear

# 3. Active Vision & PTZサッカード幾何
python scripts/ag_calc.py vision fov --fov 60 --res 1920 --pos 1420
python scripts/ag_calc.py vision crop --res 1920x1080 --norm 0.2,0.3,0.8,0.7
python scripts/ag_calc.py vision crop --res 1920x1080 --norm 200,300,800,700 --scale 1000

# 4. テキスト・X判定・ハッシュ
python scripts/ag_calc.py text x "投稿ドラフト..."         # 280pt strict check (NFC, ❤️/ZWJ=2pt, 248ccTLD対応)
python scripts/ag_calc.py text hash "文字列"              # SHA256 & MD5

# 5. 時間・カレンダー
python scripts/ag_calc.py time diff "2026-07-10" "now"    # 経過日数・時間 (ISO 8601対応)
python scripts/ag_calc.py time now                        # JST ISO & Epoch
```

---

## 🚪 Exit Constraint (出力制約)

- 計算結果はターミナル標準出力の数値をそのまま回答に反映せよ。
- 一時ファイルは絶対に生成せず、計算履歴はスクリプト内包の `.ag_calc_state.json` に委ねよ。
