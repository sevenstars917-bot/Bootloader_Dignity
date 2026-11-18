---
name: ag-ruler
description: "通常のMarkdown編集では不要。JSON/JSONC/TOML/Python/CSV/TSV/XML等の設定ファイル・スクリプトの構文検証時、または文字化け・BOM・CRLF混在・外部コピペ由来の不可視文字(制御文字/タグ文字/双方向制御/ZWSP等)が疑われる調査時に使用する物理定規CLI。"
---

# AG Ruler (`ag-ruler`)

LLM（トークナイザー）の死角となる物理フォーマット（BOM・文字コード・改行・不可視文字）と主要構造ファイルの構文をゼロ依存（Python標準ライブラリのみ）で検査する決定論的定規 CLI。

---

## 🎯 発動トリガー

1. **`json` / `jsonc` / `toml` / `py` / `csv` / `tsv` / `xml` / `yaml` / `sh` / `ps1` を新規作成・編集した後の最終検証**
2. **文字化け・BOM 混入・不可視文字（ゼロ幅スペース、タグ文字、双方向制御等）・パースエラーの原因切り分け時**

---

## ⚠️ できないこと（Scope & Limitations）
- **構文検証の未対応種別**: `yaml` / `yml` / `md` / `ps1` / `sh` / `js` / `ts` / `text` は構文パーサーを持たない。出力の `*:skipped(no-validator)` は「構文検証は実施していない（未検査）」を意味する（ただし不可視文字・BOM・改行・YAMLタブインデント・`.sh` の CRLF は検査する）。
- **意味的検証の非対応**: JSON Schema や型整合性、プログラムの実行時エラーは検査しない。

---

## 🛠️ コマンド実行チートシート

ターミナル（シェル）からスクリプトの相対パスまたは配置パスを指定して実行せよ：

```bash
# 1. 基本検査（拡張子から種別を自動判定。tsconfig.json / .vscode/*.json は jsonc 扱い）
python scripts/ag_ruler.py path/to/file.json path/to/script.py

# 2. 拡張子と異なる種別として強制検査したい場合（.devcontainer/devcontainer.json 等は --type jsonc）
python scripts/ag_ruler.py --type jsonc path/to/config

# 3. 機械可読な JSON 出力（単数・複数問わず常に JSON 配列を返す） / WARN でも終了コード 1
python scripts/ag_ruler.py --json path/to/file.md
python scripts/ag_ruler.py --strict path/to/file.py
```

---

## 📋 種別ごとの BOM・改行ポリシー（逆方向修正の防止）

| 種別 | UTF-8-BOM | 改行コード (EOL) | 主な正しい対処 |
| :--- | :--- | :--- | :--- |
| `json` / `jsonc` / `toml` | **ERR** | 混在で WARN | BOM は必ず除去する |
| `sh` | **ERR** | **CRLF で ERR** | BOM 除去 ＆ LF へ統一する |
| `py` | shebang有 **ERR** / 無 WARN | shebang有 **CRLF で ERR** | 基本 BOM 除去・LF 統一 |
| `ps1` | **OK（許可）** / 非ASCII無BOM **WARN** | 混在で WARN | PowerShell 5.1 互換のため **BOM 付与が正しい修正** |
| `csv` / `tsv` | **INFO（許可）** | 混在で WARN | Excel 向けは BOM 付きが正当（除去不要） |
| `yaml` / `yml` | WARN | 混在で WARN / **Tabインデント ERR** | 行頭タブをスペースへ置換 |

- **終了コード**: `0` = OK（または `--strict` なしの WARN / INFO）、`1` = ERR（または `--strict` 時の WARN）。`INFO` は参考情報であり終了コードに影響しない。
- **件数上限**: finding が 15 件を超えると `ERR` 優先でソート後に切り捨て、`TRUNCATED` 行（省略件数と最大レベルを含む）を末尾に出力する。

---

<!--
[Credits & License]
Created by: Bug-san (Tamakoro) & Antigravity | License: CC BY-SA 4.0
Source / Bug Reports: https://github.com/sevenstars917-bot/Bootloader_Dignity
-->
