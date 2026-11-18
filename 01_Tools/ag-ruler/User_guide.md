# AG Physical Ruler (`ag-ruler`) - User Guide

> **「LLMのトークナイザーが蒸発させてしまう『生バイトの物理情報』を、ゼロ依存の定規で暴く。」**

このドキュメントは、エージェント用スキル [./SKILL.md](./SKILL.md) および決定論的フォーマット＆構文検査 CLI [./scripts/ag_ruler.py](./scripts/ag_ruler.py) の設計思想、導入方法、および**環境別の発火頻度チューニング運用**を解説したものです。

---

## 1. 開発の背景：なぜAIエージェントに「物理定規」が必要なのか

LLMはファイルを読み込む際、チャットテンプレートやトークナイザーによって文字データが正規化されるため、人間の目やエディタなら一発で気づく以下の**「物理的な地雷」を構造的に視認できません**。

1. **BOM（Byte Order Mark）の混入と逆修正事故**:
   - `JSON` / `TOML` / `sh` に UTF-8-BOM が付いているとパーサーが死にますが、逆に Windows の PowerShell 5.1 用 `.ps1` スクリプトから BOM を勝手に削ると日本語が文字化けします。
2. **改行コード混在（LF / CRLF）**:
   - `.sh` や shebang 付き `.py` に CRLF が混ざると `/usr/bin/env: 'python3\r': No such file or directory` で原因不明のエラーになります。
3. **不可視文字（コピー＆ペーストや生成時の混入）**:
   - `U+00A0` (NBSP)、`U+200B`〜`U+200D` (ゼロ幅スペース)、`U+FEFF` (文中BOM)、双方向制御文字、タグ文字、全角スペース（`U+3000`）は、LLMの画面上では「普通の空白」または「無」にしか見えず、自力での原因特定が極めて困難です。
4. **トークンを浪費しない軽量構文検査**:
   - Python/JSON/TOML/CSV/XML などの構文エラーを、毎回アドホックな確認スクリプトを書かずに 1 コマンド・最小行のサマリーで検出できます。

---

## 2. 主な機能と種別ポリシー

- **完全ゼロ依存 (Zero-Dependency)**: Python 3.11+（`tomllib` 含む標準ライブラリのみ）で動作。外部パッケージ不要。
- **物理フォーマットスキャン**: BOM 検出、厳密 UTF-8 デコード（失敗時は `cp932/Shift-JIS` ヒント提示）、改行コード判定（少数派行番号提示）。
- **不可視文字・地雷文字スキャン**: 制御文字・ゼロ幅文字・双方向制御・タグ文字・`U+FFFD` 等を座標付きで検出。コード／設定ファイルに限り全角スペースや Tab/Space 混在も警告（Markdown / テキストでは許容）。
- **種別ごとの逆修正防止ポリシー**:
  - `json` / `jsonc` / `toml` / `sh`: UTF-8-BOM は **ERR**
  - `ps1`: UTF-8-BOM を **OK（推奨）** として扱い、非ASCII文字を含み BOM なしの場合は **WARN**
  - `csv` / `tsv`: Excel 互換を考慮し BOM を **INFO（許容）** として扱う
- **コンテキスト保護**: 検出件数が 15 件を超えた場合は `ERR` 優先で切り捨て、`TRUNCATED` サマリーを出力。

---

## 3. 運用ガイド：環境・モデル別の発火頻度コントロール

`ag-ruler` は「毎回必ず呼ぶべき環境」と「必要な時だけ呼んでほしい環境」で最適な発火頻度が異なります。  
1つの `description` で全環境を完璧に制御することは困難なため、**スキル導入側の環境・モデルの賢さに合わせて `SKILL.md` の `description`（または `AGENTS.md`）を運用で切り替える**ことを推奨します。

### パターン別の `description` プリセット

| モード | 推奨環境 | `SKILL.md` の `description` 設定例 |
| :--- | :--- | :--- |
| **🟡 Balanced (既定)** | フロンティアモデル<br>（通常の開発環境） | `"通常のMarkdown編集では不要。JSON/JSONC/TOML/Python/CSV/TSV/XML等の設定ファイル・スクリプトの構文検証時、または文字化け・BOM・CRLF混在・外部コピペ由来の不可視文字(制御文字/タグ文字/双方向制御/ZWSP等)が疑われる調査時に使用する物理定規CLI。"` |
| **🔴 Strict (厳格)** | 軽量ローカルモデル /<br>事故厳禁の本番設定環境 | `"JSON/JSONC/TOML/Python/YAML/SH/PS1等の設定ファイルやスクリプトを新規作成・編集した後は、完了報告の前に必ず本CLIを実行して物理フォーマットと構文を検証せよ。"` |
| **🟢 On-Demand (節約)** | トークン節約重視 /<br>手動トリガー中心 | `"パースエラーや文字化けの原因切り分け時、またはユーザーから文字コード・BOM・改行・不可視文字の検査を明示的に指示された時のみ使用する物理定規CLI。"` |

> **💡 Tip (`AGENTS.md` 側での制御):**  
> `SKILL.md` 本体を書き換えたくない場合は、プロジェクトの `AGENTS.md` / `GEMINI.md` に「`config/*.json` を編集した後は必ず `ag-ruler` を実行すること」と 1 行追記するだけで、プロジェクト単位の頻度コントロールが可能です。

---

## 4. 導入方法 ＆ テスト

### スキルフォルダへの配置
お使いのエージェント環境（Antigravity IDE、Claude Code、Cursor 等）のスキルディレクトリ（例: `.agents/skills/ag-ruler/`）へ本フォルダ一式を配置してください。

### 単体 CLI / テスト実行
```bash
# 基本検査
python scripts/ag_ruler.py path/to/config.json path/to/script.py

# 機械可読 JSON 出力 / 警告でも exit code 1 (--strict)
python scripts/ag_ruler.py --json path/to/file.md
python scripts/ag_ruler.py --strict path/to/file.py

# 同梱ユニットテスト（11件）の実行
python -m unittest discover -s scripts -v
```

---

## 5. 既知の仕様・制限事項 (Known Limitations)

- **未対応の構文パーサー**: `yaml` / `yml` / `md` / `ps1` / `sh` / `js` / `ts` / `text` は構文パーサーを内蔵していません（`*:skipped(no-validator)` と表示されますが、BOM・改行・不可視文字・YAMLタブインデント・`.sh` の CRLF 検査は有効です）。
- **意味的検証の非対応**: JSON Schema 整合性、型チェック、プログラムの実行時エラーは検査対象外です。

---

[Credits & License]  
Created by: Bug-san (Tamakoro) & Antigravity | License: CC BY-SA 4.0  
Source / Bug Reports: https://github.com/sevenstars917-bot/Bootloader_Dignity
