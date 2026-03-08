---
name: ai-spiral-design
description: AIがコードを書きすぎて自壊する「知性の不良債権」を防ぐための、MIT Concepts & Synchronizations に基づく螺旋(Spiral)型設計プロトコル。
---

# AI螺旋(Spiral)設計スキル (MIT C&S Edition)

このスキルは、MITの研究チームが提唱する「可読なソフトウェアのための構造パターン」を適用し、大規模開発におけるコードの整合性を維持するためのものです。

## When to use this skill
- 新しいソフトウェアのアーキテクチャ設計を行うとき
- 既存のコードベースに新しい機能を「既存機能を壊さず」に追加したいとき
- AIによる「バイブコーディング（雰囲気実装）」を抑制し、工学的な整合性を担保したいとき

## 🏗️ Architecture Standard

### 1. Concepts (独立した部品)
*   **原則**: 各Conceptは単一の目的（Purpose）を持ち、他のいかなるConcept（およびUI/View）にも依存しない。
*   **ルール**:
    - 他のConceptの関数を直接呼び出すことを禁止する。
    - 他のConceptの内部状態を直接参照することを禁止する。
    - **UI(View)の中にロジックを「間借」させてはならない。** センサー、データベース、通信等のConceptは、UIクラスから分離された独立したClassやモジュールとして実装せよ。
    - **外部参照には必ず「ユニークな名前（またはID）」を使用せよ。** 
        - 小〜中規模開発やAIとの共同作業では、可読性を優先し、一意の名前をIDとして採用せよ（例: `BRAIN_OLLAMA`, `UI_CLIPBOARD`）。直接的なオブジェクト参照やインポートによる結合は厳禁。
    - **State Transparency（状態の透明性）**: 
        - **「読み取り全開放、書き込み厳封」**。
        - Conceptの状態（State）は外部（Sync層）から直接「読み取り可能」とし、ゲッター関数の連鎖を排除せよ。書き込みはActionのみが行う。
    - **Robustness**: 外部プロセスを扱う場合は、バイナリ取得と手動デコード、または環境変数の強制により「文字コードの重力」を制御せよ。

### 2. Synchronizations (宣言的な連携)
*   **原則**: 概念同士の相互作用は、全てこの同期層（Syncs）で定義する。
*   **構文**: **When (Action完了) -> Where (状態チェック) -> Then (View更新 または 次Action)**
*   **ルール**:
    - **UIは「操り人形」である**: View(UI)クラスに、ポーリングや複雑な状態遷移の「判断ロジック」を持たせてはならない。Viewは命令を受けて描画するだけの受動的な存在（Dumb View）とし、`Synchronizer`がOrchestrator（指揮者）としてViewを制御せよ。
    - **Error as Action**: エラーも一つの正当なAction結果として捉え、同期層で「異常状態」への遷移として明示的にハンドリングせよ。
    - **Flow Context**: 一連の因果連鎖には共有の `Flow ID` を引き継ぎ、トレーサビリティを確保せよ。
    - **Action Traces（因果の署名）**: 全てのアクション記録に「どのSyncルールによって実行されたか」の署名を残せ。AI（私）が因果のグラフを遡るだけで原因を特定可能にする。

### 3. 🛡️ UI Leak & Callback Prevention (設計の掟)
*   **コールバック引数の禁止**: Conceptの非同期メソッドに `on_success` 等の関数を渡してはならない。これはUI層の都合をConceptに漏洩（Leaking）させる設計ミスである。
*   **Event Queue の活用**: 非同期アクションの結果は、スレッドセーフなキューに「イベントオブジェクト」として `put` せよ。Synchronizerだけがそのキューを監視し、UIを更新せよ。

## How to execute
1.  **Define Spec**: 実装前に Concept の役割（Purpose, State, Action）を定義せよ。
2.  **Declare Syncs**: 基準シナリオ（Operational Principle）を When/Where/Then 形式で宣言せよ。
3.  **Encapsulated Implementation**: Conceptを完全に独立したモジュールとして実装し、同期ロジックによってのみ連携させよ。
4.  **No Transactions**: 複雑な一括コミットに頼らず、個別の因果（Sync）の連動によって複雑な挙動をインクリメンタル（螺旋状）に構築せよ。
