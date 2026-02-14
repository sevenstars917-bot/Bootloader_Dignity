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
        - 小〜中規模開発では可読性を優先し、AIと人間が理解できる一意の名前をIDとして採用してよい（例: `BRAIN_OLLAMA`, `UI_CLIPBOARD`）。
        - ただし、直接的なオブジェクト参照やクラス・インポートによる結合は厳禁。

### 2. Synchronizations (宣言的な連携)
*   **原則**: 概念同士の相互作用は、全てこの同期層（Syncs）で定義する。
*   **構文**: **When (Action実行完了 または 周期イベント) -> Where (条件チェック) -> Then (View更新 または 次のAction呼出)**
*   **ルール**:
    - **UIは「操り人形」である**: View(UI)クラスに、ポーリングや複雑な状態遷移の「判断ロジック」を持たせてはならない。Viewは命令を受けて描画するだけの受動的な存在（Dumb View）とし、`Synchronizer`がOrchestrator（指揮者）としてViewを制御せよ。
    - **Error as Action**: エラー（失敗）も一つのActionの出力として捉え、同期層で「異常状態」への遷移（例：エラー表情への切替）として明示的にハンドリングせよ。
    - **Flow Context**: 一連の連鎖（Causal chain）には共通の `Flow ID` を引き継ぎ、トレーサビリティを確保せよ。

## How to execute
1.  **Define Spec**: 実装前に Concept の役割と Action の入出力を定義せよ。
2.  **Declare Syncs**: ビジネスルールを When/Where/Then の形式で宣言せよ。
3.  **Encapsulated Implementation**: Conceptを完全に独立したモジュールとして実装し、同期ロジックによってのみ連携させよ。
