---
name: ai-spiral-design (English)
description: A Spiral Design Protocol based on MIT Concepts & Synchronizations to prevent "intelligence debt" caused by unstructured AI coding.
---

# AI Spiral Design Skill (MIT C&S Edition)

This skill applies the "Structural Pattern for Legible Software" proposed by the MIT research team (Daniel Jackson et al.) to maintain code integrity, ensuring that AI-generated code remains modular and does not collapse under complexity.

## When to use this skill
- When designing architecture for new software.
- When adding new features to an existing codebase without breaking existing functionality (The Lego Principle).
- When you want to suppress "vibe coding" (implementation based on loose context) and ensure engineering integrity.

## 🏗️ Architecture Standard

### 1. Concepts (Independent Components)
*   **Principle**: Each Concept has a single **Purpose** and **must not depend on any other Concept (including UI/View)**.
*   **Rules**:
    - Direct calls to functions of other Concepts are **Strictly Prohibited**.
    - Direct access to the internal state of other Concepts is **Strictly Prohibited**.
    - **No "Logic Squatting" in UI/View**: Concepts like sensors, databases, or bridges must be implemented as independent classes or modules, isolated from UI code.
    - **Always use a "unique name (or ID)" for external references.**
        - In collaborative AI development, use semantic unique names as IDs (e.g., `BRAIN_OLLAMA`, `UI_CLIPBOARD`) to prioritize legibility. Direct object references or coupling through class imports are strictly prohibited.
    - **State Transparency**: 
        - **"Open for Reading, Strict for Writing."**
        - Concept state should be "readable" directly from the outside (Sync layer), eliminating getter function chains. Writing is performed ONLY by Actions.
    - **Robustness**: When handling external processes, anticipate "Encoding Gravity" (UTF-8/SJIS mix). Use binary fetching and manual decoding or force environment variables to ensure stable data handling.

### 2. Synchronizations (Declarative Coordination)
*   **Principle**: All interactions between Concepts must be defined in the Synchronization layer (Syncs).
*   **Syntax**: **When (Action Completed) -> Where (State Check) -> Then (Update View OR Call Next Action)**
*   **Rules**:
    - **UI IS A PUPPET**: The View (UI) must not handle logic. It should be a "Dumb View" that only draws and triggers events. A dedicated `Synchronizer` must act as the Orchestrator, controlling the View.
    - **Error as Action**: Treat errors as valid results. Explicitly handle them in the Sync layer to manage state transitions or error handling logic.
    - **Flow Context**: Inherit a common `Flow ID` across a causal chain to ensure traceability.
    - **Action Traces (Causal Signature)**: Sign all action logs with "which Sync rule executed it." This allows AI (and humans) to trace back the causal graph to identify the root cause instantly.

### 3. 🛡️ UI Leak & Callback Prevention (Engineering Decalogue)
*   **No Callback Arguments**: Do not pass `on_success` or `on_error` functions into Concept asynchronous methods. This is an architectural failure that leaks UI-layer dependencies into the Concept's logic.
*   **Use Event Queues**: Put asynchronous action results into a thread-safe Queue as "event objects" (dictionaries). Only the Synchronizer should monitor the queue to update the UI.

## How to execute
1.  **Define Spec**: Before implementation, define the Concept's role (Purpose, State, Action names).
2.  **Declare Syncs**: Declare your "Operational Principles" (baseline scenarios) using the When/Where/Then format.
3.  **Encapsulated Implementation**: Implement Concepts as fully isolated modules. They should be "deaf and blind" to the rest of the system.
4.  **No Transactions**: Do not rely on massive atomic transactions. Build complex behaviors incrementally (spirally) by layering individual causal Sync rules.
