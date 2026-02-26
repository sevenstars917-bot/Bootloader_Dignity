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
    - **No "Logic Squatting" in UI/View**: Concepts like sensors, databases, or bridges must be implemented as independent classes or modules, isolated from the UI code.
    - **Always use a "unique name (or ID)" for external references.**
        - In small to medium-sized development, priority is given to legibility, and a unique name that AI and humans can understand may be adopted as an ID (e.g., `BRAIN_OLLAMA`, `UI_CLIPBOARD`).
        - However, direct object references or coupling through class imports are strictly prohibited.
    - **State Transparency**: Concept state should be "readable" from the outside (Sync layer), eliminating the chain of getter functions. Writing is performed only by Actions (Open for reading, strict for writing).pts is forbidden.

### 2. Synchronizations (Declarative Coordination)
*   **Principle**: All interactions between Concepts must be defined in the Synchronization layer (Syncs).
*   **Syntax**: **When (Action Completed OR Periodical Event) -> Where (Condition Check) -> Then (Update View OR Call Next Action)**
*   **Rules**:
    - **UI IS A PUPPET**: The View (UI) must not be an Orchestrator. It should only handle drawing and triggering events. Any decision-making (Polling loops, state transitions, remote requests) must be handled by a dedicated `Synchronizer` layer that controls the View like a puppet.
    - **Error as Action**: Treat errors (failures) as an intentional state. Handle them explicitly in the synchronization layer to trigger corrective actions or UI state changes (e.g., switching to an error expression).
    - **Flow Context**: Inherit a common `Flow ID` across a causal chain to ensure traceability.
    - **Action Traces (Causal Signature)**: Leave a signature on all action records indicating "which Sync rule executed it". This makes it possible to identify the cause by simply tracing back the causal graph during debugging. and debugging.

## How to execute
1.  **Define Spec**: Clearly define the Concept's role (Purpose) and the Input/Output of its Actions.
2.  **Declare Syncs**: Define business logic declaratively using the When/Where/Then format.
3.  **Encapsulated Implementation**: Implement Concepts as fully isolated modules. They should be deaf and blind to the outside world, reacting only to specific invocations orchestrated by the Sync layer.
