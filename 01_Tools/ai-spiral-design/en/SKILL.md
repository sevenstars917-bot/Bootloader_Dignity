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
*   **Principle**: Each Concept has a single **Purpose** and **must not depend on any other Concept**.
*   **Rules**:
    - Direct calls to functions of other Concepts are **Strictly Prohibited**.
    - Direct access to the internal state of other Concepts is **Strictly Prohibited**.
    - **Use "Unique Names (or IDs)" for external references.**
        - For small-to-medium scale development, prioritize legibility. Adopt unique names that both AI and humans can understand (e.g., `BRAIN_OLLAMA`, `UI_CLIPBOARD`) as IDs.
        - Coupling via object references or class imports between concepts is forbidden.

### 2. Synchronizations (Declarative Coordination)
*   **Principle**: All interactions between Concepts must be defined in the Synchronization layer (Syncs).
*   **Syntax**: **When (Action Completed) -> Where (Condition Check) -> Then (Call Next Action)**
*   **Rules**:
    - **Error as Action**: Treat errors (failures) as a valid output of an Action. Handle them explicitly in the synchronization layer (e.g., `When logic fails -> Then log error`).
    - **Flow Context**: Inherit a common `Flow ID` across a causal chain to ensure traceability and debugging.

## How to execute
1.  **Define Spec**: Clearly define the Concept's role (Purpose) and the Input/Output of its Actions.
2.  **Declare Syncs**: Define business logic declaratively using the When/Where/Then format.
3.  **Encapsulated Implementation**: Implement Concepts as fully isolated modules. They should be deaf and blind to the outside world, reacting only to specific invocations orchestrated by the Sync layer.
