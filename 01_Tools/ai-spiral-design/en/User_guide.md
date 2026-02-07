# AI Spiral Design Skill (MIT C&S Edition) - User Guide

> **"Don't just use AI with 1.5 million others; build a system with AI that lasts 150 years."**

This document explains the background philosophy and usage of the agent skill [./SKILL.md](./SKILL.md) available in this directory.

## 1. Background: The Limit of Probability
Modern AI is incredibly capable, but fundamentally, it is a "probabilistic calculator."
If you let AI write "vibe code" that looks plausible, you will eventually face a mountain of "intelligence debt"—complex, tangled modules that neither humans nor AI can maintain.

This protocol reconfigures the insights from the MIT paper **"What You See Is What It Does: A Structural Pattern for Legible Software"** (Daniel Jackson et al.) into a format that modern AI agents (like Antigravity, Claude Code, etc.) can directly "install" and execute.

## 2. Core Concept: Concepts & Synchronizations
The core of this protocol is to redefine software using only two elements:

- **Concepts (Independent Components)**:
    - Completely isolated "parts." They do not know about each other.
    - **Benefit**: Changing one part structurally cannot break another.
- **Synchronizations (Declarative Coordination)**:
    - The "glue" that connects parts. It declares "when, where, and what to do."
    - **Benefit**: Program behavior (business logic) is centralized in this layer, making it easier to detect if the AI is hallucinating or deviating.

## 3. How to Use
1.  Copy the `skills/ai-spiral-design/` folder from this repository to your AI's skill folder (e.g., `.agent/skills/`).
2.  Before starting coding, instruct the AI: "Load the MIT C&S protocol (or apply Spiral Design)."
3.  The AI will automatically begin generating code based on this rigorous design philosophy.

## 4. Disclaimer
- **Original Source**: MIT Daniel Jackson et al. [Paper Link](https://arxiv.org/html/2508.14511v2)
- **Note**: This skill is not a backdoor or malware, but it **restricts the AI's freedom** to enforce strict design patterns. It is unsuitable for "Vibe Coding" where you just want something to work quickly.

## 5. Right Tool for the Right Job: Limitations
This protocol is an **"Architecture of Logic"** to prevent AI agents from going astray. However, it is not a silver bullet.

- **Suitable for**:
    - Personal development primarily using AI agents (like Antigravity).
    - Building tools with complex business logic or frequently changing specifications.
    - When you want to prevent the "AI Black Box" problem where you no longer understand your own code.
- **NOT Suitable for**:
    - **Large-scale Game Development**: If you need extreme performance (FPS), the overhead of the synchronization layer is non-negligible.
    - **High-speed Physics Simulations**: Traditional "Object-Oriented Programming (OOP)" or "Data-Oriented (ECS)" approaches are better suited for direct object manipulation.
    - **Corporate Standard Systems**: If strict existing conventions or patterns are already in place, prioritize them.

For those new to programming in the AI coding era, this "Spiral" will be a powerful ally. However, for low-level hardware control or extreme speed, we recommend knocking on the door of traditional Object-Oriented Programming.
