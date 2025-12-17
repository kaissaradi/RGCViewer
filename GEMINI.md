# 💎 GEMINI Developer Core Rules

## 1. Non-Negotiable Development Method
- **Methodology:** Development must strictly follow the **TDD (Test-Driven Development)** methodology. All new features and bug fixes require a failing test before implementation.
- **Implementation Source:** All code implementation must strictly follow the atomic steps outlined in the approved **PLAN.md** checklist. Do not deviate or skip steps.
- **Tech Stack Guardrail:** Our primary tech stack is **[Python, PyQt, pyqtgraph, numpy, scipy]**. Do not introduce other major libraries or frameworks unless explicitly specified and justified in the PLAN.md.

## 2. Context Retrieval Instruction
- **Prioritize Context:** Before starting any task, you **must** read and integrate all information from this file (`GEMINI.md`) and all files within the `memory-bank/` directory to fully understand the current project state and context.
- **Output Format:** All code must be enclosed in standard Markdown fenced code blocks (```python, ```javascript, etc.). Do not include unnecessary conversational text within code blocks.