---
name: research-paper
description: Read and evaluate research papers or long technical documents with traceable evidence. Use when the user asks to read a paper, analyze a PDF, explain a method, compare papers, connect paper claims to code, review related work, or create a research note.
---

# Research Paper

Read [the paper-reading workflow](../../workflows/01_read_paper.md) completely
and follow it.

Structure the analysis around **Why → How**. Establish the concrete problem,
failure or limitation first. Then explain every method, modeling, training and
benchmark choice in terms of which part of that problem it addresses and what
evidence shows that it works. Do not produce a component inventory detached
from the paper's problem statement.

For long PDFs or broad literature searches, isolate extraction in a subagent and
return only page/section references, claims, evidence, and unresolved questions
to the main context. Keep the final synthesis in the main task.

Read [the workspace overview](../../01_overview.md) before relating findings to
this workspace.
