---
name: run-research-experiment
description: Plan, execute, and record reproducible R&D experiments. Use when the user asks to try an idea, reproduce a paper, benchmark or ablate a method, validate a dataset or converter, run a smoke test, compare configurations, or gather empirical evidence.
---

# Run Research Experiment

Read [the experiment workflow](../../workflows/02_run_experiment.md) completely
and follow it.

Frame the experiment as **Why → How**: name the unresolved problem and the
observable success criterion before choosing a model, training procedure or
benchmark. For every experimental choice, state which problem or failure mode
it tests.

Before running commands, read [the verified command guide](../../04_commands.md)
and the relevant code or vendor README. Keep large raw logs and dataset
inspection out of the main context; use targeted extraction or a subagent.
