# Constraint Composition Harness

This folder is a standalone experimental harness for testing compositional
update rules without any learned model.

It compares:

- energy descent
- proximal/consensus composition
- weighted mixture composition

The harness reuses only the existing qualitative dataset and analytic barrier
functions from the main codebase.

Run:

```bash
python -m experiments.constraint_composition.runner --max-scenes 20 --steps 40
```
