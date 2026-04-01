# Constraint Composition Harness

This folder is a standalone experimental harness for testing compositional
update rules without any learned model.

It compares:

- energy descent
- proximal/consensus composition
- weighted mixture composition
- projected energy
- prototype-based local energy

The harness reuses only the existing qualitative dataset and analytic barrier
functions from the main codebase.

Run:

```bash
python -m experiments.constraint_composition.runner --max-scenes 20 --steps 40
```

Prototype ablation:

```bash
python -m experiments.constraint_composition.runner \
  --suite prototype \
  --max-scenes 20 \
  --steps 40 \
  --prototype-k 5 10 20 \
  --prototype-tau 0.01 0.05 0.1 0.5
```

Learned energy:

```bash
python -m experiments.constraint_composition.runner \
  --suite learned \
  --max-scenes 20 \
  --steps 40
```

Global learned energy:

```bash
python -m experiments.constraint_composition.runner \
  --suite global \
  --max-scenes 20 \
  --steps 40
```

Vector field:

```bash
python -m experiments.constraint_composition.runner \
  --suite vector \
  --max-scenes 20 \
  --steps 40
```

Unrolled vector field:

```bash
python -m experiments.constraint_composition.runner \
  --suite vector_unrolled \
  --max-scenes 20 \
  --steps 40
```
