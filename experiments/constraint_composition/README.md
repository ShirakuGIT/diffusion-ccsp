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

Time-conditioned vector field:

```bash
python -m experiments.constraint_composition.runner \
  --suite vector_time \
  --max-scenes 20 \
  --steps 40
```

Graph-conditioned noise field:

```bash
python -m experiments.constraint_composition.runner \
  --suite graph_noise \
  --max-scenes 20 \
  --steps 40
```

Graph flow-matching field:

```bash
python -m experiments.constraint_composition.runner \
  --suite graph_flow \
  --max-scenes 20 \
  --steps 40
```

Graph DAgger field:

```bash
python -m experiments.constraint_composition.runner \
  --suite graph_dagger \
  --max-scenes 20 \
  --steps 40
```

Graph score field:

```bash
python -m experiments.constraint_composition.runner \
  --suite graph_score \
  --max-scenes 20 \
  --steps 40
```

Graph score++ field:

```bash
python -m experiments.constraint_composition.runner \
  --suite graph_score_plus \
  --max-scenes 20 \
  --steps 40
```
