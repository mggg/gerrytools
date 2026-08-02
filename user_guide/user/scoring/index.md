# Scoring

GerryTools can score one assignment, a selected collection of plans, a GerryChain run, or a
recorded BEN/BENDL ensemble. Start with the overview, then use the guide for your workflow.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc

See the scoring interfaces, accepted assignment forms, metric families, and array formulas.
:::

:::{grid-item-card} Convenience Functions
:link: convenience_functions
:link-type: doc

Score one plan directly with simple functions.
:::

:::{grid-item-card} PlanEvaluator
:link: plan_evaluator
:link-type: doc

Prepare metrics once and reuse them across one plan, many plans, or recorded ensembles.
:::

:::{grid-item-card} Scoring BENDL files
:link: bendl
:link-type: doc

Inspect assignments, save Parquet scores, compare runs, and extend an existing score directory.
:::

:::{grid-item-card} Working with GerryChain
:link: basic
:link-type: doc

Use GerryTools metrics as updaters or evaluate selected partitions after a run.
:::

::::

The {doc}`scoring API <../../api/scoring>` contains the complete signatures, formulas, result
shapes, sign conventions, and resource requirements.

```{toctree}
:hidden:
:maxdepth: 1

Overview <overview>
Convenience Functions <convenience_functions>
PlanEvaluator <plan_evaluator>
Scoring BENDL files <bendl>
Working with GerryChain <basic>
```
