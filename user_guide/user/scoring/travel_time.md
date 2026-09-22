# Travel-time scores (PTT / CTT)

Measure how long it takes to drive *within* a district using a finished
unit-to-unit drive-time table. These scores are **not** part of the compiled
`PlanEvaluator` engine: you build (or load) a `TravelTimeTable`, then call the
convenience functions or reusable metric objects.

Routes may leave the district (shortest paths use the full network). Scores are
easiest to interpret when the state OD is fully routable and plans are
road-contiguous.

## Weight modes

| Mode | Trip weight | Interpretation |
| --- | --- | --- |
| `none` | `1` | Each unit pair counts equally |
| `p+p` | `pop_a + pop_b` | Population-involved pair weight |
| `p*p` | `pop_a * pop_b` | Person-pair weight |

District scores use $\sum (t w) / (\#\ \mathrm{trips})$ so population weights stay
in the numerator: for PTT the trip count is the number of finite pairs, and for
CTT it is the number of finite centroid-to-unit trips. Units are time (`none`),
people-time per trip (`p+p`), or people²-time per trip (`p*p`). Weighted maxes
are $\max(t w)$ in the same weight units.


## Symmetric OD tables

Scoring reads each trip as `t(a, b)` and treats it as interchangeable with
`t(b, a)`. `TravelTimeTable` does not check or fix asymmetry. If you build a
table from a directed road network (one-ways), symmetrize it before scoring
(for example by taking the min or mean of the two directions).

## Centroid (CTT)

- `none`: choose unit $c$ minimizing $\sum_u t(c,u)$
- `p+p` / `p*p`: choose $c$ with $p_c > 0$ minimizing $\sum_u t(c,u)\,p_u$

(The $p_c$ self-penalty is intentionally omitted so centers are not pushed toward
empty units.)

## Example

```python
import networkx as nx
import gerrytools.scoring as gs

G = nx.Graph()
G.add_edge(10, 20, travel_time=10.0)
G.add_edge(20, 30, travel_time=10.0)
G.add_edge(20, 10, travel_time=10.0)
G.add_edge(30, 20, travel_time=10.0)

od = gs.build_od_from_graph(G, {0: 10, 1: 20, 2: 30})
parts = {1: [0, 1], 2: [1, 2]}
pops = {0: 1.0, 1: 1.0, 2: 1.0}

gs.pairwise_travel_time(parts, pops, od, weight="none", plan_agg="mean")
gs.CentroidTravelTime(weight="p*p", plan_agg="max_district").score(parts, pops, od)
```

See the {doc}`scoring API <../../api/scoring>` for full signatures under
`gerrytools.scoring.travel_time`.
