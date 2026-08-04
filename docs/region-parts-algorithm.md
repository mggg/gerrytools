# Incremental region-parts algorithm

## Purpose

The region-parts metric counts connected components after restricting the graph by both a fixed
region and a proposed district. It does not merely count the number of region-district pairs that
occur.

For a finite undirected graph $G = (V, E)$, let $V_R \subseteq V$ be the nodes with a fixed region
label. Given $r: V_R \to \mathcal{R}$ and district assignment $d: V \to \mathcal{D}$, define

$$
V_{a,b} = \{v \in V_R \mid r(v) = a \land d(v) = b\}.
$$

The contribution of $(a,b)$ is the number of connected components in the induced subgraph
$G[V_{a,b}]$. If $\operatorname{cc}(H)$ denotes the number of connected components of $H$, a
region column's score is

$$
P(G,r,d) = \sum_{a \in \mathcal{R}} \sum_{b \in \mathcal{D}}
\operatorname{cc}\!\left(G[V_{a,b}]\right).
$$

Here $\operatorname{cc}(G[\varnothing]) = 0$, so summing over all labels is equivalent to summing
over only the represented region-district pairs.

Nodes with missing region labels are excluded. Edges may connect nodes outside a region, but those
edges cannot connect parts inside the region. Each prepared region column is scored independently.

## Why occupancy counting is incorrect

Consider one region of interest on a four-by-four rook-adjacency grid:

```text
---------
|x o o o|
|x x x x|
|x x o o|
|x o o o|
---------
```

The `x` nodes form one connected component. The top three `o` nodes form one component, and the
bottom five `o` nodes form another. This single region therefore contributes three parts, even
though it contains only two district labels.

The `o` nodes may connect outside the region, as in the larger schematic below, but that external
path is intentionally excluded from the induced regional subgraph:

```text
  -----------
x | x o o o | o
x | x x x x | o
x | x x o o | o
x | x o o o | o
  -----------
```

Both districts are globally connected in the larger map. The regional intersection still has three
parts. 

This extends naturally to disconnected graphs. If a region of interest is split 


```text
xx <space> oo
oo <space> xx
```

(two components), then for each component, there are two connected district subgraphs so this 
contributes a total of 4 parts.  

## State maintained by the incremental scorer

For each region column, the part scorer maintains:

- `component_of[node]`: the active component containing the node, or a sentinel for no component;
- `position[node]`: the node's position in that component's member vector;
- `components[id]`: the member nodes of each component;
- `component_keys[id]`: a packed `(region, district)` key;
- `free_components`: component IDs available for reuse;
- `parts`: the number of active components;
- generation-marked visitation state and reusable stacks for connectivity searches;
- dense reusable buckets that group removed nodes by their former component; and
- a small reusable list of additional neighboring components encountered during addition.

The packed component key is collision-free for every accepted input. Preparation rejects the
impractical case of more than $2^{48}$ regions before packing a 48-bit region ID with a 16-bit
district ID.

### Load-bearing invariants

After initialization and after every successful update:

1. Every node with a region label belongs to exactly one active component.
2. Every node in a component has the component's `(region, district)` key.
3. Every component is connected using only edges whose endpoints have that key.
4. Components are maximal: two active components with the same key have no edge between them.
5. `component_of`, `position`, and the component member vectors are exact inverses.
6. `parts` equals the number of active components.

The update algorithm exists to restore these invariants after a batch of simultaneous label
changes. The score itself is then just `parts`.

## Initialization and full scoring

Initialization scans the nodes in graph order. When it finds an included node without a component,
it allocates a component and performs a graph traversal restricted to the node's region and district.
Every reached node is assigned to that component. Each traversal discovers one maximal connected
component and increments `parts` once.

The stateless full scorer uses the same definition with a fresh visited vector. It is useful for
independent frames and as a reference implementation, but rebuilding all components for every local
Markov-chain change is wasteful.

## Incremental update

An update receives the assignment before the delta and a list of `(node, old, new)` changes.
Validation checks assignment length, node bounds, old-label agreement, district bounds, and strictly
increasing node IDs before state is mutated. BEN and BENDL decoding emit changes in graph-node order;
the incremental Rust interface requires callers to preserve that ordering.

All real changes are removed first, every affected old component is repaired, and only then are the
nodes added under their new keys:

```text
validate all changes

for every changed node:
    remove it from its old component
    group it with the other removals from that component

for every affected old component:
    repair connectivity after all simultaneous removals

for every changed node:
    add it under its new (region, district) key
```

Removing the entire batch before adding anything gives the batch simultaneous semantics. It prevents
a node's new label from being used as an accidental bridge while an old component is being repaired.

## Removing nodes and repairing components

Removing a node is constant time. Its component member vector uses `swap_remove`, and the position
of the swapped node is repaired. Removed nodes are collected in a dense bucket indexed by their old
component ID.

For each affected component:

1. If no members survive, release the component ID and decrement `parts`.
2. Otherwise, collect every surviving neighbor of every removed node that still belongs to the old
   component. These nodes are the boundary seeds.
3. If there are zero or one distinct seeds, the surviving component cannot have split.
4. Start a traversal from the first seed.
5. Stop immediately if that traversal reaches every seed.
6. If some seeds remain unreachable, traverse the fragment containing each unreached seed.
7. Perform a defensive sweep for any unvisited survivor.
8. Keep the largest fragment under the old component ID, allocate IDs for the others, and increase
   `parts` by the number of additional fragments.


### Worked batch removal: the old component does not split

Start with one solid seven-by-eight component. Remove the seven nodes in the interior T as one
simultaneous batch:

```text
x x x x x x x x
x x x . x x x x
x x x . x x x x
x . . . x x x x
x x x . x x x x
x x x . x x x x
x x x x x x x x
```

The batch produces 14 distinct boundary seeds on several sides of the removed set. The T does not
reach the component boundary, so surviving paths go around its top, bottom, and left arms. A
traversal from any seed eventually reaches all 14 seeds. The algorithm can then stop without
visiting every survivor. The surviving `x` nodes therefore remain one component. During this
removal-repair phase, the old component ID and `parts` remain unchanged.

This is not the score after the complete assignment update. The removed nodes are subsequently
added under their new `(region, district)` key. If all seven become a previously absent district,
their connected T forms one additional component, so the completed update increases `parts` from
one to two. They may instead join or merge existing components with that key, depending on the
surrounding assignment.

### Worked batch removal: genuine three-way split

Now let the vertical arm reach the top and bottom boundaries and the horizontal arm reach the left
boundary:

```text
x x x . x x x x
x x x . x x x x
x x x . x x x x
. . . . x x x x
x x x . x x x x
x x x . x x x x
x x x . x x x x
```

The ten removed nodes separate the survivors into a top-left fragment, a bottom-left fragment, and a
larger right fragment. The first traversal exhausts one fragment while seeds remain unreached. The
later traversals discover the other two. The right fragment retains the old component ID, two new
IDs are allocated, and `parts` increases from one to three.

### Why the boundary seeds are complete

Let $C$ be one connected component before removal, let $X \subseteq C$ be the removed nodes, and
define the boundary-seed set

$$
S = \{u \in C \setminus X \mid \exists x \in X \text{ such that } \{u,x\} \in E\}.
$$

For every connected component $K$ of $C \setminus X$,

$$
K \cap S \neq \varnothing.
$$

To see why, choose $u \in K$ and $x \in X$. Because $C$ was connected, a path joined $u$ to $x$
before removal. Starting inside $K$, that path must eventually leave $K$. It cannot first enter a
different component of $C \setminus X$, because an edge between the two would make them the same
component. The first edge leaving $K$ must therefore enter $X$. Its endpoint in $K$ belongs to $S$.
Consequently, every surviving fragment contains at least one boundary seed.

This proves the early exit: if one traversal reaches every seed, there cannot be another fragment.
Any other fragment would contain an unreached seed. The traversal may stop without visiting every
interior node because those nodes keep the old component ID and need no relabeling.

The defensive survivor sweep is retained as a final completeness check. Under the invariants above,
the seed argument already accounts for every fragment.

## Adding nodes and merging components

After every old component is repaired, changed nodes are inserted under their new keys. For one node:

1. Scan its already assigned neighbors.
2. Ignore neighbors with a different region or district key.
3. Keep the first matching component in a scalar.
4. Record only additional distinct matching components in a small reusable vector.
5. If there are no matching components, create a singleton component and increment `parts`.
6. If there is one matching component, append the node to it.
7. If there are several, append the node to the largest component, move the smaller component
   members into it, release their IDs, and decrement `parts` once per merge.

```text
before addition       after addition

x x . x x             x x x x x
```

The new center node joins two components with the same key. Two parts become one, so `parts`
decreases by one.

### Why sequential addition is safe

After some prefix of the additions has been processed, the state contains the exact connected
components induced by the surviving nodes and that prefix. Adding one node either has no same-key
edge, attaches to one component, or joins every same-key component adjacent to it. Those are exactly
the possible changes caused by adding one vertex.

An edge between two changed nodes is considered when the later endpoint is added. By induction, after
the last addition, components are connected and maximal for the completed batch. Choosing the largest
target changes only the amount of relabeling, not the result.

## Why pairwise cut-edge summaries are not enough

For a two-district change between $A$ and $B$, the new $A$-$B$ boundary is useful. A surviving $A$
neighbor of a removed $A$ node is now an endpoint of an $A$-$B$ cut edge. The removal algorithm
uses this information in a tighter form by considering only boundary seeds created by the current
delta, not every unchanged $A$-$B$ cut edge in the plan.

The stronger proposal is to retain a part count for every realized district pair, possibly together
with the number of raw cut edges for that pair. The two rook-adjacency maps below have the same graph
and region boundary. Their pairwise summaries are identical:

$$
q_{AB}=3,\qquad q_{AC}=2,\qquad q_{BC}=3,
$$

where $q$ counts distinct parts incident to the pair, and

$$
e_{AB}=3,\qquad e_{AC}=2,\qquad e_{BC}=2,
$$

where $e$ counts raw cut edges. Nevertheless, the first map has four parts and the second has five.

The counterexample is illustrated here:

```text
B1 C1 C1 B2 A1     C1 B1 A1 A1 C2
A1 A1 A1 A1 A1     B2 A1 A1 A1 A1
```

*Summary for both maps *

| district pair | A-B | A-C | B-C |
| ------------- | :---: | :---: | :---: |
| incident pieces | 3 | 2 | 3 |
| raw cut edges | 3 | 2 | 2 | 

In the first map, the $C$ part counted for the $A$-$C$ pair is the same part counted for the
$B$-$C$ pair. In the second map, those appearances belong to two disconnected $C$ parts. Counts by
district pair discard that cross-pair identity. A part with no in-region cut edge would be invisible
to the pairwise summary as well.

Attaching a persistent component ID to each pairwise appearance makes the summary sufficient because
the IDs reveal which appearances must be deduplicated. That is the information maintained by
`component_of`. When a removal may divide one component ID, connectivity still has to determine how
many replacement IDs are required.

There is also no free boundary list in a TwoDelta frame. The frame supplies changed node labels, not
changed edges. Discovering the new $A$-$B$ cut edges requires scanning the changed nodes' adjacency
lists, which is the same graph information needed by addition.

Cut edges therefore identify where connectivity may have changed, but their pairwise counts do not
determine the new connectivity. The boundary-seed traversal supplies exactly the missing information.


## Complexity

Let:

- $k$ be the number of changed nodes;
- $E_{\Delta}$ be the incident adjacency entries scanned for those nodes;
- $R_{\Delta}$ be the nodes and adjacency entries traversed while repairing removals; and
- $M_{\Delta}$ be the members relabeled while merging smaller components.

Sorted validation is $O(k)$. The update is

$$
O\!\left(k + E_{\Delta} + R_{\Delta} + M_{\Delta}\right).
$$

In the worst case, a genuine split can require a full traversal of a large component. In the common
no-split case, repair stops as soon as every boundary seed is connected. Memory is
$O\!\left(\lvert V \rvert + \lvert E \rvert\right)$ plus component member storage and reused
scratch buffers.

## Verification

Correctness is checked at several levels:

- named examples for regional clipping, articulation splits, merging, missing regions, and
  simultaneous changes;
- exhaustive assignments on a four-node graph;
- generated full scores checked against an independent union-find oracle;
- generated incremental updates checked against both full recomputation and the oracle;
- Standard, MkvChain, TwoDelta, and bundle stream equivalence tests; and
- large Florida and Tennessee comparisons
