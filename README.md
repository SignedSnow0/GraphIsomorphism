# Objective

The objective of this project is to evaluate graph isomoprhism and the speed using a CPU only approach and GPU.

### Glossary

- **Graph**: a tuple $G=(V,E)$ where $V$ is the set of vertices and $E\subseteq V\times V$ is the set of edges.
- **Adjacency**: two vertices $u,v$ are adjacent if there is an edge connecting the two $(u,v)\in E$.
- **Ismorphism**: a (invertible) function $f:V(G)\rightarrow V(H)$ that preserves the adjacency of the two graphs.

### Uses

Graph adjacency are useful in many different sectors:

- Chemistry: molecule identification
- Cybersecurity: malvare detection (code obfuscation)
- Computer science: pattern recognition in ASTs or dependency graphs

## Algorithm

We have in input two graphs $G,H$ and we want to check for isomorphism. In order to speed up the algorithm we
can check the two graphs using increasingly expensive computational functions.

The first thing we can do is to check some graph invariants with low cost:

1. Both graphs must have the same vertex count: $|V_G|=|V_H|$
2. Both graphs must have the same edge count: $|E_G|=|E_H|$
3. Both graphs must have the same in and out degree: $\text{deg}_{-}(V_{G,i})=\text{deg}_{-}(V_{H,i})$ and $\text{deg}_{+}(V_{G,i})=\text{deg}_{+}(V_{H,i})\quad\forall i\in[0,N]$

after these relatively simple checks, any remaining graphs are checked using more complex algorithms:

4. 1-WL color refinement
5. Full search using Ullman's algorithm

### Weisfeiler-Lehman color refinement

The 1-WL algorithm takes a graph and gives each vertex a color, after a number of theps the colors converge to a stable distribution. If the distributions of the two graphs match the graph might be isomorphic.

**In detail**:
We start by inizializing all colors in the graph. Then for each vertex a new color is calculated bease on its own color and the colors of its neighbors.
After a number of steps the algorithm will generate the same colors.
Build two histograms with the colors of graph $G$ and $H$, it the two match the graph might be isomorphic, otherwise it is not.

### Ullman's algorithm

Ullman's algorithm is a search algorithm used to find subgraph isomorphism.
Is it built on top of a candiate matrix $M$ where $M_{ij}=1$ if vertex $i$ might map to vertex $j$.

1. The initial candidate matrix is build, in our case it is given by WL algorithm.
2. All items $M_{ij}$ are refined by checking if every neighbor $x$ of $V_{G,i}$ maps to at least a neighbor in $V_{G,j}$ (Arc consistency search)
3. Because changing the value in a cell might affect other cells, repeat until convergence. If any row is all zeroes, the mapping is invalid.
4. Search each row using DFS, in each column update $M'$ by setting the mapping $d\rightarrow j$, zero out row $d$ (except $j$) and remove any mapping to $j$ in every row $\ge d$
5. Repeat the refinement check until all vertices are assigned $d=n$

## Graph representation

To favour GPU usage the graph is represented using Compressed Sparse Row format.
A graph contains:

- The number of vertices in the graph
- The source vertex list
- The destination vertex list

For example: `s=[0, 2, 2, 4]` and `d=[1, 2, 0, 1]`, then the vertex `n` has edges `N=s[n+1]-s[n]` and their values are in `d[n..n+N]`, so vertex $0$ has $2-0=2$ vertices and they are $d[0]=1,d[1]=2$

### Parallelization opportunities

- The vertex degree check can be done in parallel for each node
- The final search can be parallelized, but some modifications have to be implemented to support GPU hardware limitations. The best algorithms for GPUs are Ullman's of VF2, but since CubeCL does not support dynamic parallelism Ullman's algorithm is the best choice.

### Tests an comparison

The main comparison metrix is speed, the test will run the algorithm on both CPU and GPU to compare the running times.
The test can focus on varius densities to check the best representations for different graphs.

## Additional features

Planar graphs can be checked for isomorphism in $O(n)$ time, and a graph
can be checked if it is planar in $O(n)$ time.
