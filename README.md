# Objective

The objective of this project is to evaluate graph isomoprhism and the speed using a CPU only approach and GPU.

### Algorithm

1. Prune basic cases: check $|V_a|=|V_b|$ and $|E_a|=|E_b|$
2. Compare vertex degrees in a matrix: $M[u][v]=deg(u)==deg(v) ? 1 : 0$
3. 1-WL color refinement
4. If the pair passes al basic cases, run either BFS or DFS to check for isomorphism

### Parallelization opportunities

- The vertex degree check can be done in parallel for each node
- The final search can be parallelized, but some modifications have to be implemented to support GPU hardware limitations. The best algorithms for GPUs are Ullman's of VF2, but since CubeCL does not support dynamic parallelism Ullman's algorithm is the best choice.

### Tests an comparison

The main comparison metrix is speed, the test will run the algorithm on both CPU and GPU to compare the running times.
The test can focus on varius densities to check the best representations for different graphs.
