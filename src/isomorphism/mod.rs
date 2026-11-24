//! Graph isomorphism testing.
//!
//! The decision procedure uses increasingly expensive filters:
//!
//! 1. Compare vertex and arc counts.
//! 2. Compare the sorted multiset of `(in_degree, out_degree)` pairs.
//! 3. Run 1-dimensional Weisfeiler-Lehman (1-WL) color refinement. If the
//!    color histograms differ, the graphs are definitely not isomorphic.
//! 4. Run Ullman's candidate-matrix refinement and fixed-order backtracking
//!    search over the remaining equal-color candidates.
//!
//! Steps 1-3 are necessary conditions and reject many non-isomorphic pairs
//! cheaply. Step 4 is the actual decision procedure. Graphs that 1-WL cannot
//! distinguish, in particular regular graphs, are decided by the search.

mod ullman;
mod weisfeiler_lehman;

use crate::graph::{DegreePair, Graph, Vertex};

/// Maps each vertex of `graph_a` to its image in `graph_b`.
pub struct Mapping(Vec<Vertex>);

impl std::fmt::Display for Mapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, value) in self.0.iter().enumerate() {
            write!(f, "({} -> {})", i, value)?;
            if i < self.0.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

/// Decides whether `graph_a` and `graph_b` are isomorphic.
///
/// The graphs are treated as simple, unweighted graphs. For directed graphs,
/// edge direction is part of the structure being compared.
///
/// The function returns `None` as soon as a cheap invariant or 1-WL rejects
/// the pair. If those filters pass, Ullman's algorithm refines the remaining
/// candidate matrix and performs a fixed-order backtracking search.
///
/// The fixed order makes the branches independent, which is useful for a
/// future parallel or GPU implementation. See the `ullman` module for details.
///
/// The worst-case running time is exponential in the number of vertices.
/// Graphs that are hard for color refinement, such as large regular graphs,
/// are the expensive cases.
pub fn find_isomorphism(graph_a: &Graph, graph_b: &Graph) -> Option<Mapping> {
    if graph_a.num_vertices != graph_b.num_vertices || graph_a.num_edges() != graph_b.num_edges() {
        return None;
    }

    let graph_a_t = graph_a.transpose();
    let graph_b_t = graph_b.transpose();

    if !check_degree(graph_a, graph_b, &graph_a_t, &graph_b_t) {
        return None;
    }

    let (colors_a, colors_b) = weisfeiler_lehman::refine(graph_a, graph_b, &graph_a_t, &graph_b_t)?;

    let (candidate_matrix, words_per_row) =
        weisfeiler_lehman::build_candidate_matrix(&colors_a, &colors_b);

    ullman::find_isomorphism(
        graph_a,
        graph_b,
        &graph_a_t,
        &graph_b_t,
        &candidate_matrix,
        words_per_row,
    )
}

/// Returns the sorted multiset of `(in_degree, out_degree)` pairs.
///
/// Isomorphism preserves this multiset, so comparing it is a valid and
/// labeling-invariant necessary condition. Comparing degrees index by index
/// would incorrectly reject labeled copies of the same graph.
fn degree_sequence(graph: &Graph, graph_transposed: &Graph) -> Vec<DegreePair> {
    let mut degrees: Vec<DegreePair> = (0..graph.num_vertices)
        .map(|vertex| {
            (
                graph_transposed.out_degree(vertex),
                graph.out_degree(vertex),
            )
        })
        .collect();
    degrees.sort_unstable();
    degrees
}

/// Checks whether two graphs have the same degree multiset.
fn check_degree(graph_a: &Graph, graph_b: &Graph, graph_a_t: &Graph, graph_b_t: &Graph) -> bool {
    degree_sequence(graph_a, graph_a_t) == degree_sequence(graph_b, graph_b_t)
}

#[cfg(test)]
mod tests {
    use crate::graph::{BitMatrix, Edge, Vertex, VertexCount, Word};

    use super::*;

    /// Builds an undirected test graph.
    fn graph(num_vertices: VertexCount, edges: &[Edge]) -> Graph {
        Graph::new(num_vertices, edges, false)
    }

    /// Builds a directed test graph.
    fn directed_graph(num_vertices: VertexCount, edges: &[Edge]) -> Graph {
        Graph::new(num_vertices, edges, true)
    }

    /// Applies a vertex permutation to an edge list.
    fn permute_edges(edges: &[Edge], permutation: &[Vertex]) -> Vec<Edge> {
        edges
            .iter()
            .map(|&(u, v)| (permutation[u as usize], permutation[v as usize]))
            .collect()
    }

    /// Calls `f` once for every permutation of `0..n`.
    fn for_each_permutation(n: usize, f: &mut impl FnMut(&[Vertex])) {
        fn recurse(k: usize, permutation: &mut [Vertex], f: &mut impl FnMut(&[Vertex])) {
            if k == permutation.len() {
                f(permutation);
                return;
            }
            for i in k..permutation.len() {
                permutation.swap(k, i);
                recurse(k + 1, permutation, f);
                permutation.swap(k, i);
            }
        }

        let mut permutation: Vec<Vertex> = (0..n as Vertex).collect();
        recurse(0, &mut permutation, f);
    }

    /// Computes a canonical adjacency bit matrix by trying every relabeling.
    ///
    /// This is only used as a brute-force oracle for tiny graphs in tests.
    fn canonical_form(graph: &Graph) -> BitMatrix {
        let n = graph.num_vertices as usize;
        let bits_per_word = Word::BITS as usize;
        let words = n.div_ceil(bits_per_word);
        let mut best: Option<BitMatrix> = None;

        for_each_permutation(n, &mut |permutation| {
            let mut permuted: BitMatrix = vec![0; n * words];
            for u in 0..n {
                for &v in graph.neighbors(u as Vertex) {
                    let from = permutation[u] as usize;
                    let to = permutation[v as usize] as usize;
                    permuted[from * words + to / bits_per_word] |= 1 << (to % bits_per_word);
                }
            }

            if best.as_ref().is_none_or(|current| permuted < *current) {
                best = Some(permuted);
            }
        });

        best.unwrap_or_default()
    }

    /// Calls `f` once for every graph on `num_vertices` vertices.
    fn for_each_graph(num_vertices: VertexCount, directed: bool, f: &mut impl FnMut(Graph)) {
        let pairs: Vec<Edge> = if directed {
            (0..num_vertices)
                .flat_map(|u| {
                    (0..num_vertices)
                        .filter(move |&v| v != u)
                        .map(move |v| (u, v))
                })
                .collect()
        } else {
            (0..num_vertices)
                .flat_map(|u| (u + 1..num_vertices).map(move |v| (u, v)))
                .collect()
        };

        for mask in 0..(1u64 << pairs.len()) {
            let edges: Vec<Edge> = pairs
                .iter()
                .enumerate()
                .filter_map(|(i, &edge)| (mask & (1u64 << i) != 0).then_some(edge))
                .collect();
            f(Graph::new(num_vertices, &edges, directed));
        }
    }

    #[test]
    fn test_same_graph_is_isomorphic() {
        let edges = [(0, 1), (1, 2), (2, 0)];
        let graph_a = graph(3, &edges);
        let graph_b = graph(3, &edges);

        assert!(find_isomorphism(&graph_a, &graph_b).is_some());
    }

    #[test]
    fn test_different_edge_count_is_not_isomorphic() {
        let graph_a = graph(3, &[(0, 1), (1, 2), (2, 0)]);
        let graph_b = graph(3, &[(0, 1), (1, 2)]);

        assert!(find_isomorphism(&graph_a, &graph_b).is_none());
    }

    #[test]
    fn test_different_vertex_count_is_not_isomorphic() {
        let graph_a = graph(3, &[(0, 1), (1, 2), (2, 0)]);
        let graph_b = graph(4, &[(0, 1), (1, 2), (2, 0)]);

        assert!(find_isomorphism(&graph_a, &graph_b).is_none());
    }

    #[test]
    fn test_different_degree_sequence_is_not_isomorphic() {
        // A 4-vertex path and a 4-vertex star have the same vertex and edge
        // counts, but their degree sequences differ.
        let path = graph(4, &[(0, 1), (1, 2), (2, 3)]);
        let star = graph(4, &[(0, 1), (0, 2), (0, 3)]);

        assert!(find_isomorphism(&path, &star).is_none());
    }

    #[test]
    fn test_relabeled_path_is_isomorphic() {
        // Both graphs are the path 0-1-2, but the labels differ. This case
        // catches index-by-index degree comparisons.
        let graph_a = graph(3, &[(0, 1), (1, 2)]);
        let graph_b = graph(3, &[(1, 0), (0, 2)]);

        assert!(find_isomorphism(&graph_a, &graph_b).is_some());
    }

    #[test]
    fn test_relabeled_graph_with_isolated_vertices() {
        let graph_a = graph(4, &[(0, 1)]);
        let graph_b = graph(4, &[(2, 3)]);

        assert!(find_isomorphism(&graph_a, &graph_b).is_some());
    }

    #[test]
    fn test_relabeled_cycle_is_isomorphic() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
        let permutation = [3, 0, 5, 1, 4, 2];
        let permuted_edges = permute_edges(&edges, &permutation);

        assert!(find_isomorphism(&graph(6, &edges), &graph(6, &permuted_edges)).is_some());
    }

    #[test]
    fn test_cycle_and_two_triangles_are_not_isomorphic() {
        // Both graphs are 2-regular with 6 vertices and 6 edges. 1-WL assigns
        // every vertex the same color in both graphs, so only the exact
        // search can reject this pair.
        let cycle = graph(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]);
        let triangles = graph(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]);

        assert!(find_isomorphism(&cycle, &triangles).is_none());
    }

    #[test]
    fn test_three_by_three_bipartite_and_prism_are_not_isomorphic() {
        // Both graphs are 3-regular with 6 vertices and 9 edges. They are a
        // standard 1-WL-indistinguishable non-isomorphic pair.
        let complete_bipartite = graph(
            6,
            &[
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
            ],
        );
        let prism = graph(
            6,
            &[
                (0, 1),
                (1, 2),
                (2, 0),
                (3, 4),
                (4, 5),
                (5, 3),
                (0, 3),
                (1, 4),
                (2, 5),
            ],
        );

        assert!(find_isomorphism(&complete_bipartite, &prism).is_none());
    }

    #[test]
    fn test_directed_cycle_and_two_directed_triangles_are_not_isomorphic() {
        // Every vertex has in-degree 1 and out-degree 1 in both graphs, so
        // 1-WL cannot distinguish them.
        let cycle = directed_graph(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]);
        let triangles = directed_graph(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]);

        assert!(find_isomorphism(&cycle, &triangles).is_none());
    }

    #[test]
    fn test_relabeled_directed_cycle_is_isomorphic() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
        let permutation = [4, 2, 0, 5, 3, 1];
        let permuted_edges = permute_edges(&edges, &permutation);

        assert!(
            find_isomorphism(
                &directed_graph(6, &edges),
                &directed_graph(6, &permuted_edges)
            )
            .is_some()
        );
    }

    #[test]
    fn test_empty_graphs_are_isomorphic() {
        assert!(find_isomorphism(&graph(0, &[]), &graph(0, &[])).is_some());
    }

    #[test]
    fn test_single_vertex_graphs_are_isomorphic() {
        assert!(find_isomorphism(&graph(1, &[]), &graph(1, &[])).is_some());
    }

    #[test]
    fn test_graph_with_isolated_vertices_is_not_empty_graph() {
        assert!(find_isomorphism(&graph(1, &[]), &graph(2, &[])).is_none());
    }

    #[test]
    fn test_matches_brute_force_for_small_undirected_graphs() {
        for num_vertices in 0..=4 {
            let mut graphs = Vec::new();
            for_each_graph(num_vertices, false, &mut |graph| graphs.push(graph));

            let canonical_forms: Vec<BitMatrix> = graphs.iter().map(canonical_form).collect();

            for (i, graph_a) in graphs.iter().enumerate() {
                for (j, graph_b) in graphs.iter().enumerate() {
                    assert_eq!(
                        find_isomorphism(graph_a, graph_b).is_some(),
                        canonical_forms[i] == canonical_forms[j],
                        "undirected mismatch for n={num_vertices}, i={i}, j={j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_matches_brute_force_for_small_directed_graphs() {
        for num_vertices in 0..=3 {
            let mut graphs = Vec::new();
            for_each_graph(num_vertices, true, &mut |graph| graphs.push(graph));

            let canonical_forms: Vec<BitMatrix> = graphs.iter().map(canonical_form).collect();

            for (i, graph_a) in graphs.iter().enumerate() {
                for (j, graph_b) in graphs.iter().enumerate() {
                    assert_eq!(
                        find_isomorphism(graph_a, graph_b).is_some(),
                        canonical_forms[i] == canonical_forms[j],
                        "directed mismatch for n={num_vertices}, i={i}, j={j}"
                    );
                }
            }
        }
    }
}
