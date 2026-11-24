//! 1-dimensional Weisfeiler-Lehman (1-WL) color refinement.
//!
//! For directed graphs, a vertex signature contains:
//!
//! 1. the vertex's current color,
//! 2. the sorted colors of its incoming neighbors,
//! 3. the sorted colors of its outgoing neighbors.
//!
//! Both graphs are interned into the same palette during each round, so equal
//! color IDs mean equal signatures across the two graphs. If the color
//! histograms ever differ, the graphs cannot be isomorphic.
//!
//! Passing this filter is only a necessary condition. For example, every
//! vertex of two different regular graphs with the same degree receives the
//! same color in every round.
//!
//! # References
//! <https://algorithm.tamnd.com/weisfeiler-leman/>

use crate::graph::{BitMatrix, Graph, Vertex, Word, WordsPerRow};
use std::collections::HashMap;

/// Color assigned to a vertex during refinement.
pub type Color = u32;

/// Signature of a vertex in the refinement process:
/// `(current_color, incoming_neighbor_colors, outgoing_neighbor_colors)`.
type Signature = (Color, Vec<Color>, Vec<Color>);

/// Palette of signatures to color IDs. Used to intern signatures into stable
/// color IDs across both graphs.
type Palette = HashMap<Signature, Color>;

/// Refines vertex colors in both graphs until the partition stabilizes.
///
/// Returns `None` when the color histograms differ, which proves that the
/// graphs are not isomorphic. Returns `Some((colors_a, colors_b))` otherwise;
/// this means 1-WL could not separate the graphs, not that they are
/// isomorphic.
///
/// The two returned color vectors use the same color IDs, so a color that
/// appears in both vectors denotes the same signature in both graphs.
///
/// # Panics
///
/// Panics if the graphs have different vertex counts.
pub(super) fn refine(
    graph_a: &Graph,
    graph_b: &Graph,
    graph_a_t: &Graph,
    graph_b_t: &Graph,
) -> Option<(Vec<Color>, Vec<Color>)> {
    assert_eq!(
        graph_a.num_vertices, graph_b.num_vertices,
        "1-WL refinement requires graphs with the same vertex count"
    );

    // (current color, incoming neighbor colors, outgoing neighbor colors) -> color ID
    let mut palette = Palette::new();
    let mut next_id: Color = 0;

    let n = graph_a.num_vertices;

    // Initial colors encode the in-degree and out-degree. Interned in the
    // shared palette, equal initial colors mean equal degree pairs across the
    // two graphs.
    let mut colors_a: Vec<Color> = (0..n)
        .map(|u| {
            intern(
                &mut palette,
                &mut next_id,
                (
                    0,
                    vec![graph_a_t.out_degree(u)],
                    vec![graph_a.out_degree(u)],
                ),
            )
        })
        .collect();

    let mut colors_b: Vec<Color> = (0..n)
        .map(|v| {
            intern(
                &mut palette,
                &mut next_id,
                (
                    0,
                    vec![graph_b_t.out_degree(v)],
                    vec![graph_b.out_degree(v)],
                ),
            )
        })
        .collect();

    let mut previous_color_count = 0usize;

    for _ in 0..n {
        // Histogram equality is a necessary condition for isomorphism.
        if !same_histogram(&colors_a, &colors_b) {
            return None;
        }

        let mut unique = colors_a.clone();
        unique.sort_unstable();
        unique.dedup();

        // Refinement only splits color classes, so if the number of classes
        // stops growing the partition is stable. The number of classes in
        // both graphs is the same because the histograms matched above.
        if unique.len() == previous_color_count || unique.len() == n as usize {
            break;
        }
        previous_color_count = unique.len();

        // IDs are local to one round. The signature includes the previous
        // round's colors, so reusing IDs from round to round is unnecessary.
        palette.clear();
        next_id = 0;

        let next_colors_a: Vec<Color> = (0..n)
            .map(|u| {
                intern(
                    &mut palette,
                    &mut next_id,
                    vertex_signature(graph_a, graph_a_t, u, &colors_a),
                )
            })
            .collect();

        let next_colors_b: Vec<Color> = (0..n)
            .map(|v| {
                intern(
                    &mut palette,
                    &mut next_id,
                    vertex_signature(graph_b, graph_b_t, v, &colors_b),
                )
            })
            .collect();

        colors_a = next_colors_a;
        colors_b = next_colors_b;
    }

    if !same_histogram(&colors_a, &colors_b) {
        return None;
    }

    Some((colors_a, colors_b))
}

/// Builds a row-major bit matrix of color-compatible vertex pairs.
///
/// Bit `(u, v)` is set when `colors_a[u] == colors_b[v]`.
/// `words_per_row` is the number of [`Word`]s needed for one row
/// (`n.div_ceil(Word::BITS)`).
///
/// Rows correspond to vertices of the first graph and columns to vertices of
/// the second graph. Ullman's search in [`super::ullman`] uses this matrix to
/// restrict candidate mappings.
pub(super) fn build_candidate_matrix(
    colors_a: &[Color],
    colors_b: &[Color],
) -> (BitMatrix, WordsPerRow) {
    assert_eq!(
        colors_a.len(),
        colors_b.len(),
        "candidate matrix requires equal color-vector lengths"
    );

    let n = colors_a.len();
    let bits_per_word = Word::BITS as usize;
    let words_per_row = n.div_ceil(bits_per_word);
    let mut matrix: BitMatrix = vec![0; n * words_per_row];

    for (u, &color_u) in colors_a.iter().enumerate() {
        let row_offset = u * words_per_row;
        for (v, &color_v) in colors_b.iter().enumerate() {
            if color_u == color_v {
                matrix[row_offset + v / bits_per_word] |= 1 << (v % bits_per_word);
            }
        }
    }

    (matrix, words_per_row)
}

/// Returns the stable ID for `signature`, assigning a fresh one if needed.
fn intern(palette: &mut Palette, next_id: &mut Color, signature: Signature) -> Color {
    if let Some(&id) = palette.get(&signature) {
        return id;
    }

    let id = *next_id;
    *next_id += 1;
    palette.insert(signature, id);
    id
}

/// Builds the refinement signature of `vertex`.
///
/// Incoming neighbors come from the transposed graph; outgoing neighbors
/// come from the original graph. Sorting makes the signature independent of
/// neighbor order.
fn vertex_signature(
    graph: &Graph,
    graph_transposed: &Graph,
    vertex: Vertex,
    colors: &[Color],
) -> Signature {
    let mut in_colors: Vec<Color> = graph_transposed
        .neighbors(vertex)
        .iter()
        .map(|&neighbor| colors[neighbor as usize])
        .collect();
    let mut out_colors: Vec<Color> = graph
        .neighbors(vertex)
        .iter()
        .map(|&neighbor| colors[neighbor as usize])
        .collect();

    in_colors.sort_unstable();
    out_colors.sort_unstable();

    (colors[vertex as usize], in_colors, out_colors)
}

/// Checks whether two color vectors contain the same multiset of colors.
fn same_histogram(colors_a: &[Color], colors_b: &[Color]) -> bool {
    let mut sorted_a = colors_a.to_vec();
    let mut sorted_b = colors_b.to_vec();
    sorted_a.sort_unstable();
    sorted_b.sort_unstable();
    sorted_a == sorted_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_histogram() {
        assert!(same_histogram(&[1, 2, 3], &[3, 2, 1]));
        assert!(!same_histogram(&[1, 2, 3], &[4, 5, 6]));
    }

    #[test]
    fn test_intern() {
        let mut palette = Palette::new();
        let mut next_id: Color = 0;
        let sig1 = (1, vec![2], vec![3]);
        let sig2 = (1, vec![2], vec![3]);
        let sig3 = (4, vec![5], vec![6]);

        let id1 = intern(&mut palette, &mut next_id, sig1);
        let id2 = intern(&mut palette, &mut next_id, sig2);
        let id3 = intern(&mut palette, &mut next_id, sig3);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_vertex_signature_undirected() {
        let edges = [(0, 1), (1, 2), (2, 0)];
        let graph = Graph::new(3, &edges, false);
        let graph_t = graph.transpose();
        let colors: Vec<Color> = vec![0, 1, 2];

        assert_eq!(
            vertex_signature(&graph, &graph_t, 0, &colors),
            (0, vec![1, 2], vec![1, 2])
        );
    }

    #[test]
    fn test_vertex_signature_directed() {
        // Directed path 0 -> 1 -> 2.
        let edges = [(0, 1), (1, 2)];
        let graph = Graph::new(3, &edges, true);
        let graph_t = graph.transpose();
        let colors: Vec<Color> = vec![0, 1, 2];

        // Vertex 1 has incoming neighbor 0 and outgoing neighbor 2.
        assert_eq!(
            vertex_signature(&graph, &graph_t, 1, &colors),
            (1, vec![0], vec![2])
        );
    }

    #[test]
    fn test_build_candidate_matrix() {
        let colors_a = vec![0, 1, 0];
        let colors_b = vec![1, 0, 0];
        let (matrix, words_per_row) = build_candidate_matrix(&colors_a, &colors_b);

        assert_eq!(words_per_row, 1);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0], 0b110); // Vertex 0 in A (color 0) matches vertices 1 and 2 in B
        assert_eq!(matrix[1], 0b001); // Vertex 1 in A (color 1) matches vertex 0 in B
        assert_eq!(matrix[2], 0b110); // Vertex 2 in A (color 0) matches vertices 1 and 2 in B
    }

    #[test]
    fn test_build_candidate_matrix_empty() {
        let (matrix, words_per_row) = build_candidate_matrix(&[], &[]);

        assert!(matrix.is_empty());
        assert_eq!(words_per_row, 0);
    }

    #[test]
    fn test_refine_accepts_same_graph() {
        let edges = [(0, 1), (1, 2), (2, 0)];
        let graph_a = Graph::new(3, &edges, false);
        let graph_b = Graph::new(3, &edges, false);
        let graph_a_t = graph_a.transpose();
        let graph_b_t = graph_b.transpose();

        let (colors_a, colors_b) =
            refine(&graph_a, &graph_b, &graph_a_t, &graph_b_t).expect("1-WL should not separate");
        assert_eq!(colors_a, colors_b);
    }

    #[test]
    fn test_refine_rejects_different_degree_histograms() {
        // 4-vertex path versus 4-vertex star: same vertex and edge counts,
        // different degree sequences.
        let path = Graph::new(4, &[(0, 1), (1, 2), (2, 3)], false);
        let star = Graph::new(4, &[(0, 1), (0, 2), (0, 3)], false);
        let path_t = path.transpose();
        let star_t = star.transpose();

        assert!(refine(&path, &star, &path_t, &star_t).is_none());
    }

    #[test]
    fn test_refine_cannot_separate_regular_graphs() {
        // C6 and 2 x C3 are both 2-regular and non-isomorphic. 1-WL assigns
        // every vertex the same color in both graphs, so it returns Some.
        // The exact search, not this filter, must reject the pair.
        let cycle = Graph::new(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], false);
        let triangles = Graph::new(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], false);
        let cycle_t = cycle.transpose();
        let triangles_t = triangles.transpose();

        assert!(refine(&cycle, &triangles, &cycle_t, &triangles_t).is_some());
    }
}
