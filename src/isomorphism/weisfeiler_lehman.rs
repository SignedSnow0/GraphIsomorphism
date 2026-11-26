//! 1-dimensional Weisfeiler-Lehman (1-WL) color refinement.
//!
//! For directed graphs, a vertex signature contains:
//!
//! 1. the vertex's current color,
//! 2. the colors of its incoming neighbors,
//! 3. the colors of its outgoing neighbors.
//!
//! The CPU and GPU implementations use the same hash-based colors. Instead of
//! interning signatures in a palette, each signature is hashed directly into a
//! [`Color`]. Neighbor colors are order-independent: each is mixed with
//! [`mix32`] and accumulated with both `sum` and `xor`.
//!
//! Equal signatures always produce equal colors. A hash collision can merge
//! two color classes, which only weakens the filter; it cannot reject an
//! isomorphic pair. If the color histograms ever differ, the graphs cannot be
//! isomorphic.
//!
//! Passing this filter is only a necessary condition. For example, every
//! vertex of two different regular graphs with the same degree receives the
//! same color in every round.
//!
//! # References
//! <https://algorithm.tamnd.com/weisfeiler-leman/>

use crate::graph::{BitMatrix, Graph, Vertex, Word, WordsPerRow};

/// Color assigned to a vertex during refinement.
///
/// This is `u32` to match the GPU color buffers.
pub type Color = u32;

/// Refines vertex colors in both graphs until the colors stabilize.
///
/// Returns `None` when the color histograms differ, which proves that the
/// graphs are not isomorphic. Returns `Some((colors_a, colors_b))` otherwise;
/// this means 1-WL could not separate the graphs, not that they are
/// isomorphic.
///
/// The two returned color vectors use the same hash function, so a color that
/// appears in both vectors denotes the same signature in both graphs, up to
/// hash collisions.
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

    let n = graph_a.num_vertices;

    // Initial colors encode the in-degree and out-degree. The same initial
    // color is computed by the GPU `init_kernel`.
    let mut colors_a: Vec<Color> = (0..n)
        .map(|u| initial_color(graph_a, graph_a_t, u))
        .collect();

    let mut colors_b: Vec<Color> = (0..n)
        .map(|v| initial_color(graph_b, graph_b_t, v))
        .collect();

    for _ in 0..n {
        if !same_histogram(&colors_a, &colors_b) {
            return None;
        }

        let next_colors_a: Vec<Color> = (0..n)
            .map(|u| hash_vertex(graph_a, graph_a_t, u, &colors_a))
            .collect();

        let next_colors_b: Vec<Color> = (0..n)
            .map(|v| hash_vertex(graph_b, graph_b_t, v, &colors_b))
            .collect();

        let stable = next_colors_a == colors_a && next_colors_b == colors_b;

        colors_a = next_colors_a;
        colors_b = next_colors_b;

        if stable {
            break;
        }
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

/// Computes the initial color of a vertex from its in-degree and out-degree.
///
/// This mirrors the GPU `init_kernel`:
///
/// ```text
/// h_in  = mix32(in_degree)
/// h_out = mix32(out_degree)
/// color = hash_palette(0, h_in, h_in, h_out, h_out)
/// ```
///
/// Passing each degree twice matches the GPU kernel's `sum == xor == h`
/// representation of the initial one-element neighbor multisets.
fn initial_color(graph: &Graph, graph_transposed: &Graph, vertex: Vertex) -> Color {
    let in_degree = graph_transposed.out_degree(vertex);
    let out_degree = graph.out_degree(vertex);

    let h_in = mix32(in_degree);
    let h_out = mix32(out_degree);

    hash_palette(0u32, h_in, h_in, h_out, h_out)
}

/// Computes the hashed 1-WL signature of one vertex.
///
/// Incoming neighbors come from the transposed graph; outgoing neighbors come
/// from the original graph. For each neighbor color `c`, the hash contribution
/// is `mix32(c)`. The contributions are accumulated with wrapping `sum` and
/// `xor`, so the result does not depend on neighbor order.
///
/// This mirrors the GPU `hash_vertex` function.
fn hash_vertex(graph: &Graph, graph_transposed: &Graph, vertex: Vertex, colors: &[Color]) -> Color {
    let (in_sum, in_xor) = hash_neighbor_colors(graph_transposed.neighbors(vertex), colors);
    let (out_sum, out_xor) = hash_neighbor_colors(graph.neighbors(vertex), colors);

    hash_palette(colors[vertex as usize], in_sum, in_xor, out_sum, out_xor)
}

/// Aggregates neighbor colors into an order-independent `(sum, xor)` pair.
fn hash_neighbor_colors(neighbors: &[Vertex], colors: &[Color]) -> (u32, u32) {
    let mut sum = 0u32;
    let mut xor = 0u32;

    for &neighbor in neighbors {
        let h = mix32(colors[neighbor as usize]);
        sum = sum.wrapping_add(h);
        xor ^= h;
    }

    (sum, xor)
}

/// 32-bit MurmurHash3 finalizer.
///
/// This is the CPU equivalent of the GPU `mix32` function. Overflow is
/// intentional and wraps modulo 2^32.
fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^= x >> 16;
    x
}

/// Hashes one 1-WL signature.
///
/// The signature consists of:
///
/// - `seed`: the vertex's previous color (`0` during initialization),
/// - incoming neighbor color aggregates: `in_sum`, `in_xor`,
/// - outgoing neighbor color aggregates: `out_sum`, `out_xor`.
///
/// This is the CPU equivalent of the GPU `hash_palette` function.
fn hash_palette(seed: Color, in_sum: u32, in_xor: u32, out_sum: u32, out_xor: u32) -> Color {
    let mut h = mix32(seed ^ 0x9e37_79b9);

    h = mix32(h.wrapping_mul(0x85eb_ca6b).wrapping_add(in_sum));
    h = mix32(h.wrapping_mul(0xc2b2_ae35).wrapping_add(in_xor));
    h = mix32(h.wrapping_mul(0x85eb_ca6b).wrapping_add(out_sum));
    h = mix32(h.wrapping_mul(0xc2b2_ae35).wrapping_add(out_xor));

    h
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
    fn test_hash_palette_is_deterministic() {
        let first = hash_palette(1, 2, 3, 4, 5);
        let second = hash_palette(1, 2, 3, 4, 5);

        assert_eq!(first, second);
    }

    #[test]
    fn test_hash_vertex_uses_incoming_and_outgoing_neighbors() {
        // Directed path 0 -> 1 -> 2. Vertex 1 has incoming neighbor 0 and
        // outgoing neighbor 2.
        let edges = [(0, 1), (1, 2)];
        let graph = Graph::new(3, &edges, true);
        let graph_t = graph.transpose();
        let colors: Vec<Color> = vec![0, 1, 2];

        let (in_sum, in_xor) = hash_neighbor_colors(graph_t.neighbors(1), &colors);
        let (out_sum, out_xor) = hash_neighbor_colors(graph.neighbors(1), &colors);

        assert_eq!(in_sum, mix32(0));
        assert_eq!(in_xor, mix32(0));
        assert_eq!(out_sum, mix32(2));
        assert_eq!(out_xor, mix32(2));

        let expected = hash_palette(1, in_sum, in_xor, out_sum, out_xor);
        assert_eq!(hash_vertex(&graph, &graph_t, 1, &colors), expected);
    }

    #[test]
    fn test_hash_vertex_ignores_neighbor_order() {
        let colors: Vec<Color> = vec![0, 1, 2, 3];

        let ordered = Graph {
            num_vertices: 4,
            row_offsets: vec![0, 2, 4, 4, 4],
            col_indices: vec![1, 2, 3, 0],
        };
        let reordered = Graph {
            num_vertices: 4,
            row_offsets: vec![0, 2, 4, 4, 4],
            col_indices: vec![2, 1, 0, 3],
        };
        let empty_transpose = Graph {
            num_vertices: 4,
            row_offsets: vec![0, 0, 0, 0, 0],
            col_indices: vec![],
        };

        assert_eq!(
            hash_vertex(&ordered, &empty_transpose, 0, &colors),
            hash_vertex(&reordered, &empty_transpose, 0, &colors)
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
