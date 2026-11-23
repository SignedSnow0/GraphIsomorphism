use crate::graph::Graph;
use std::collections::HashMap;

pub fn refine(
    graph_a: &Graph,
    graph_b: &Graph,
    graph_a_t: &Graph,
    graph_b_t: &Graph,
) -> Option<(Vec<u32>, Vec<u32>)> {
    // (color, in_neighbors_colors, out_neighbors_colors) -> color_id
    let mut palette: HashMap<(u32, Vec<u32>, Vec<u32>), u32> = HashMap::new();
    let mut next_id = 0u32;

    // Iteration 0: Initial colors from (in_degree, out_degree)
    let n = graph_a.num_vertices;
    let mut c1: Vec<u32> = (0..n)
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

    let mut c2: Vec<u32> = (0..n)
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

    let mut prev_unique = 0usize;

    for _ in 0..n {
        if !same_histogram(&c1, &c2) {
            return None;
        }

        let mut unique = c1.clone();
        unique.sort_unstable();
        unique.dedup();

        if unique.len() == prev_unique || unique.len() == n as usize {
            break;
        }
        prev_unique = unique.len();

        palette.clear();
        next_id = 0;

        let next_c1: Vec<u32> = (0..n)
            .map(|u| {
                intern(
                    &mut palette,
                    &mut next_id,
                    vertex_signature(graph_a, &graph_a_t, u, &c1),
                )
            })
            .collect();

        let next_c2: Vec<u32> = (0..n)
            .map(|v| {
                intern(
                    &mut palette,
                    &mut next_id,
                    vertex_signature(graph_b, &graph_b_t, v, &c2),
                )
            })
            .collect();

        c1 = next_c1;
        c2 = next_c2;
    }

    if !same_histogram(&c1, &c2) {
        return None;
    }

    Some((c1, c2))
}

pub fn build_candidate_matrix(c1: &[u32], c2: &[u32]) -> (Vec<u32>, usize) {
    let n = c1.len();
    let words_per_row = n.div_ceil(32);
    let mut m = vec![0u32; n * words_per_row];

    for (u, &cu) in c1.iter().enumerate() {
        let row_offset = u * words_per_row;
        for (v, &cv) in c2.iter().enumerate() {
            if cu == cv {
                m[row_offset + (v / 32)] |= 1u32 << (v % 32);
            }
        }
    }

    (m, words_per_row)
}

fn intern(
    palette: &mut HashMap<(u32, Vec<u32>, Vec<u32>), u32>,
    next_id: &mut u32,
    signature: (u32, Vec<u32>, Vec<u32>),
) -> u32 {
    if let Some(&id) = palette.get(&signature) {
        return id;
    }

    let id = *next_id;
    *next_id += 1;
    palette.insert(signature, id);
    id
}

fn vertex_signature(g: &Graph, g_t: &Graph, u: u32, colors: &[u32]) -> (u32, Vec<u32>, Vec<u32>) {
    let mut in_colors: Vec<u32> = g_t
        .neighbors(u)
        .iter()
        .map(|&w| colors[w as usize])
        .collect();
    let mut out_colors: Vec<u32> = g.neighbors(u).iter().map(|&v| colors[v as usize]).collect();
    in_colors.sort_unstable();
    out_colors.sort_unstable();
    (colors[u as usize], in_colors, out_colors)
}

fn same_histogram(c1: &[u32], c2: &[u32]) -> bool {
    let mut s1 = c1.to_vec();
    let mut s2 = c2.to_vec();
    s1.sort_unstable();
    s2.sort_unstable();
    s1 == s2
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
        let mut palette = HashMap::new();
        let mut next_id = 0;
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
    fn test_vertex_signature() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = Graph::new(3, &edges, false);
        let graph_t = graph.transpose();
        let colors = vec![0, 1, 2];

        let sig0 = vertex_signature(&graph, &graph_t, 0, &colors);
        let sig1 = vertex_signature(&graph, &graph_t, 1, &colors);
        let sig2 = vertex_signature(&graph, &graph_t, 2, &colors);

        assert_eq!(sig0.0, 0);
        assert_eq!(sig1.0, 1);
        assert_eq!(sig2.0, 2);
    }

    #[test]
    fn test_build_candidate_matrix() {
        let c1 = vec![0, 1, 0];
        let c2 = vec![1, 0, 0];
        let (matrix, words_per_row) = build_candidate_matrix(&c1, &c2);

        assert_eq!(words_per_row, 1);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0], 0b110); // Vertex 0 in c1 (color 0) matches vertices 1 and 2 in c2
        assert_eq!(matrix[1], 0b001); // Vertex 1 in c1 (color 1) matches vertex 0 in c2
        assert_eq!(matrix[2], 0b110); // Vertex 2 in c1 (color 0) matches vertices 1 and 2 in c2
    }

    #[test]
    fn test_refine() {
        let edges_a = vec![(0, 1), (1, 2), (2, 0)];
        let edges_b = vec![(0, 1), (1, 2), (2, 0)];
        let graph_a = Graph::new(3, &edges_a, false);
        let graph_b = Graph::new(3, &edges_b, false);
        let graph_a_t = graph_a.transpose();
        let graph_b_t = graph_b.transpose();

        let result = refine(&graph_a, &graph_b, &graph_a_t, &graph_b_t);
        assert!(result.is_some());
    }
}
