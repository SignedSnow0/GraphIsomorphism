mod weisfeiler_lehman;

use crate::graph::Graph;

/// Checks if the out-degrees of the vertices in both graphs are the same.
fn check_degree(graph_a: &Graph, graph_b: &Graph, graph_a_t: &Graph, graph_b_t: &Graph) -> bool {
    for vertex in 0..graph_a.num_vertices {
        if graph_a.out_degree(vertex) != graph_b.out_degree(vertex)
            || graph_a_t.out_degree(vertex) != graph_b_t.out_degree(vertex)
        {
            return false;
        }
    }

    true
}

pub fn check_ismomorphism(graph_a: &Graph, graph_b: &Graph) -> bool {
    if graph_a.num_vertices != graph_b.num_vertices || graph_a.num_edges() != graph_b.num_edges() {
        return false;
    }

    let graph_a_t = graph_a.transpose();
    let graph_b_t = graph_b.transpose();
    if !check_degree(graph_a, graph_b, &graph_a_t, &graph_b_t) {
        return false;
    }

    let (c1, c2) = match weisfeiler_lehman::refine(graph_a, graph_b, &graph_a_t, &graph_b_t) {
        Some((c1, c2)) => (c1, c2),
        None => return false,
    };

    let (candidate_matrix, words_per_row) = weisfeiler_lehman::build_candidate_matrix(&c1, &c2);

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_ismomorphism_same_graph() {
        let edges_a = vec![(0, 1), (1, 2), (2, 0)];
        let edges_b = vec![(0, 1), (1, 2), (2, 0)];
        let graph_a = Graph::new(3, &edges_a, false);
        let graph_b = Graph::new(3, &edges_b, false);

        assert!(check_ismomorphism(&graph_a, &graph_b));
    }

    #[test]
    fn test_check_non_isomorphism_different_edges() {
        let edges_a = vec![(0, 1), (1, 2), (2, 0)];
        let edges_b = vec![(0, 1), (1, 2)];
        let graph_a = Graph::new(3, &edges_a, false);
        let graph_b = Graph::new(3, &edges_b, false);

        assert!(!check_ismomorphism(&graph_a, &graph_b));
    }

    #[test]
    fn test_check_non_isomorphism_different_vertices() {
        let edges_a = vec![(0, 1), (1, 2), (2, 0)];
        let edges_b = vec![(0, 1), (1, 2), (2, 0)];
        let graph_a = Graph::new(3, &edges_a, false);
        let graph_b = Graph::new(4, &edges_b, false);

        assert!(!check_ismomorphism(&graph_a, &graph_b));
    }

    #[test]
    fn test_check_non_isomorphism_different_degrees() {
        let edges_a = vec![(0, 1), (1, 2), (2, 0)];
        let edges_b = vec![(0, 1), (1, 2), (2, 3)];
        let graph_a = Graph::new(3, &edges_a, false);
        let graph_b = Graph::new(4, &edges_b, false);

        assert!(!check_ismomorphism(&graph_a, &graph_b));
    }
}
