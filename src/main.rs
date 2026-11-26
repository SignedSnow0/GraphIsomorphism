//! Small demonstration of the graph isomorphism implementation.

mod gpu;
mod graph;
mod isomorphism;

use graph::Graph;
use isomorphism::find_isomorphism;

fn main() {
    // Two labelings of the same triangle.
    let graph_a = Graph::new(3, &[(0, 1), (1, 2), (2, 0)], false);
    let graph_b = Graph::new(3, &[(1, 0), (0, 2), (2, 1)], false);

    let mapping = find_isomorphism(&graph_a, &graph_b);
    let gpu_mapping = gpu::find_isomorphism(&graph_a, &graph_b);
    assert_eq!(mapping, gpu_mapping);

    match mapping {
        Some(mapping) => {
            println!("Found isomorphism for two triangles: {}", mapping);
        }
        None => {
            println!("No isomorphism found for two triangles.");
        }
    }

    match gpu_mapping {
        Some(mapping) => {
            println!("Found isomorphism for two triangles (GPU): {}", mapping);
        }
        None => {
            println!("No isomorphism found for two triangles (GPU).");
        }
    }

    // The Frucht graph: 12 vertices, 3-regular, and asymmetric. Every vertex
    // gets the same 1-WL color, so the cheap filters cannot narrow the search;
    // Ullman's refinement and backtracking find the only possible mapping.
    // `frucht_b` is `frucht_a` relabeled by
    // 0->7, 1->2, 2->11, 3->0, 4->5, 5->10, 6->1, 7->6, 8->9, 9->3, 10->8, 11->4.
    let frucht_a = [
        (0, 1),
        (0, 7),
        (0, 11),
        (1, 2),
        (1, 11),
        (2, 3),
        (2, 10),
        (3, 4),
        (3, 5),
        (4, 5),
        (4, 9),
        (5, 6),
        (6, 7),
        (6, 8),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
    ];
    let frucht_b = [
        (0, 5),
        (0, 10),
        (0, 11),
        (1, 6),
        (1, 9),
        (1, 10),
        (2, 4),
        (2, 7),
        (2, 11),
        (3, 5),
        (3, 8),
        (3, 9),
        (4, 7),
        (4, 8),
        (5, 10),
        (6, 7),
        (6, 9),
        (8, 11),
    ];
    let graph_a = Graph::new(12, &frucht_a, false);
    let graph_b = Graph::new(12, &frucht_b, false);

    let mapping =
        find_isomorphism(&graph_a, &graph_b).expect("the relabeled Frucht graph is isomorphic");
    println!("Frucht graph mapping (position u -> value): {}", mapping);
    let gpu_mapping = gpu::find_isomorphism(&graph_a, &graph_b)
        .expect("the relabeled Frucht graph is isomorphic (GPU)");
    println!(
        "Frucht graph mapping (GPU) (position u -> value): {}",
        gpu_mapping
    );
    assert_eq!(mapping, gpu_mapping);

    // A 6-cycle and two disjoint triangles. Both are 2-regular with 6 edges,
    // so the cheap filters and 1-WL cannot separate them. Ullman's
    // candidate-matrix search is what rejects the pair.
    let cycle = Graph::new(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], false);
    let triangles = Graph::new(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], false);

    let mapping = find_isomorphism(&cycle, &triangles);
    let gpu_mapping = gpu::find_isomorphism(&cycle, &triangles);
    assert_eq!(mapping, gpu_mapping);
    match mapping {
        Some(mapping) => {
            println!("C6 and two triangles are isomorphic: {}", mapping);
        }
        None => {
            println!("No isomorphism found between C6 and two triangles.");
        }
    }

    match gpu_mapping {
        Some(mapping) => {
            println!("C6 and two triangles are isomorphic (GPU): {}", mapping);
        }
        None => {
            println!("No isomorphism found between C6 and two triangles (GPU).");
        }
    }
}
