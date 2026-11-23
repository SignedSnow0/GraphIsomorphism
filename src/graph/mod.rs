/// A simple graph representation using Compressed Sparse Row (CSR) format.
///
/// This representation is efficient for storing sparse graphs and also stores data in a linear format,
/// which is a more cache friendly way and better suited for GPUs.
pub struct Graph {
    /// The number of vertices in the graph.
    pub num_vertices: u32,
    /// The row offsets indicating the start of each vertex's adjacency list in the `col_indices` vector.
    pub row_offsets: Vec<u32>,
    /// The column indices representing the adjacency lists of the vertices.
    pub col_indices: Vec<u32>,
}

impl Graph {
    /// Creates a new graph from the given number of vertices and edges.
    ///
    /// # Arguments
    ///
    /// * `num_vertices` - The number of vertices in the graph.
    /// * `edges` - A slice of tuples representing the edges in the graph.
    /// * `is_directed` - A boolean indicating whether the graph is directed or undirected.
    pub fn new(num_vertices: u32, edges: &[(u32, u32)], is_directed: bool) -> Self {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); num_vertices as usize];

        for &(u, v) in edges {
            if u != v {
                adj[u as usize].push(v);
                if !is_directed {
                    adj[v as usize].push(u);
                }
            }
        }

        let mut row_offsets = Vec::with_capacity(num_vertices as usize + 1);
        let mut col_indices = Vec::with_capacity(if is_directed {
            edges.len()
        } else {
            edges.len() * 2
        });

        row_offsets.push(0);
        for neighbors in &mut adj {
            neighbors.sort_unstable();
            neighbors.dedup();
            col_indices.extend_from_slice(neighbors);
            row_offsets.push(col_indices.len() as u32);
        }

        Self {
            num_vertices,
            row_offsets,
            col_indices,
        }
    }

    /// Creates a new directed graph representing the transpose of the current graph.
    ///
    /// A transpose of a graph is a new graph where every edge (u, v) -> (v, u)
    pub fn transpose(&self) -> Self {
        let mut transposed_edges = Vec::new();

        for u in 0..self.num_vertices {
            let start = self.row_offsets[u as usize];
            let end = self.row_offsets[u as usize + 1];

            for &v in &self.col_indices[start as usize..end as usize] {
                transposed_edges.push((v, u));
            }
        }

        Graph::new(self.num_vertices, &transposed_edges, true)
    }

    /// Returns the out-degree of the given vertex.
    ///
    /// The out-degree of a vertex is the number of edges leaving that vertex.
    pub fn out_degree(&self, vertex: u32) -> u32 {
        self.row_offsets[vertex as usize + 1] - self.row_offsets[vertex as usize]
    }

    pub fn num_edges(&self) -> u32 {
        self.col_indices.len() as u32
    }

    pub fn neighbors(&self, vertex: u32) -> &[u32] {
        let start = self.row_offsets[vertex as usize] as usize;
        let end = self.row_offsets[vertex as usize + 1] as usize;
        &self.col_indices[start..end]
    }

    pub fn to_bitmatrix(&self) -> (Vec<u32>, usize) {
        let n = self.num_vertices as usize;
        let words_per_row = n.div_ceil(32);
        let mut matrix = vec![0u32; n * words_per_row];

        for u in 0..self.num_vertices {
            let row_offset = (u as usize) * words_per_row;
            for &v in self.neighbors(u) {
                matrix[row_offset + (v as usize / 32)] |= 1u32 << (v % 32);
            }
        }

        (matrix, words_per_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = Graph::new(3, &edges, false);
        assert_eq!(graph.out_degree(0), 2);
        assert_eq!(graph.out_degree(1), 2);
        assert_eq!(graph.out_degree(2), 2);
    }

    #[test]
    fn test_transpose() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = Graph::new(3, &edges, false);
        let transposed = graph.transpose();
        assert_eq!(transposed.out_degree(0), 2);
        assert_eq!(transposed.out_degree(1), 2);
        assert_eq!(transposed.out_degree(2), 2);
    }
}
