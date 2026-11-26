//! Compressed sparse row (CSR) graph representation.
//!
//! The graph is a simple, unweighted graph:
//!
//! * self-loops are ignored;
//! * duplicate arcs are removed;
//! * adjacency lists are sorted, which makes neighbor lookups and edge tests
//!   deterministic.
//!
//! Both directed and undirected graphs use the same CSR layout. An undirected
//! edge `(u, v)` is stored as two arcs, `u -> v` and `v -> u`. The
//! `is_directed` flag only controls construction; it is not stored in the
//! resulting graph.

mod parser;
pub use parser::read_graph_file;

/// Vertex identifier.
pub type Vertex = u32;

/// Directed arc or undirected edge between two vertices.
pub type Edge = (Vertex, Vertex);

/// Number of incident arcs at one vertex.
pub type Degree = u32;

/// Pair of `(in_degree, out_degree)` values for one vertex.
pub type DegreePair = (Degree, Degree);

/// Number of vertices in a graph.
pub type VertexCount = u32;

/// Number of stored arcs in a graph.
pub type ArcCount = u32;

/// Index into the CSR `col_indices` array.
pub type Offset = u32;

/// One word of a packed bit matrix.
pub type Word = u32;

/// Row-major packed bit matrix.
pub type BitMatrix = Vec<Word>;

/// Number of [`Word`]s in one bit-matrix row.
pub type WordsPerRow = usize;

/// A simple graph represented in compressed sparse row (CSR) format.
///
/// CSR stores all adjacency lists in one contiguous vector, which is
/// cache-friendly and maps well to GPU memory. Each vertex owns the slice
/// `col_indices[row_offsets[u]..row_offsets[u + 1]]`.
#[derive(Debug, Clone)]
pub struct Graph {
    /// The number of vertices in the graph.
    pub num_vertices: VertexCount,
    /// Row offsets into `col_indices`, with `num_vertices + 1` entries.
    pub row_offsets: Vec<Offset>,
    /// Concatenated, sorted adjacency lists.
    pub col_indices: Vec<Vertex>,
}

impl Graph {
    /// Creates a graph from a vertex count and an edge list.
    ///
    /// If `is_directed` is `false`, every edge `(u, v)` is inserted in both
    /// directions. Self-loops are ignored and duplicate arcs are removed, so
    /// the result is always a simple graph.
    ///
    /// # Panics
    ///
    /// Panics if an edge endpoint is outside `0..num_vertices`.
    pub fn new(num_vertices: VertexCount, edges: &[Edge], is_directed: bool) -> Self {
        let mut adjacency: Vec<Vec<Vertex>> = vec![Vec::new(); num_vertices as usize];

        for &(u, v) in edges {
            assert!(
                u < num_vertices,
                "edge endpoint {u} is outside 0..{num_vertices}"
            );
            assert!(
                v < num_vertices,
                "edge endpoint {v} is outside 0..{num_vertices}"
            );

            if u != v {
                adjacency[u as usize].push(v);
                if !is_directed {
                    adjacency[v as usize].push(u);
                }
            }
        }

        let mut row_offsets: Vec<Offset> = Vec::with_capacity(num_vertices as usize + 1);
        let mut col_indices: Vec<Vertex> = Vec::with_capacity(if is_directed {
            edges.len()
        } else {
            edges.len() * 2
        });

        row_offsets.push(0);
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
            col_indices.extend_from_slice(neighbors);
            row_offsets.push(col_indices.len() as Offset);
        }

        Self {
            num_vertices,
            row_offsets,
            col_indices,
        }
    }

    /// Builds the transpose of this graph.
    ///
    /// Every arc `u -> v` becomes `v -> u`. For an undirected graph the
    /// result is an equivalent graph because both directions are already
    /// present.
    pub fn transpose(&self) -> Self {
        let mut transposed_edges: Vec<Edge> = Vec::new();

        for u in 0..self.num_vertices {
            let start = self.row_offsets[u as usize];
            let end = self.row_offsets[u as usize + 1];

            for &v in &self.col_indices[start as usize..end as usize] {
                transposed_edges.push((v, u));
            }
        }

        Graph::new(self.num_vertices, &transposed_edges, true)
    }

    /// Returns the out-degree of `vertex`.
    ///
    /// For an undirected graph this is the ordinary degree. For a directed
    /// graph it is the number of outgoing arcs only; use the transpose to
    /// obtain in-degrees.
    pub fn out_degree(&self, vertex: Vertex) -> Degree {
        self.row_offsets[vertex as usize + 1] - self.row_offsets[vertex as usize]
    }

    /// Returns the number of stored arcs.
    ///
    /// For an undirected graph every edge is stored twice, so this returns
    /// `2 * |E|`. For a directed graph it returns `|E|`.
    pub fn num_edges(&self) -> ArcCount {
        self.col_indices.len() as ArcCount
    }

    /// Returns the sorted adjacency slice of `vertex`.
    ///
    /// For a directed graph these are the outgoing neighbors. Use the
    /// transpose to obtain incoming neighbors.
    pub fn neighbors(&self, vertex: Vertex) -> &[Vertex] {
        let start = self.row_offsets[vertex as usize] as usize;
        let end = self.row_offsets[vertex as usize + 1] as usize;
        &self.col_indices[start..end]
    }

    /// Builds a row-major adjacency bit matrix.
    ///
    /// Bit `(u, v)` is set when the arc `u -> v` exists. The returned
    /// `words_per_row` value is the number of [`Word`]s per row
    /// (`num_vertices.div_ceil(Word::BITS)`).
    ///
    /// Bit matrices make edge tests O(1) and are also used by the planned GPU
    /// implementation.
    pub fn to_bitmatrix(&self) -> (BitMatrix, WordsPerRow) {
        let n = self.num_vertices as usize;
        let bits_per_word = Word::BITS as usize;
        let words_per_row = n.div_ceil(bits_per_word);
        let mut matrix: BitMatrix = vec![0; n * words_per_row];

        for u in 0..self.num_vertices {
            let row_offset = (u as usize) * words_per_row;
            for &v in self.neighbors(u) {
                matrix[row_offset + (v as usize / bits_per_word)] |= 1 << (v % Word::BITS);
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
        let edges = [(0, 1), (1, 2), (2, 0)];
        let graph = Graph::new(3, &edges, false);
        assert_eq!(graph.out_degree(0), 2);
        assert_eq!(graph.out_degree(1), 2);
        assert_eq!(graph.out_degree(2), 2);
    }

    #[test]
    fn test_transpose_undirected() {
        let edges = [(0, 1), (1, 2), (2, 0)];
        let graph = Graph::new(3, &edges, false);
        let transposed = graph.transpose();
        assert_eq!(transposed.out_degree(0), 2);
        assert_eq!(transposed.out_degree(1), 2);
        assert_eq!(transposed.out_degree(2), 2);
    }

    #[test]
    fn test_transpose_directed() {
        // Directed path 0 -> 1 -> 2.
        let graph = Graph::new(3, &[(0, 1), (1, 2)], true);
        let transposed = graph.transpose();

        assert_eq!(transposed.out_degree(0), 0);
        assert_eq!(transposed.out_degree(1), 1);
        assert_eq!(transposed.out_degree(2), 1);
        assert_eq!(transposed.neighbors(1), &[0]);
        assert_eq!(transposed.neighbors(2), &[1]);
    }

    #[test]
    fn test_self_loops_are_ignored() {
        let graph = Graph::new(2, &[(0, 0), (0, 1)], false);

        assert_eq!(graph.out_degree(0), 1);
        assert_eq!(graph.out_degree(1), 1);
        assert_eq!(graph.num_edges(), 2);
    }

    #[test]
    fn test_duplicate_edges_are_removed() {
        let graph = Graph::new(2, &[(0, 1), (0, 1), (1, 0)], false);

        assert_eq!(graph.out_degree(0), 1);
        assert_eq!(graph.out_degree(1), 1);
        assert_eq!(graph.num_edges(), 2);
    }

    #[test]
    fn test_to_bitmatrix() {
        let graph = Graph::new(3, &[(0, 1), (2, 0)], false);
        let (matrix, words_per_row) = graph.to_bitmatrix();

        assert_eq!(words_per_row, 1);
        assert_eq!(matrix[0], 0b110); // 0 is adjacent to 1 and 2
        assert_eq!(matrix[1], 0b001); // 1 is adjacent to 0
        assert_eq!(matrix[2], 0b001); // 2 is adjacent to 0
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn test_out_of_bounds_edge_panics() {
        let _ = Graph::new(2, &[(0, 2)], false);
    }
}
