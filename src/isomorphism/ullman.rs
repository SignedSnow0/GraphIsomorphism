//! Ullman's backtracking algorithm for graph isomorphism.
//!
//! The algorithm maintains an `n x n` candidate bit matrix `M`, where bit
//! `M[i][j]` means that vertex `i` of `graph_a` may still be mapped to vertex
//! `j` of `graph_b`.
//!
//! # Algorithm
//!
//! 1. Start from the 1-WL candidate matrix, which already excludes pairs with
//!    different final colors.
//! 2. Refine `M`: remove `M[i][j]` when an outgoing or incoming neighbor of
//!    `i` has no possible image among the corresponding neighbors of `j`.
//!    Repeat until no candidate is removed.
//! 3. Backtrack in a fixed vertex order. Assigning `i -> j` keeps only that
//!    candidate in row `i`, removes column `j` from every other row, and
//!    refines `M` again.
//! 4. A complete assignment is verified explicitly before it is returned.
//!
//! Refinement is a necessary-condition filter, so it never removes a candidate
//! that belongs to a valid isomorphism. The backtracking search is exact.
//!
//! # Parallelization
//!
//! The vertex order is fixed before the search starts. All branches at a given
//! depth are therefore independent: each candidate can fork its own candidate
//! matrix, mapping, and trail, then continue as a separate subproblem. The
//! graph data in [`Solver`] is read-only during the search, so a parallel
//! worker can share it and own only those three mutable fields. Refinement is
//! also row-parallel because every row can be tested against the immutable
//! graph bit matrices before the removals are applied.
//!
//! The serial implementation below keeps one solver instance and uses an undo
//! trail to avoid copying the matrix at every node.
//!
//! # References
//! <https://algorithm.tamnd.com/ullmann-algorithm/>

use crate::{
    graph::{BitMatrix, Graph, Vertex, Word, WordsPerRow},
    isomorphism::Mapping,
};
use std::cmp::Reverse;

const BITS_PER_WORD: usize = Word::BITS as usize;

/// State for one isomorphism query.
///
/// The graph references and adjacency bit matrices are read-only during the
/// search. Only `matrix`, `mapping`, and `trail` change as branches are tried;
/// a parallel worker can fork those three fields for one branch.
struct Solver<'a> {
    /// Original graph, used for outgoing neighbors.
    graph_a: &'a Graph,
    /// Transpose of `graph_a`, used for incoming neighbors.
    graph_a_t: &'a Graph,
    /// Number of vertices in both graphs.
    n: usize,
    /// Number of `Word`s in one candidate-matrix row.
    words_per_row: WordsPerRow,
    /// Adjacency bit matrix of `graph_b`.
    b_out_bits: BitMatrix,
    /// Adjacency bit matrix of the transpose of `graph_b`.
    b_in_bits: BitMatrix,
    /// Row-major candidate matrix, `n * words_per_row` words.
    matrix: BitMatrix,
    /// Assignment built so far.
    mapping: Vec<Option<Vertex>>,
    /// Undo log: `(word index, previous word)`.
    trail: Vec<(usize, Word)>,
}

impl<'a> Solver<'a> {
    /// Builds the solver from the initial candidate matrix.
    fn new(
        graph_a: &'a Graph,
        graph_a_t: &'a Graph,
        graph_b: &Graph,
        graph_b_t: &Graph,
        candidate_matrix: &[Word],
        words_per_row: WordsPerRow,
    ) -> Self {
        let n = graph_a.num_vertices as usize;
        debug_assert_eq!(graph_b.num_vertices as usize, n);
        debug_assert_eq!(graph_a_t.num_vertices as usize, n);
        debug_assert_eq!(graph_b_t.num_vertices as usize, n);
        debug_assert_eq!(words_per_row, n.div_ceil(BITS_PER_WORD));
        debug_assert_eq!(candidate_matrix.len(), n * words_per_row);

        let (b_out_bits, b_out_words) = graph_b.to_bitmatrix();
        let (b_in_bits, b_in_words) = graph_b_t.to_bitmatrix();
        debug_assert_eq!(b_out_words, words_per_row);
        debug_assert_eq!(b_in_words, words_per_row);

        Self {
            graph_a,
            graph_a_t,
            n,
            words_per_row,
            b_out_bits,
            b_in_bits,
            matrix: candidate_matrix.to_vec(),
            mapping: vec![None; n],
            trail: Vec::new(),
        }
    }

    /// Applies Ullman refinement until the candidate matrix stops changing.
    ///
    /// Returns `false` if a row or column runs out of candidates, which proves
    /// that this branch cannot contain an isomorphism.
    fn refine(&mut self) -> bool {
        loop {
            let mut changed = false;

            for row in 0..self.n {
                for word_index in 0..self.words_per_row {
                    let mut bits = self.matrix[row * self.words_per_row + word_index];

                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let col = word_index * BITS_PER_WORD + bit;

                        if !self.candidate_supported(row, col) {
                            self.clear_candidate(row, col);
                            changed = true;
                        }
                    }
                }

                if self.row_is_empty(row) {
                    return false;
                }
            }

            if !self.every_column_has_candidate() {
                return false;
            }

            if !changed {
                return true;
            }
        }
    }

    /// Checks the Ullman support condition for candidate `(row, col)`.
    ///
    /// For every outgoing neighbor `k` of `row`, at least one candidate of `k`
    /// must be an outgoing neighbor of `col`. The same test is applied to
    /// incoming neighbors using the transposed graphs, so directed graphs are
    /// handled correctly.
    fn candidate_supported(&self, row: usize, col: usize) -> bool {
        let words = self.words_per_row;

        let b_out_row = &self.b_out_bits[col * words..(col + 1) * words];
        for &neighbor in self.graph_a.neighbors(row as Vertex) {
            let start = neighbor as usize * words;
            if !rows_intersect(&self.matrix[start..start + words], b_out_row) {
                return false;
            }
        }

        let b_in_row = &self.b_in_bits[col * words..(col + 1) * words];
        for &neighbor in self.graph_a_t.neighbors(row as Vertex) {
            let start = neighbor as usize * words;
            if !rows_intersect(&self.matrix[start..start + words], b_in_row) {
                return false;
            }
        }

        true
    }

    /// Assigns `row -> col` and removes incompatible candidates.
    ///
    /// The rest of row `row` is cleared, and column `col` is cleared in every
    /// other row. The previous matrix words are recorded on the undo trail.
    fn assign(&mut self, row: usize, col: usize) {
        debug_assert!(self.mapping[row].is_none());
        debug_assert!(
            self.matrix[row * self.words_per_row + col / BITS_PER_WORD] & bit_mask(col) != 0
        );

        let words = self.words_per_row;
        let col_word = col / BITS_PER_WORD;
        let col_mask = bit_mask(col);

        // Only `col` remains possible for this row.
        for word in 0..words {
            let index = row * words + word;
            let new_value = if word == col_word { col_mask } else { 0 };
            if self.matrix[index] != new_value {
                self.trail.push((index, self.matrix[index]));
                self.matrix[index] = new_value;
            }
        }

        // No other row may use `col`.
        for other_row in 0..self.n {
            if other_row == row {
                continue;
            }

            let index = other_row * words + col_word;
            if self.matrix[index] & col_mask != 0 {
                self.trail.push((index, self.matrix[index]));
                self.matrix[index] &= !col_mask;
            }
        }

        self.mapping[row] = Some(col as Vertex);
    }

    /// Restores the matrix to `checkpoint` and clears the mapping for `row`.
    fn rollback(&mut self, checkpoint: usize, row: usize) {
        while self.trail.len() > checkpoint {
            let (index, old_value) = self.trail.pop().expect("trail length was checked");
            self.matrix[index] = old_value;
        }

        self.mapping[row] = None;
    }

    /// Clears one candidate bit and records the change for rollback.
    fn clear_candidate(&mut self, row: usize, col: usize) {
        let word = col / BITS_PER_WORD;
        let mask = bit_mask(col);
        let index = row * self.words_per_row + word;

        if self.matrix[index] & mask != 0 {
            self.trail.push((index, self.matrix[index]));
            self.matrix[index] &= !mask;
        }
    }

    /// Collects the currently possible columns for `row`.
    fn row_candidates(&self, row: usize) -> Vec<Vertex> {
        let mut candidates = Vec::new();

        for word_index in 0..self.words_per_row {
            let mut bits = self.matrix[row * self.words_per_row + word_index];

            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                candidates.push((word_index * BITS_PER_WORD + bit) as Vertex);
            }
        }

        candidates
    }

    /// Returns whether a row has no candidates left.
    fn row_is_empty(&self, row: usize) -> bool {
        let start = row * self.words_per_row;
        self.matrix[start..start + self.words_per_row]
            .iter()
            .all(|&word| word == 0)
    }

    /// Returns whether every column still has at least one candidate.
    ///
    /// With `n` rows and `n` columns, a column with no candidates cannot be
    /// part of a perfect matching, so the branch can be rejected.
    fn every_column_has_candidate(&self) -> bool {
        let mut occupied: Vec<Word> = vec![0; self.words_per_row];

        for row in 0..self.n {
            let start = row * self.words_per_row;
            for (word, occupied_word) in occupied.iter_mut().enumerate() {
                *occupied_word |= self.matrix[start + word];
            }
        }

        for col in 0..self.n {
            if occupied[col / BITS_PER_WORD] & bit_mask(col) == 0 {
                return false;
            }
        }

        true
    }

    /// Counts the candidates currently available for `row`.
    fn candidate_count(&self, row: usize) -> usize {
        let start = row * self.words_per_row;
        self.matrix[start..start + self.words_per_row]
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Chooses a fixed vertex order: fewest candidates first, then highest
    /// total degree. Ties retain vertex-number order because `sort_by_key` is
    /// stable.
    fn compute_order(&self) -> Vec<Vertex> {
        let mut order: Vec<Vertex> = (0..self.n as Vertex).collect();

        order.sort_by_key(|&vertex| {
            let row = vertex as usize;
            let candidate_count = self.candidate_count(row);
            let degree = self.graph_a.out_degree(vertex) + self.graph_a_t.out_degree(vertex);
            (candidate_count, Reverse(degree))
        });

        order
    }

    /// Depth-first Ullman search with a fixed row order.
    fn search(&mut self, order: &[Vertex], depth: usize) -> bool {
        if depth == self.n {
            return self.verify_mapping();
        }

        let row = order[depth] as usize;
        let checkpoint = self.trail.len();
        let candidates = self.row_candidates(row);

        for candidate in candidates {
            let col = candidate as usize;
            self.assign(row, col);

            if self.refine() && self.search(order, depth + 1) {
                return true;
            }

            self.rollback(checkpoint, row);
        }

        false
    }

    /// Verifies that the current mapping is a permutation that preserves both
    /// edge directions. Refinement should already guarantee this, but the
    /// explicit check keeps the decision procedure independent of refinement
    /// details.
    fn verify_mapping(&self) -> bool {
        let mut seen = vec![false; self.n];

        for &target in &self.mapping {
            let Some(target) = target else {
                return false;
            };
            let target = target as usize;

            if target >= self.n || seen[target] {
                return false;
            }
            seen[target] = true;
        }

        for (u, &target) in self.mapping.iter().enumerate() {
            let v = target.expect("mapping was checked above") as usize;

            for &neighbor in self.graph_a.neighbors(u as Vertex) {
                let mapped_neighbor =
                    self.mapping[neighbor as usize].expect("mapping was checked above") as usize;
                if !bit_get(&self.b_out_bits, self.words_per_row, v, mapped_neighbor) {
                    return false;
                }
            }

            for &neighbor in self.graph_a_t.neighbors(u as Vertex) {
                let mapped_neighbor =
                    self.mapping[neighbor as usize].expect("mapping was checked above") as usize;
                if !bit_get(&self.b_in_bits, self.words_per_row, v, mapped_neighbor) {
                    return false;
                }
            }
        }

        true
    }
}

/// Finds an isomorphism from `graph_a` to `graph_b`, if one exists.
///
/// `candidate_matrix` is the row-major bit matrix returned by
/// [`super::weisfeiler_lehman::build_candidate_matrix`]. Rows correspond to
/// vertices of `graph_a`, columns to vertices of `graph_b`, and bit `(u, v)`
/// is set when the two vertices have the same final 1-WL color.
///
/// On success, returns a vector `mapping` of length `graph_a.num_vertices`
/// such that `mapping[u]` is the vertex of `graph_b` matched with `u`.
pub(super) fn find_isomorphism(
    graph_a: &Graph,
    graph_b: &Graph,
    graph_a_t: &Graph,
    graph_b_t: &Graph,
    candidate_matrix: &[Word],
    words_per_row: WordsPerRow,
) -> Option<Mapping> {
    if graph_a.num_vertices == 0 {
        return Some(Mapping(Vec::new()));
    }

    let mut solver = Solver::new(
        graph_a,
        graph_a_t,
        graph_b,
        graph_b_t,
        candidate_matrix,
        words_per_row,
    );

    // Propagate the initial degree/color restrictions before branching.
    if !solver.refine() {
        return None;
    }

    // The order is computed once and reused by every branch. This fixed order
    // is what makes the search straightforward to split across threads or GPU
    // work items.
    let order = solver.compute_order();

    if solver.search(&order, 0) {
        Some(Mapping(
            solver
                .mapping
                .into_iter()
                .map(|target| target.expect("search succeeded with a complete mapping"))
                .collect(),
        ))
    } else {
        None
    }
}

/// Returns whether two candidate rows share at least one set bit.
fn rows_intersect(a: &[Word], b: &[Word]) -> bool {
    a.iter().zip(b).any(|(&left, &right)| left & right != 0)
}

/// Returns a mask with only bit `column` set within its word.
#[inline]
fn bit_mask(column: usize) -> Word {
    1 << (column % BITS_PER_WORD)
}

/// Reads bit `(row, column)` from a row-major bit matrix.
#[inline]
fn bit_get(words: &[Word], words_per_row: usize, row: usize, column: usize) -> bool {
    (words[row * words_per_row + column / BITS_PER_WORD] & bit_mask(column)) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isomorphism::weisfeiler_lehman::{Color, build_candidate_matrix};

    /// Runs the solver with a candidate matrix built from the given colors.
    fn run(
        graph_a: &Graph,
        graph_b: &Graph,
        colors_a: &[Color],
        colors_b: &[Color],
    ) -> Option<Mapping> {
        let graph_a_t = graph_a.transpose();
        let graph_b_t = graph_b.transpose();
        let (candidate_matrix, words_per_row) = build_candidate_matrix(colors_a, colors_b);

        find_isomorphism(
            graph_a,
            graph_b,
            &graph_a_t,
            &graph_b_t,
            &candidate_matrix,
            words_per_row,
        )
    }

    #[test]
    fn test_bit_get() {
        let words = [0b1010u32, 0b0101u32];
        assert!(bit_get(&words, 1, 0, 1));
        assert!(!bit_get(&words, 1, 0, 0));
        assert!(bit_get(&words, 1, 1, 0));
        assert!(!bit_get(&words, 1, 1, 1));
    }

    #[test]
    fn test_rows_intersect() {
        assert!(rows_intersect(&[0b0101], &[0b0110]));
        assert!(!rows_intersect(&[0b0101], &[0b1010]));
    }

    #[test]
    fn test_assign_and_rollback_restore_matrix() {
        let graph = Graph::new(2, &[] as &[(Vertex, Vertex)], false);
        let mut solver = Solver::new(&graph, &graph, &graph, &graph, &[0b11, 0b11], 1);
        let checkpoint = solver.trail.len();

        solver.assign(0, 1);
        assert_eq!(solver.matrix, vec![0b10, 0b01]);
        assert_eq!(solver.mapping, vec![Some(1), None]);

        solver.rollback(checkpoint, 0);
        assert_eq!(solver.matrix, vec![0b11, 0b11]);
        assert_eq!(solver.mapping, vec![None, None]);
    }

    #[test]
    fn test_find_isomorphism_accepts_permuted_path() {
        let edges_a = [(0, 1), (1, 2)];
        let edges_b = [(1, 0), (0, 2)];
        let graph_a = Graph::new(3, &edges_a, false);
        let graph_b = Graph::new(3, &edges_b, false);

        let mapping = run(&graph_a, &graph_b, &[0, 0, 0], &[0, 0, 0])
            .expect("the relabeled path is isomorphic");

        assert_eq!(mapping.0.len(), 3);

        let mut images = mapping.0.clone();
        images.sort_unstable();
        assert_eq!(images, vec![0, 1, 2], "mapping must be a bijection");

        for (u, &v) in mapping.0.iter().enumerate() {
            let mut mapped_neighbors: Vec<Vertex> = graph_a
                .neighbors(u as Vertex)
                .iter()
                .map(|&neighbor| mapping.0[neighbor as usize])
                .collect();
            mapped_neighbors.sort_unstable();

            let mut actual_neighbors = graph_b.neighbors(v).to_vec();
            actual_neighbors.sort_unstable();

            assert_eq!(mapped_neighbors, actual_neighbors);
        }
    }

    #[test]
    fn test_find_isomorphism_rejects_regular_graphs() {
        let cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
        let triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let graph_a = Graph::new(6, &cycle, false);
        let graph_b = Graph::new(6, &triangles, false);

        assert!(run(&graph_a, &graph_b, &[0; 6], &[0; 6]).is_none());
    }
}
