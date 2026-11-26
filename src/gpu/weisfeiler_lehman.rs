//! GPU implementation of hash-based 1-dimensional Weisfeiler-Lehman refinement.
//!
//! This mirrors the CPU 1-WL pipeline. Both implementations replace signature
//! interning and neighbor sorting with the same hashing scheme:
//!
//! 1. Initial colors encode the `(in_degree, out_degree)` pair of each vertex.
//! 2. Each refinement round hashes:
//!    - the vertex's current color,
//!    - an order-independent aggregate of incoming neighbor colors,
//!    - an order-independent aggregate of outgoing neighbor colors.
//! 3. Two graphs use the same hash function, so equal signatures produce equal
//!    color IDs.
//!
//! # Buffer layout
//!
//! All graphs use CSR:
//!
//! - `*_offsets[v]..*_offsets[v + 1]` is the neighbor range of vertex `v`.
//! - `*_cols` are the concatenated neighbor vertex IDs.
//! - `in_*` buffers come from the transposed graph, so they describe incoming
//!   neighbors in the original graph.
//! - `out_*` buffers come from the original graph.

use cubecl::prelude::*;

/// Checks whether two color vectors contain the same multiset of colors.
///
/// TODO: Maybe swap to a parallel representation to speed this step also.
pub(super) fn same_histogram(colors_a: &[u32], colors_b: &[u32]) -> bool {
    let mut sorted_a = colors_a.to_vec();
    let mut sorted_b = colors_b.to_vec();
    sorted_a.sort_unstable();
    sorted_b.sort_unstable();
    sorted_a == sorted_b
}

/// Assigns initial colors from vertex degrees.
///
/// The CPU implementation computes the same initial color from the degree
/// pair:
///
/// - `h_in_a = mix32(in_a)`,
/// - `h_out_a = mix32(out_a)`,
/// - initial color is `hash_palette(0, h_in_a, h_in_a, h_out_a, h_out_a)`.
///
/// `sum == xor == h` because the initial neighbor multiset has one element.
///
/// # Parameters
///
/// - `a_in_offsets`: `graph_a.transpose().row_offsets`, length `n + 1`.
/// - `a_out_offsets`: `graph_a.row_offsets`, length `n + 1`.
/// - `b_in_offsets`: `graph_b.transpose().row_offsets`, length `n + 1`.
/// - `b_out_offsets`: `graph_b.row_offsets`, length `n + 1`.
/// - `colors_a`: output color buffer for graph A, length `n`.
/// - `colors_b`: output color buffer for graph B, length `n`.
///
/// # Parallelisation
///
/// The kernel is launched with `n` work items, one per vertex.
/// Each thread computes the initial color of a single vertex for graph A and B.
#[cube(launch)]
pub(super) fn init_kernel(
    a_in_offsets: &[u32],
    a_out_offsets: &[u32],
    b_in_offsets: &[u32],
    b_out_offsets: &[u32],
    colors_a: &mut [u32],
    colors_b: &mut [u32],
    n: usize,
) {
    let v = ABSOLUTE_POS;
    if v >= n {
        terminate!();
    }

    // The in/out degree is the number of entries in the corresponding CSR row.
    let in_a = a_in_offsets[v + 1] - a_in_offsets[v];
    let out_a = a_out_offsets[v + 1] - a_out_offsets[v];

    let in_b = b_in_offsets[v + 1] - b_in_offsets[v];
    let out_b = b_out_offsets[v + 1] - b_out_offsets[v];

    let h_in_a = mix32(in_a);
    let h_out_a = mix32(out_a);
    let h_in_b = mix32(in_b);
    let h_out_b = mix32(out_b);

    colors_a[v] = hash_palette(0u32, h_in_a, h_in_a, h_out_a, h_out_a);
    colors_b[v] = hash_palette(0u32, h_in_b, h_in_b, h_out_b, h_out_b);
}

/// Performs one 1-WL refinement round for both graphs.
///
/// One work item handles vertex `v` in both graph A and graph B. Computing both
/// in the same kernel guarantees that the two color vectors use the same hash
/// function and are directly comparable.
///
/// # Parameters
///
/// - `colors_a`, `colors_b`: previous-round colors, length `n`.
/// - `a_out_offsets`, `a_out_cols`: CSR of graph A.
/// - `a_in_offsets`, `a_in_cols`: CSR of transposed graph A.
/// - `b_out_offsets`, `b_out_cols`: CSR of graph B.
/// - `b_in_offsets`, `b_in_cols`: CSR of transposed graph B.
/// - `next_a`, `next_b`: output color buffers, length `n`.
///
/// # Parallelisation
///
/// The kernel is launched with `n` work items, one per vertex.
/// Since each thread only reads the previous-round colors and writes to its own
/// output color, there are no data races.
///
/// Since the convergence loop cannot be parallelized, this version should be called
/// multiple times and check the histogram between each iteration.
#[cube(launch)]
pub(super) fn refine_kernel(
    colors_a: &[u32],
    colors_b: &[u32],
    a_out_offsets: &[u32],
    a_out_cols: &[u32],
    a_in_offsets: &[u32],
    a_in_cols: &[u32],
    b_out_offsets: &[u32],
    b_out_cols: &[u32],
    b_in_offsets: &[u32],
    b_in_cols: &[u32],
    next_a: &mut [u32],
    next_b: &mut [u32],
    n: usize,
) {
    let v = ABSOLUTE_POS;
    if v >= n {
        terminate!();
    }

    next_a[v] = hash_vertex(
        colors_a,
        a_out_offsets,
        a_out_cols,
        a_in_offsets,
        a_in_cols,
        v,
    );

    next_b[v] = hash_vertex(
        colors_b,
        b_out_offsets,
        b_out_cols,
        b_in_offsets,
        b_in_cols,
        v,
    );
}

/// Builds the row-major candidate bit matrix from final 1-WL colors.
///
/// Bit `(row, col)` is set when `colors_a[row] == colors_b[col]`.
///
/// `words_per_row` must be `n.div_ceil(32)`. The output buffer must contain
/// `n * words_per_row` words and **must be zero-initialized**, because this
/// kernel only sets bits with `|=`.
///
/// One work item handles one row, so row `row` writes to
/// `out[row * words_per_row .. (row + 1) * words_per_row]`.
///
/// # Parallelisation
///
/// The kernel is launched with `n` work items, one per row.
/// Each thread only reads from the color buffers and writes
/// to its own row in the output buffer, so there are no data races.
#[cube(launch)]
pub(super) fn build_candidate_matrix_kernel(
    colors_a: &[u32],
    colors_b: &[u32],
    out: &mut [u32],
    n: usize,
    words_per_row: usize,
) {
    let row = ABSOLUTE_POS;
    if row >= n {
        terminate!();
    }

    let color = colors_a[row];
    let base = row * words_per_row;

    for col in 0..n {
        if colors_b[col] == color {
            let word = col / 32;
            let bit = col % 32;
            out[base + word] |= 1u32 << (bit as u32);
        }
    }
}

/// 32-bit MurmurHash3 finalizer.
///
/// Used for:
///
/// - hashing a degree during initialization,
/// - hashing a neighbor color during refinement,
/// - hashing neighbor colors in the CPU implementation, which mirrors this
///   kernel exactly.
///
/// u32 overflow is intentional and wraps modulo 2^32.
#[cube]
fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16u32;
    x *= 0x85ebca6bu32;
    x ^= x >> 13u32;
    x *= 0xc2b2ae35u32;
    x ^= x >> 16u32;
    x
}

/// Hashes one 1-WL signature.
///
/// The signature consists of:
///
/// - `seed`: the vertex's previous color (`0` during initialization),
/// - incoming neighbor color aggregates:
///   - `in_sum`,
///   - `in_xor`,
/// - outgoing neighbor color aggregates:
///   - `out_sum`,
///   - `out_xor`.
///
/// Using both `sum` and `xor` makes the hash order-independent while reducing
/// the chance that two different neighbor multisets collide.
///
/// The CPU implementation has an equivalent `hash_palette` function with the
/// same constants and arithmetic.
#[cube]
fn hash_palette(seed: u32, in_sum: u32, in_xor: u32, out_sum: u32, out_xor: u32) -> u32 {
    let mut h = mix32(seed ^ 0x9e3779b9u32);

    h = mix32(h * 0x85ebca6bu32 + in_sum);
    h = mix32(h * 0xc2b2ae35u32 + in_xor);
    h = mix32(h * 0x85ebca6bu32 + out_sum);
    h = mix32(h * 0xc2b2ae35u32 + out_xor);

    h
}

/// Computes the hashed 1-WL signature of one vertex.
///
/// Neighbor colors are aggregated with `sum` and `xor`, so the result does not
/// depend on the order in which neighbors appear in the CSR list. The CPU
/// implementation uses the same aggregation over its adjacency slices.
///
/// # Parameters
///
/// - `colors`: previous-round colors.
/// - `out_offsets`: CSR row offsets of the original graph.
/// - `out_cols`: CSR column indices of the original graph.
/// - `in_offsets`: CSR row offsets of the transposed graph.
/// - `in_cols`: CSR column indices of the transposed graph.
/// - `vertex`: vertex whose signature is computed.
#[cube]
fn hash_vertex(
    colors: &[u32],
    out_offsets: &[u32],
    out_cols: &[u32],
    in_offsets: &[u32],
    in_cols: &[u32],
    vertex: usize,
) -> u32 {
    let mut in_sum = 0u32;
    let mut in_xor = 0u32;

    let start = in_offsets[vertex] as usize;
    let end = in_offsets[vertex + 1] as usize;

    for k in start..end {
        let h = mix32(colors[in_cols[k] as usize]);
        in_sum += h;
        in_xor ^= h;
    }

    let mut out_sum = 0u32;
    let mut out_xor = 0u32;

    let start = out_offsets[vertex] as usize;
    let end = out_offsets[vertex + 1] as usize;

    for k in start..end {
        let h = mix32(colors[out_cols[k] as usize]);
        out_sum += h;
        out_xor ^= h;
    }

    hash_palette(colors[vertex], in_sum, in_xor, out_sum, out_xor)
}
