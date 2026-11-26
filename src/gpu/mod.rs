//! Host-side caller code for the GPU 1-WL and Ullman kernels.
//!
//! This module owns the CubeCL client, uploads graph data, launches the
//! kernels, and controls the Ullman DFS. The structure mirrors the CPU
//! `isomorphism::ullman::Solver`: the host keeps the row order, mapping, and
//! search stack, while the GPU owns the candidate matrix and undo trail.

mod ullman;
mod weisfeiler_lehman;

use std::cmp::Reverse;

use crate::{
    gpu::{
        ullman::{
            assign_kernel, refine_once_kernel, rollback_kernel, row_candidates_kernel,
            validity_kernel, verify_kernel,
        },
        weisfeiler_lehman::same_histogram,
    },
    graph::{Graph, Vertex},
    isomorphism::{Mapping, check_degree},
};
use cubecl::{prelude::*, server::Handle};

use weisfeiler_lehman::{build_candidate_matrix_kernel, init_kernel, refine_kernel};

/// Color type used by both the CPU and GPU implementations.
pub type Color = u32;

const CUBE_DIM: u32 = 256;
const UNMAPPED: u32 = u32::MAX;

/// GPU friendly CSR buffers for one graph.
///
/// `out_*` comes from the original graph. `in_*` comes from its transpose.
struct GpuGraph {
    out_offsets: Handle,
    out_offsets_len: usize,
    out_cols: Handle,
    out_cols_len: usize,
    in_offsets: Handle,
    in_offsets_len: usize,
    in_cols: Handle,
    in_cols_len: usize,
}

/// Read-only graph buffers used by the GPU Ullman kernels.
struct UllmanBuffers {
    a_out_offsets: Handle,
    a_out_offsets_len: usize,
    a_out_cols: Handle,
    a_out_cols_len: usize,
    a_in_offsets: Handle,
    a_in_offsets_len: usize,
    a_in_cols: Handle,
    a_in_cols_len: usize,
    b_out_bits: Handle,
    b_out_bits_len: usize,
    b_in_bits: Handle,
    b_in_bits_len: usize,
}

/// Finds an isomorphism with GPU 1-WL and GPU Ullman search.
///
/// Returns `mapping[u] == v` when vertex `u` of `graph_a` maps to vertex `v`
/// of `graph_b`. Cheap filters should be applied by the caller so this
/// function can focus on the GPU pipeline.
pub fn find_isomorphism(graph_a: &Graph, graph_b: &Graph) -> Option<Mapping> {
    assert_eq!(
        graph_a.num_vertices, graph_b.num_vertices,
        "1-WL refinement requires graphs with the same vertex count"
    );

    if graph_a.num_vertices == 0 {
        return Some(Mapping::new(Vec::new()));
    }

    if graph_a.num_vertices != graph_b.num_vertices || graph_a.num_edges() != graph_b.num_edges() {
        return None;
    }

    let graph_a_t = graph_a.transpose();
    let graph_b_t = graph_b.transpose();

    if !check_degree(graph_a, graph_b, &graph_a_t, &graph_b_t) {
        return None;
    }

    let client = cubecl::Device::default().client();
    let mut solver = Solver::new(&client, graph_a, graph_b, &graph_a_t, &graph_b_t)?;

    // Propagate the initial degree/color restrictions before branching.
    if !solver.refine() {
        return None;
    }

    let order = solver.compute_order();

    if solver.search(&order, 0) {
        Some(Mapping::new(
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

/// Refines colors using an existing client.
fn refine_colors(
    client: &Client,
    graph_a: &Graph,
    graph_b: &Graph,
    graph_a_t: &Graph,
    graph_b_t: &Graph,
) -> Option<(Vec<Color>, Vec<Color>)> {
    assert_eq!(
        graph_a.num_vertices, graph_b.num_vertices,
        "1-WL refinement requires graphs with the same vertex count"
    );

    let n = graph_a.num_vertices as usize;

    if n == 0 {
        return Some((Vec::new(), Vec::new()));
    }

    let buffers_a = upload_graph(client, graph_a, graph_a_t);
    let buffers_b = upload_graph(client, graph_b, graph_b_t);

    let count = launch_count(n);
    let dim = launch_dim();

    let mut current_a = create_u32_buffer(client, &vec![0u32; n]);
    let mut current_b = create_u32_buffer(client, &vec![0u32; n]);
    let mut next_a = create_u32_buffer(client, &vec![0u32; n]);
    let mut next_b = create_u32_buffer(client, &vec![0u32; n]);

    init_kernel::launch(
        client,
        count.clone(),
        dim,
        buffer_arg(&buffers_a.in_offsets, buffers_a.in_offsets_len),
        buffer_arg(&buffers_a.out_offsets, buffers_a.out_offsets_len),
        buffer_arg(&buffers_b.in_offsets, buffers_b.in_offsets_len),
        buffer_arg(&buffers_b.out_offsets, buffers_b.out_offsets_len),
        buffer_arg(&current_a, n),
        buffer_arg(&current_b, n),
        n,
    );

    let mut previous_a = read_u32_buffer(client, &current_a);
    let mut previous_b = read_u32_buffer(client, &current_b);

    if !same_histogram(&previous_a, &previous_b) {
        return None;
    }

    for _ in 0..n {
        refine_kernel::launch(
            client,
            count.clone(),
            dim,
            buffer_arg(&current_a, n),
            buffer_arg(&current_b, n),
            buffer_arg(&buffers_a.out_offsets, buffers_a.out_offsets_len),
            buffer_arg(&buffers_a.out_cols, buffers_a.out_cols_len),
            buffer_arg(&buffers_a.in_offsets, buffers_a.in_offsets_len),
            buffer_arg(&buffers_a.in_cols, buffers_a.in_cols_len),
            buffer_arg(&buffers_b.out_offsets, buffers_b.out_offsets_len),
            buffer_arg(&buffers_b.out_cols, buffers_b.out_cols_len),
            buffer_arg(&buffers_b.in_offsets, buffers_b.in_offsets_len),
            buffer_arg(&buffers_b.in_cols, buffers_b.in_cols_len),
            buffer_arg(&next_a, n),
            buffer_arg(&next_b, n),
            n,
        );

        std::mem::swap(&mut current_a, &mut next_a);
        std::mem::swap(&mut current_b, &mut next_b);

        let colors_a = read_u32_buffer(client, &current_a);
        let colors_b = read_u32_buffer(client, &current_b);

        if !same_histogram(&colors_a, &colors_b) {
            return None;
        }

        let stable = colors_a == previous_a && colors_b == previous_b;
        previous_a = colors_a;
        previous_b = colors_b;

        if stable {
            break;
        }
    }

    Some((previous_a, previous_b))
}

/// Builds the candidate matrix using an existing client.
fn build_candidate_matrix_handle(
    client: &Client,
    colors_a: &[Color],
    colors_b: &[Color],
) -> Handle {
    assert_eq!(
        colors_a.len(),
        colors_b.len(),
        "candidate matrix requires equal color-vector lengths"
    );

    let n = colors_a.len();

    if n == 0 {
        return create_u32_buffer(client, &[]);
    }

    let words_per_row = words_per_row(n);
    let matrix_len = n * words_per_row;

    let colors_a_handle = create_u32_buffer(client, colors_a);
    let colors_b_handle = create_u32_buffer(client, colors_b);
    let matrix_handle = create_u32_buffer(client, &vec![0u32; matrix_len]);

    build_candidate_matrix_kernel::launch(
        client,
        launch_count(n),
        launch_dim(),
        buffer_arg(&colors_a_handle, n),
        buffer_arg(&colors_b_handle, n),
        buffer_arg(&matrix_handle, matrix_len),
        n,
        words_per_row,
    );

    matrix_handle
}

/// Host-controlled Ullman solver backed by GPU kernels.
struct Solver<'a> {
    client: Client,
    graph_a: &'a Graph,
    graph_a_t: &'a Graph,
    n: usize,
    words_per_row: usize,
    trail_capacity: usize,
    matrix: Handle,
    next_matrix: Handle,
    trail_index: Handle,
    trail_old: Handle,
    trail_len: Handle,
    overflow: Handle,
    invalid: Handle,
    candidates: Handle,
    candidate_count: Handle,
    mapping: Vec<Option<Vertex>>,
    mapping_handle: Handle,
    buffers: UllmanBuffers,
}

impl<'a> Solver<'a> {
    fn new(
        client: &Client,
        graph_a: &'a Graph,
        graph_b: &Graph,
        graph_a_t: &'a Graph,
        graph_b_t: &Graph,
    ) -> Option<Self> {
        let n = graph_a.num_vertices as usize;
        let words_per_row = words_per_row(n);
        let matrix_len = n * words_per_row;

        debug_assert_eq!(graph_b.num_vertices as usize, n);
        debug_assert_eq!(graph_a_t.num_vertices as usize, n);
        debug_assert_eq!(graph_b_t.num_vertices as usize, n);

        let (colors_a, colors_b) = refine_colors(client, graph_a, graph_b, graph_a_t, graph_b_t)?;
        let matrix = build_candidate_matrix_handle(client, &colors_a, &colors_b);

        let (b_out_bits, b_out_words) = graph_b.to_bitmatrix();
        let (b_in_bits, b_in_words) = graph_b_t.to_bitmatrix();
        debug_assert_eq!(b_out_words, words_per_row);
        debug_assert_eq!(b_in_words, words_per_row);

        // Each matrix word can lose at most 32 bits, so it can be appended to
        // the trail at most 32 times along any root-to-leaf search path.
        let trail_capacity = matrix_len * 32;

        let buffers = UllmanBuffers {
            a_out_offsets: create_u32_buffer(client, &graph_a.row_offsets),
            a_out_offsets_len: graph_a.row_offsets.len(),
            a_out_cols: create_u32_buffer(client, &graph_a.col_indices),
            a_out_cols_len: graph_a.col_indices.len(),
            a_in_offsets: create_u32_buffer(client, &graph_a_t.row_offsets),
            a_in_offsets_len: graph_a_t.row_offsets.len(),
            a_in_cols: create_u32_buffer(client, &graph_a_t.col_indices),
            a_in_cols_len: graph_a_t.col_indices.len(),
            b_out_bits: create_u32_buffer(client, &b_out_bits),
            b_out_bits_len: b_out_bits.len(),
            b_in_bits: create_u32_buffer(client, &b_in_bits),
            b_in_bits_len: b_in_bits.len(),
        };

        Some(Self {
            client: client.clone(),
            graph_a,
            graph_a_t,
            n,
            words_per_row,
            trail_capacity,
            matrix,
            next_matrix: create_u32_buffer(client, &vec![0u32; matrix_len]),
            trail_index: create_u32_buffer(client, &vec![0u32; trail_capacity]),
            trail_old: create_u32_buffer(client, &vec![0u32; trail_capacity]),
            trail_len: zero_counter(client),
            overflow: zero_counter(client),
            invalid: zero_counter(client),
            candidates: create_u32_buffer(client, &vec![0u32; n]),
            candidate_count: zero_counter(client),
            mapping: vec![None; n],
            mapping_handle: create_u32_buffer(client, &vec![UNMAPPED; n]),
            buffers,
        })
    }

    fn matrix_len(&self) -> usize {
        self.n * self.words_per_row
    }

    /// Applies Ullman refinement until the matrix stops changing.
    ///
    /// Returns `false` if a row or column runs out of candidates, which proves
    /// that this branch cannot contain an isomorphism.
    fn refine(&mut self) -> bool {
        loop {
            let before = self.trail_len();
            reset_counter(&self.client, &mut self.overflow);

            refine_once_kernel::launch(
                &self.client,
                launch_count(self.matrix_len()),
                launch_dim(),
                buffer_arg(&self.matrix, self.matrix_len()),
                buffer_arg(&self.next_matrix, self.matrix_len()),
                buffer_arg(&self.trail_index, self.trail_capacity),
                buffer_arg(&self.trail_old, self.trail_capacity),
                buffer_arg(&self.trail_len, 1),
                buffer_arg(&self.overflow, 1),
                buffer_arg(&self.buffers.a_out_offsets, self.buffers.a_out_offsets_len),
                buffer_arg(&self.buffers.a_out_cols, self.buffers.a_out_cols_len),
                buffer_arg(&self.buffers.a_in_offsets, self.buffers.a_in_offsets_len),
                buffer_arg(&self.buffers.a_in_cols, self.buffers.a_in_cols_len),
                buffer_arg(&self.buffers.b_out_bits, self.buffers.b_out_bits_len),
                buffer_arg(&self.buffers.b_in_bits, self.buffers.b_in_bits_len),
                self.n,
                self.words_per_row,
                self.trail_capacity,
            );

            std::mem::swap(&mut self.matrix, &mut self.next_matrix);

            let after = self.trail_len();
            self.check_overflow();

            if after == before {
                break;
            }
        }

        self.valid()
    }

    /// Assigns `row -> col` on the GPU and records the matrix changes.
    fn assign(&mut self, row: usize, col: usize) {
        reset_counter(&self.client, &mut self.overflow);

        assign_kernel::launch(
            &self.client,
            launch_count(self.matrix_len()),
            launch_dim(),
            buffer_arg(&self.matrix, self.matrix_len()),
            buffer_arg(&self.trail_index, self.trail_capacity),
            buffer_arg(&self.trail_old, self.trail_capacity),
            buffer_arg(&self.trail_len, 1),
            buffer_arg(&self.overflow, 1),
            row,
            col,
            self.n,
            self.words_per_row,
            self.trail_capacity,
        );

        self.check_overflow();
        self.mapping[row] = Some(col as Vertex);
    }

    /// Restores the matrix to `checkpoint` and clears the mapping for `row`.
    fn rollback(&mut self, checkpoint: usize, row: usize) {
        let old_len = self.trail_len();
        assert!(old_len >= checkpoint);

        // One work item per matrix word, plus one to reset `trail_len`.
        let work_items = self.matrix_len() + 1;

        rollback_kernel::launch(
            &self.client,
            launch_count(work_items),
            launch_dim(),
            buffer_arg(&self.matrix, self.matrix_len()),
            buffer_arg(&self.trail_index, self.trail_capacity),
            buffer_arg(&self.trail_old, self.trail_capacity),
            buffer_arg(&self.trail_len, 1),
            checkpoint as u32,
            old_len as u32,
            self.matrix_len(),
        );

        self.mapping[row] = None;
    }

    /// Collects the currently possible columns for `row`.
    fn row_candidates(&mut self, row: usize) -> Vec<Vertex> {
        reset_counter(&self.client, &mut self.candidate_count);
        reset_counter(&self.client, &mut self.overflow);

        row_candidates_kernel::launch(
            &self.client,
            launch_count(self.words_per_row),
            launch_dim(),
            buffer_arg(&self.matrix, self.matrix_len()),
            buffer_arg(&self.candidates, self.n),
            buffer_arg(&self.candidate_count, 1),
            buffer_arg(&self.overflow, 1),
            row,
            self.n,
            self.words_per_row,
            self.n,
        );

        let count = read_counter(&self.client, &self.candidate_count) as usize;
        self.check_overflow();

        let mut candidates = read_u32_buffer(&self.client, &self.candidates);
        candidates.truncate(count);
        candidates
    }

    /// Checks whether every row and column still has at least one candidate.
    fn valid(&mut self) -> bool {
        reset_counter(&self.client, &mut self.invalid);

        validity_kernel::launch(
            &self.client,
            launch_count(self.n),
            launch_dim(),
            buffer_arg(&self.matrix, self.matrix_len()),
            buffer_arg(&self.invalid, 1),
            self.n,
            self.words_per_row,
        );

        read_counter(&self.client, &self.invalid) == 0
    }

    /// Chooses a fixed vertex order: fewest candidates first, then highest
    /// total degree. Ties retain vertex-number order.
    fn compute_order(&self) -> Vec<Vertex> {
        let matrix = read_u32_buffer(&self.client, &self.matrix);
        let mut order: Vec<Vertex> = (0..self.n as Vertex).collect();

        order.sort_by_key(|&vertex| {
            let row = vertex as usize;
            let start = row * self.words_per_row;
            let candidate_count: usize = matrix[start..start + self.words_per_row]
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum();
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
        let checkpoint = self.trail_len();
        let candidates = self.row_candidates(row);

        for candidate in candidates {
            self.assign(row, candidate as usize);

            if self.refine() && self.search(order, depth + 1) {
                return true;
            }

            self.rollback(checkpoint, row);
        }

        false
    }

    /// Verifies the complete mapping on the GPU.
    fn verify_mapping(&mut self) -> bool {
        let mapping: Vec<u32> = self
            .mapping
            .iter()
            .map(|target| target.map_or(UNMAPPED, |vertex| vertex))
            .collect();
        self.mapping_handle = create_u32_buffer(&self.client, &mapping);
        reset_counter(&self.client, &mut self.invalid);

        verify_kernel::launch(
            &self.client,
            launch_count(self.n),
            launch_dim(),
            buffer_arg(&self.mapping_handle, self.n),
            buffer_arg(&self.buffers.a_out_offsets, self.buffers.a_out_offsets_len),
            buffer_arg(&self.buffers.a_out_cols, self.buffers.a_out_cols_len),
            buffer_arg(&self.buffers.a_in_offsets, self.buffers.a_in_offsets_len),
            buffer_arg(&self.buffers.a_in_cols, self.buffers.a_in_cols_len),
            buffer_arg(&self.buffers.b_out_bits, self.buffers.b_out_bits_len),
            buffer_arg(&self.buffers.b_in_bits, self.buffers.b_in_bits_len),
            buffer_arg(&self.invalid, 1),
            self.n,
            self.words_per_row,
        );

        read_counter(&self.client, &self.invalid) == 0
    }

    fn trail_len(&self) -> usize {
        read_counter(&self.client, &self.trail_len) as usize
    }

    fn check_overflow(&self) {
        assert_eq!(
            read_counter(&self.client, &self.overflow),
            0,
            "GPU Ullman trail overflow; increase trail_capacity"
        );
    }
}

/// Uploads the original and transposed CSR buffers of one graph.
fn upload_graph(client: &Client, graph: &Graph, graph_t: &Graph) -> GpuGraph {
    GpuGraph {
        out_offsets: create_u32_buffer(client, &graph.row_offsets),
        out_offsets_len: graph.row_offsets.len(),
        out_cols: create_u32_buffer(client, &graph.col_indices),
        out_cols_len: graph.col_indices.len(),
        in_offsets: create_u32_buffer(client, &graph_t.row_offsets),
        in_offsets_len: graph_t.row_offsets.len(),
        in_cols: create_u32_buffer(client, &graph_t.col_indices),
        in_cols_len: graph_t.col_indices.len(),
    }
}

/// Number of `u32` words in one bit-matrix row.
fn words_per_row(n: usize) -> usize {
    n.div_ceil(u32::BITS as usize)
}

/// Creates a GPU buffer initialized with `data`.
fn create_u32_buffer(client: &Client, data: &[u32]) -> Handle {
    client.create_from_slice(u32::as_bytes(data))
}

/// Creates a zero-initialized atomic counter buffer.
fn zero_counter(client: &Client) -> Handle {
    create_u32_buffer(client, &[0u32])
}

/// Resets an atomic counter handle to zero.
fn reset_counter(client: &Client, counter: &mut Handle) {
    *counter = zero_counter(client);
}

/// Reads a GPU counter back to the host.
fn read_counter(client: &Client, counter: &Handle) -> u32 {
    read_u32_buffer(client, counter)[0]
}

/// Reads a GPU buffer back into a `Vec<u32>`.
fn read_u32_buffer(client: &Client, handle: &Handle) -> Vec<u32> {
    let bytes = client
        .read_one(handle.clone())
        .expect("GPU buffer read failed");
    u32::from_bytes(&bytes).to_vec()
}

/// Creates a `BufferArg` with the length measured in `u32` elements.
///
/// The caller must pass the element count declared by the kernel argument.
fn buffer_arg(handle: &Handle, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(handle.clone(), len) }
}

/// Number of cubes needed to cover `work_items` threads.
fn launch_count(work_items: usize) -> CubeCount {
    let cubes = work_items.div_ceil(CUBE_DIM as usize) as u32;
    CubeCount::Static(cubes, 1, 1)
}

fn launch_dim() -> CubeDim {
    CubeDim::new_1d(CUBE_DIM)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word changed several times must roll back to its checkpoint value.
    /// The old rollback kernel could race and restore an intermediate value.
    #[test]
    fn rollback_restores_checkpoint_when_word_changes_repeatedly() {
        let client = cubecl::Device::default().client();

        let matrix_len = 1usize;
        let matrix = create_u32_buffer(&client, &[0b0001]);
        let trail_index = create_u32_buffer(&client, &[0, 0, 0]);
        let trail_old = create_u32_buffer(&client, &[0b1111, 0b0111, 0b0011]);
        let trail_len = create_u32_buffer(&client, &[3]);

        rollback_kernel::launch(
            &client,
            launch_count(matrix_len + 1),
            launch_dim(),
            buffer_arg(&matrix, matrix_len),
            buffer_arg(&trail_index, 3),
            buffer_arg(&trail_old, 3),
            buffer_arg(&trail_len, 1),
            0u32,
            3u32,
            matrix_len,
        );

        assert_eq!(read_u32_buffer(&client, &matrix), vec![0b1111]);
        assert_eq!(read_counter(&client, &trail_len), 0);
    }
}
