//! GPU kernels for Ullman's algorithm using a single search state.
//!
//! The host owns the DFS stack. The GPU owns the current candidate matrix and
//! an undo trail. This avoids the branch-expansion machinery: instead of
//! cloning a matrix for every child branch, the host applies an assignment,
//! refines on the GPU, and rolls the matrix back to a checkpoint.
//!
//! # Buffer layout
//!
//! The candidate matrix is row-major with `n * words_per_row` `u32` words.
//! Bit `(row, col)` is `matrix[row * words_per_row + col / 32] & (1 << col % 32)`.
//!
//! The trail is a pair of parallel buffers:
//!
//! - `trail_index[slot]`: matrix word index changed by the operation,
//! - `trail_old[slot]`: value of that word before the change.
//!
//! `trail_len[0]` is an atomic counter holding the current trail length.
//! `overflow[0]` is an atomic counter incremented whenever the trail is too
//! small. The host must check it after every operation that writes trail
//! entries.
//!
//! `invalid[0]` is an atomic counter used by the validity and verification
//! kernels. Reset it to zero before launching those kernels; a nonzero value
//! means the branch was rejected.

use cubecl::prelude::*;

/// Performs one Ullman refinement pass on the single candidate matrix.
///
/// One work item handles one matrix word. It reads the previous matrix and
/// writes the cleaned word into `next`. Because all support checks read the
/// previous matrix, removals within one pass are not visible to other work
/// items. That is conservative but safe; the host repeats this kernel until
/// the trail length stops increasing.
///
/// Every word that changes is appended to the undo trail.
///
/// Launch with `n * words_per_row` work items.
#[cube(launch)]
pub(super) fn refine_once_kernel(
    matrix: &[u32],
    next: &mut [u32],
    trail_index: &mut [u32],
    trail_old: &mut [u32],
    trail_len: &[Atomic<u32>],
    overflow: &[Atomic<u32>],
    a_out_offsets: &[u32],
    a_out_cols: &[u32],
    a_in_offsets: &[u32],
    a_in_cols: &[u32],
    b_out_bits: &[u32],
    b_in_bits: &[u32],
    n: usize,
    words_per_row: usize,
    trail_capacity: usize,
) {
    let pos = ABSOLUTE_POS;
    let matrix_len = n * words_per_row;

    if pos >= matrix_len {
        terminate!();
    }

    let row = pos / words_per_row;
    let word_index = pos % words_per_row;

    let old_word = matrix[pos];
    let mut word = old_word;

    for bit in 0..32u32 {
        let mask = 1u32 << bit;

        if word & mask != 0 {
            let col = word_index * 32 + bit as usize;

            if col < n
                && !candidate_supported(
                    row,
                    col,
                    matrix,
                    a_out_offsets,
                    a_out_cols,
                    a_in_offsets,
                    a_in_cols,
                    b_out_bits,
                    b_in_bits,
                    words_per_row,
                )
            {
                word &= !mask;
            }
        }
    }

    next[pos] = word;

    if word != old_word {
        append_trail(
            pos as u32,
            old_word,
            trail_index,
            trail_old,
            trail_len,
            overflow,
            trail_capacity,
        );
    }
}

/// Assigns `row -> col` and records every changed matrix word on the trail.
///
/// Row `row` is reduced to only `col`, and column `col` is cleared from every
/// other row. The host is responsible for setting/unsetting its own mapping
/// entry and for passing that mapping to [`verify_kernel`].
///
/// Launch with `n * words_per_row` work items.
#[cube(launch)]
pub(super) fn assign_kernel(
    matrix: &mut [u32],
    trail_index: &mut [u32],
    trail_old: &mut [u32],
    trail_len: &[Atomic<u32>],
    overflow: &[Atomic<u32>],
    row: usize,
    col: usize,
    n: usize,
    words_per_row: usize,
    trail_capacity: usize,
) {
    let pos = ABSOLUTE_POS;

    if pos >= n * words_per_row {
        terminate!();
    }

    let current_row = pos / words_per_row;
    let word_index = pos % words_per_row;
    let col_word = col / 32;
    let col_mask = bit_mask(col);

    let old_word = matrix[pos];
    let mut new_word = old_word;

    if current_row == row {
        if word_index == col_word {
            new_word = col_mask;
        } else {
            new_word = 0u32;
        }
    } else if word_index == col_word {
        new_word &= !col_mask;
    }

    if new_word != old_word {
        matrix[pos] = new_word;
        append_trail(
            pos as u32,
            old_word,
            trail_index,
            trail_old,
            trail_len,
            overflow,
            trail_capacity,
        );
    }
}

/// Restores matrix words from `checkpoint` to `old_len`.
///
/// One work item owns one matrix word and scans that word's trail entries in
/// reverse order. Writing every matching entry back in reverse makes the last
/// write the oldest entry, which is the value at `checkpoint`. This preserves
/// LIFO rollback without racing work items against each other.
///
/// The atomic `trail_len` is then reset to `checkpoint`.
///
/// Launch with `matrix_len + 1` work items so the reset always runs.
#[cube(launch)]
pub(super) fn rollback_kernel(
    matrix: &mut [u32],
    trail_index: &[u32],
    trail_old: &[u32],
    trail_len: &[Atomic<u32>],
    checkpoint: u32,
    old_len: u32,
    matrix_len: usize,
) {
    let pos = ABSOLUTE_POS;

    if old_len <= checkpoint {
        if pos == 0 {
            trail_len[0].store(checkpoint);
        }
        terminate!();
    }

    if pos < matrix_len {
        let count = (old_len - checkpoint) as usize;

        // Walk newest -> oldest. The final write for this word is its oldest
        // old value, i.e. the value it had at the checkpoint.
        for offset in 0..count {
            let slot = old_len - 1 - offset as u32;

            if trail_index[slot as usize] as usize == pos {
                matrix[pos] = trail_old[slot as usize];
            }
        }
    } else if pos == matrix_len {
        trail_len[0].store(checkpoint);
    }
}

/// Collects the currently possible columns for one row.
///
/// Results are appended to `candidates` with an atomic counter. The host
/// resets `count` and `overflow` to zero, launches `words_per_row` work items,
/// then reads `count` and the first `count` candidate entries.
#[cube(launch)]
pub(super) fn row_candidates_kernel(
    matrix: &[u32],
    candidates: &mut [u32],
    count: &[Atomic<u32>],
    overflow: &[Atomic<u32>],
    row: usize,
    n: usize,
    words_per_row: usize,
    candidate_capacity: usize,
) {
    let word_index = ABSOLUTE_POS;

    if word_index >= words_per_row {
        terminate!();
    }

    let word = matrix[row * words_per_row + word_index];

    for bit in 0..32u32 {
        let mask = 1u32 << bit;

        if word & mask != 0 {
            let col = word_index * 32 + bit as usize;

            if col < n {
                let slot = count[0].fetch_add(1u32) as usize;

                if slot < candidate_capacity {
                    candidates[slot] = col as u32;
                } else {
                    overflow[0].fetch_add(1u32);
                }
            }
        }
    }
}

/// Checks that every row and every column still has at least one candidate.
///
/// Each detected empty row/column increments `invalid[0]`. The host should
/// reset `invalid` to zero before launching this kernel.
///
/// Launch with `n` work items.
#[cube(launch)]
pub(super) fn validity_kernel(
    matrix: &[u32],
    invalid: &[Atomic<u32>],
    n: usize,
    words_per_row: usize,
) {
    let vertex = ABSOLUTE_POS;

    if vertex >= n {
        terminate!();
    }

    let row_base = vertex * words_per_row;
    let mut row_has_candidate = false;

    for word_index in 0..words_per_row {
        if matrix[row_base + word_index] != 0u32 {
            row_has_candidate = true;
        }
    }

    if !row_has_candidate {
        invalid[0].fetch_add(1u32);
    }

    let col_word = vertex / 32;
    let col_mask = bit_mask(vertex);
    let mut col_has_candidate = false;

    for row in 0..n {
        if matrix[row * words_per_row + col_word] & col_mask != 0u32 {
            col_has_candidate = true;
        }
    }

    if !col_has_candidate {
        invalid[0].fetch_add(1u32);
    }
}

/// Verifies a complete mapping from graph A to graph B.
///
/// `mapping[row]` is the image of `row`; unused entries should be set to a
/// sentinel outside `0..n`. The kernel checks:
///
/// - every mapping entry is in range,
/// - no two rows map to the same target,
/// - every outgoing arc of A maps to an outgoing arc of B,
/// - every incoming arc of A maps to an incoming arc of B.
///
/// Each detected problem increments `invalid[0]`. The host should reset
/// `invalid` to zero before launching this kernel.
///
/// Launch with `n` work items.
#[cube(launch)]
pub(super) fn verify_kernel(
    mapping: &[u32],
    a_out_offsets: &[u32],
    a_out_cols: &[u32],
    a_in_offsets: &[u32],
    a_in_cols: &[u32],
    b_out_bits: &[u32],
    b_in_bits: &[u32],
    invalid: &[Atomic<u32>],
    n: usize,
    words_per_row: usize,
) {
    let row = ABSOLUTE_POS;

    if row >= n {
        terminate!();
    }

    let target = mapping[row] as usize;

    if target >= n {
        invalid[0].fetch_add(1u32);
        terminate!();
    }

    for other in 0..n {
        if other != row && mapping[other] as usize == target {
            invalid[0].fetch_add(1u32);
        }
    }

    let out_start = a_out_offsets[row] as usize;
    let out_end = a_out_offsets[row + 1] as usize;

    for k in out_start..out_end {
        let neighbor = a_out_cols[k] as usize;
        let mapped_neighbor = mapping[neighbor] as usize;

        if mapped_neighbor >= n || !bit_get(b_out_bits, words_per_row, target, mapped_neighbor) {
            invalid[0].fetch_add(1u32);
        }
    }

    let in_start = a_in_offsets[row] as usize;
    let in_end = a_in_offsets[row + 1] as usize;

    for k in in_start..in_end {
        let neighbor = a_in_cols[k] as usize;
        let mapped_neighbor = mapping[neighbor] as usize;

        if mapped_neighbor >= n || !bit_get(b_in_bits, words_per_row, target, mapped_neighbor) {
            invalid[0].fetch_add(1u32);
        }
    }
}

/// Returns whether candidate `(row, col)` satisfies the Ullman support test.
///
/// For every outgoing neighbor of `row`, at least one of its candidates must
/// be an outgoing neighbor of `col`. The same check is applied to incoming
/// neighbors using the transposed graph.
#[cube]
fn candidate_supported(
    row: usize,
    col: usize,
    matrix: &[u32],
    a_out_offsets: &[u32],
    a_out_cols: &[u32],
    a_in_offsets: &[u32],
    a_in_cols: &[u32],
    b_out_bits: &[u32],
    b_in_bits: &[u32],
    words_per_row: usize,
) -> bool {
    let mut supported = true;
    let b_row_base = col * words_per_row;

    let out_start = a_out_offsets[row] as usize;
    let out_end = a_out_offsets[row + 1] as usize;

    for k in out_start..out_end {
        let neighbor = a_out_cols[k] as usize;
        let neighbor_row = neighbor * words_per_row;

        if !rows_intersect(matrix, neighbor_row, b_out_bits, b_row_base, words_per_row) {
            supported = false;
        }
    }

    let in_start = a_in_offsets[row] as usize;
    let in_end = a_in_offsets[row + 1] as usize;

    for k in in_start..in_end {
        let neighbor = a_in_cols[k] as usize;
        let neighbor_row = neighbor * words_per_row;

        if !rows_intersect(matrix, neighbor_row, b_in_bits, b_row_base, words_per_row) {
            supported = false;
        }
    }

    supported
}

/// Returns whether two packed rows share at least one bit.
#[cube]
fn rows_intersect(
    a: &[u32],
    a_start: usize,
    b: &[u32],
    b_start: usize,
    words_per_row: usize,
) -> bool {
    let mut intersects = false;

    for word_index in 0..words_per_row {
        if a[a_start + word_index] & b[b_start + word_index] != 0u32 {
            intersects = true;
        }
    }

    intersects
}

/// Appends one changed matrix word to the undo trail.
#[cube]
fn append_trail(
    index: u32,
    old_word: u32,
    trail_index: &mut [u32],
    trail_old: &mut [u32],
    trail_len: &[Atomic<u32>],
    overflow: &[Atomic<u32>],
    trail_capacity: usize,
) {
    let slot = trail_len[0].fetch_add(1u32) as usize;

    if slot < trail_capacity {
        trail_index[slot] = index;
        trail_old[slot] = old_word;
    } else {
        overflow[0].fetch_add(1u32);
    }
}

/// Returns a mask with only bit `column` set.
#[cube]
fn bit_mask(column: usize) -> u32 {
    let shift = (column % 32) as u32;
    1u32 << shift
}

/// Reads one bit from a row-major packed bit matrix.
#[cube]
fn bit_get(matrix: &[u32], words_per_row: usize, row: usize, column: usize) -> bool {
    let index = row * words_per_row + column / 32;
    matrix[index] & bit_mask(column) != 0u32
}
