# Atomic Ordering Relaxation Log

## Step 1: Relax read-path loads to Acquire (Completed 2026-01-16)

### Changes Made

**In [cell.rs](../src/cache_oblivious/cell.rs):**

- `CellGuard::from_raw()` (lines ~518-521): Changed initial `version.load()`, `load_marker_state()`, and `move_dest.load()` from `SeqCst` → `Acquire`
- `CellGuard::cache()` (line ~344): Changed `version.load()` from `SeqCst` → `Acquire`
- **Exception preserved**: The version re-read at line ~523 remains `SeqCst` as required for seqlock validation

**In [btree_map.rs](../src/cache_oblivious/btree_map.rs):**

- Line ~415: Changed destination `load_marker_state()` from `SeqCst` → `Acquire` (rebalance overwrite check)
- Lines ~442-443: Changed source cell `version.load()` and `load_marker_state()` from `SeqCst` → `Acquire` (rebalance source read)
- Line ~2748: Changed test code `version.load()` from `SeqCst` → `Acquire`

### Tests Added

Added 6 new tests in `cell.rs` to verify Acquire ordering semantics:

1. `test_from_raw_captures_version_with_acquire` - Verifies Release/Acquire synchronization
2. `test_from_raw_captures_marker_state_with_acquire` - Verifies marker state and move_dest capture
3. `test_cache_detects_version_change_with_acquire` - Verifies cache() validation with Acquire
4. `test_seqlock_detects_concurrent_modification_in_from_raw` - Verifies seqlock pattern works
5. `test_from_raw_with_empty_cell_acquire` - Verifies empty cell handling
6. `test_cache_returns_cached_data_on_subsequent_calls` - Verifies caching behavior

All 109 tests pass.

### Rationale

Acquire ordering is sufficient for read-path loads because:

- It synchronizes-with the writer's Release stores on `version` and `marker_state`
- It prevents subsequent reads (of `key`/`value` UnsafeCells) from being reordered before the version check
- On ARM, this creates a load-acquire barrier ensuring data reads are sequenced after the version is known
- We don't need total ordering with writes to other cells; we only need to synchronize with the writer of the same cell

### Notes for Next Contributor

- Step 2 (Relax write-path stores to Release) is next
- Focus on `version.store()` and `marker_state.store()` calls after UnsafeCell writes
- Lines to check: ~130, ~135, ~149, ~153, ~179, ~183 in btree_map.rs
