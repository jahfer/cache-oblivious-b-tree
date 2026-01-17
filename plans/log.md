# Atomic Ordering Relaxation Log

## Step 2: Relax write-path stores to Release (Completed 2026-01-16)

### Changes Made

**In [btree_map.rs](../src/cache_oblivious/btree_map.rs):**

- Lines ~138, ~142: Changed `version.store()` and `marker_state.store()` from `SeqCst` → `Release` (overwrite existing key path)
- Lines ~170, ~174: Changed `version.store()` and `marker_state.store()` from `SeqCst` → `Release` (insert into empty cell path, inner branch)
- Lines ~217, ~221: Changed `version.store()` and `marker_state.store()` from `SeqCst` → `Release` (insert into empty cell path, outer branch)

All 6 write-path stores now use Release ordering instead of SeqCst.

### Tests Added

Added 5 new tests in `btree_map.rs` to verify Release ordering semantics:

1. `test_write_path_release_synchronizes_with_read_acquire` - Verifies Release stores synchronize-with Acquire loads
2. `test_multiple_writes_with_release_all_observable` - Verifies multiple writes are all observable via Acquire
3. `test_marker_state_release_visible_after_insert` - Verifies marker state is Empty after insert completes
4. `test_version_release_invalidates_stale_reader` - Verifies Release store invalidates stale guards
5. `test_release_ensures_data_visible_before_version` - Verifies data written before Release is visible after Acquire

All 116 tests pass.

### Rationale

Release ordering is sufficient for write-path stores because:

- It ensures all preceding writes (UnsafeCell writes to `key`/`value`) are visible to threads that Acquire-load `version` or `marker_state`
- This is the "publication" side of the Release/Acquire pattern
- Once a reader sees the new version via Acquire, they're guaranteed to see the data written before the Release store
- We don't need total ordering with writes to other cells; we only need pairwise synchronization with readers of this cell

### Notes for Next Contributor

- Step 3 (Relax CAS operations to AcqRel/Acquire) is next
- Focus on `compare_exchange` and `compare_exchange_weak` calls on `marker_state`
- Change success ordering from `SeqCst` → `AcqRel`, failure ordering from `SeqCst` → `Acquire`
- This applies to both `btree_map.rs` and `cell.rs`

---

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
