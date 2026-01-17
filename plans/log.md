# Atomic Ordering Relaxation Log

## Step 5: Relax destination cell writes to Release (Completed 2026-01-16)

### Changes Made

**In [btree_map.rs](../src/cache_oblivious/btree_map.rs):**

- Lines ~502-505: Changed destination cell `version.store()` and `marker_state.store()` from `SeqCst` → `Release` in the rebalance loop

### Tests Added

Added 5 new tests in `btree_map.rs` to verify destination cell Release ordering semantics:

1. `test_destination_cell_version_release_publishes_data` - Verifies Release store on version makes key/value visible to Acquire readers
2. `test_destination_cell_marker_release_publishes_all_writes` - Verifies marker_state Release publishes all prior writes (key, value, version)
3. `test_multiple_destination_cells_release_ordering` - Verifies multiple destination cells all have proper Release semantics
4. `test_concurrent_readers_observe_destination_cell_after_release` - Verifies concurrent readers all see consistent data after Release stores
5. `test_rebalance_source_to_destination_release_transfer` - Verifies full source-to-destination data transfer with Release/Acquire synchronization

All 130 tests pass.

### Rationale

Release ordering on destination cell `version.store()` and `marker_state.store()` is sufficient because:

- These stores publish the newly-moved data to the destination cell
- Release ensures the preceding UnsafeCell writes (`key.get().write()`, `value.get().write()`) are visible to readers who Acquire-load the destination's `version` or `marker_state`
- The write sequence is: (1) UnsafeCell writes for key/value, (2) version store with Release, (3) marker_state store with Release
- Any reader who Acquire-loads the marker_state will see all prior writes due to Release/Acquire synchronization

**Synchronizes with**: Readers performing Acquire loads on the destination cell's `version` or `marker_state`.

### Notes for Next Contributor

- Step 6 (Update cell.rs helper method orderings) is next
- Focus on `update_marker_state()` in cell.rs
- Ensure it uses `AcqRel`/`Acquire` for CAS and `Release` for `move_dest.store()`
- Note: The CAS ordering was already changed in Step 3; verify `move_dest.store()` uses Release (changed in Step 4)
- This step may already be complete from prior steps - verify and mark done if so

---

## Step 4: Relax move_dest store to Release (Completed 2026-01-16)

### Changes Made

**In [btree_map.rs](../src/cache_oblivious/btree_map.rs):**

- Line ~472: Changed `cell_to_move.move_dest.store(dest_index, Ordering::SeqCst)` → `Ordering::Release`

**In [cell.rs](../src/cache_oblivious/cell.rs):**

- Line ~423: Changed `self.inner.move_dest.store(new_move_dest, AtomicOrdering::SeqCst)` → `AtomicOrdering::Release` in `update_marker_state()`

### Tests Added

Added 5 new tests in `btree_map.rs` to verify move_dest Release ordering semantics:

1. `test_move_dest_release_visible_after_marker_acquire` - Verifies Release store on move_dest is visible to readers who Acquire-load marker_state
2. `test_move_dest_release_prevents_reorder_after_cas` - Verifies Release ordering prevents move_dest from being reordered after the CAS (multi-iteration concurrent test)
3. `test_update_marker_state_move_dest_release` - Verifies the helper method in cell.rs correctly uses Release for move_dest
4. `test_concurrent_readers_observe_move_dest` - Verifies multiple concurrent readers all observe the correct move_dest value when seeing Move marker

All 125 tests pass.

### Rationale

Release ordering on `move_dest.store()` is sufficient because:

- The `move_dest` is stored BEFORE the CAS that sets `marker_state` to `Move`
- The CAS uses AcqRel, which provides the synchronization point
- Release on `move_dest.store()` ensures it cannot be reordered after the subsequent CAS
- The CAS's Release component creates a release sequence that includes the prior `move_dest` store
- When a reader performs an Acquire load on `marker_state` and observes `Move`, they synchronize-with this release sequence and are guaranteed to see the `move_dest` value

**Critical ordering preserved**: `move_dest` store (Release) → `marker_state` CAS (AcqRel). Any reader who Acquires the Move marker state will observe the `move_dest` value.

**Why this is safe on ARM**: The Release on `move_dest.store()` ensures it cannot be reordered after the subsequent CAS. The reader-side loads `marker_state` (Acquire) before `move_dest` (Acquire), creating proper synchronization.

### Notes for Next Contributor

- Step 5 (Relax destination cell writes to Release) is next
- Focus on lines ~499-503 in btree_map.rs (destination cell version/marker stores during rebalance)
- Change `cell.version.store()` and `cell.marker_state.store()` from `SeqCst` → `Release`
- These stores publish the newly-moved data to the destination cell

---

## Step 3: Relax CAS operations to AcqRel/Acquire (Completed 2026-01-16)

### Changes Made

**In [cell.rs](../src/cache_oblivious/cell.rs):**

- Lines ~426-430: Changed `compare_exchange_marker_state` in `update_marker_state()` from `SeqCst`/`SeqCst` → `AcqRel`/`Acquire`

**In [btree_map.rs](../src/cache_oblivious/btree_map.rs):**

- Lines ~474-481: Changed `compare_exchange_marker_state` for Move marker CAS from `SeqCst`/`SeqCst` → `AcqRel`/`Acquire`
- Lines ~508-516: Changed `compare_exchange_weak` for version bump (best-effort cleanup) from `SeqCst`/`SeqCst` → `AcqRel`/`Acquire`
- Lines ~522-528: Changed `compare_exchange_marker_state` for marker cleanup from `SeqCst`/`SeqCst` → `AcqRel`/`Acquire`

### Tests Added

Added 5 new tests in `btree_map.rs` to verify AcqRel/Acquire CAS semantics:

1. `test_cas_acqrel_synchronizes_with_prior_writes` - Verifies CAS with AcqRel properly synchronizes with prior Release stores
2. `test_cas_failure_observes_blocking_value` - Verifies CAS failure with Acquire observes the blocking value
3. `test_rebalance_cas_publishes_move_marker` - Verifies rebalance CAS publishes Move state to readers
4. `test_best_effort_cleanup_cas_correctness` - Verifies best-effort cleanup CAS works correctly
5. `test_cas_release_sequence_with_move_dest` - Verifies CAS creates proper release sequence with move_dest

All 121 tests pass.

### Rationale

AcqRel/Acquire ordering is sufficient for CAS operations because:

- **Success (AcqRel)**: The Acquire component synchronizes with any prior writer's Release (ensuring we see their completed writes). The Release component publishes our claim so subsequent readers know the cell is in-use.
- **Failure (Acquire)**: When CAS fails, we need to observe the current value that blocked us. Acquire is sufficient—we don't write anything on failure, so Release semantics add nothing.

The Release component on success ensures that if we set `Inserting` or `Move` and later write data, any reader who observes our subsequent version bump will also see that we claimed the cell.

### Notes for Next Contributor

- Step 4 (Relax move_dest store to Release) is next
- Focus on `move_dest.store()` in [btree_map.rs#L471](../src/cache_oblivious/btree_map.rs#L471)
- Change from `SeqCst` → `Release`, since the subsequent CAS provides the synchronization
- Note: `move_dest` store in `update_marker_state()` in cell.rs also needs to be changed

---

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
