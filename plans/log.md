# Summary of work completed

## Step 1: Change version to `AtomicU32` (Completed)

**Date:** January 12, 2026

**Changes made:**

- Changed `Cell.version` from `AtomicU16` to `AtomicU32` in [cell.rs](../src/cache_oblivious/cell.rs)
- Updated `CellGuard.cache_version` from `u16` to `u32`
- Updated all `Marker` enum variants from `u16` to `u32` versions
- Changed initial version from `1` to `0` (even = stable state for SeqLock)
- Updated `Cell::new()` and `Default` impl to initialize with version `0`

**Tests added:**

- `test_cell_default_version_is_zero_even` - verifies default Cell starts with version 0
- `test_cell_new_version_is_zero_even` - verifies Cell::new starts with version 0
- `test_marker_version_is_u32` - verifies Marker uses u32 for version
- `test_marker_all_variants_have_u32_version` - tests all Marker variants
- `test_cell_guard_cache_version_is_u32` - verifies CellGuard uses u32
- `test_version_can_hold_large_values` - verifies values > u16::MAX work

**Notes for next contributor:**

- The version field now uses SeqLock semantics: even = stable, odd = write in progress
- All 65 tests pass (59 original + 6 new)
- Next step: Add SeqLock primitives (`begin_write()`, `read_consistent()`, `SeqLockWriteGuard`)

---

## Step 2: Add SeqLock primitives to `Cell` (Completed)

**Date:** January 12, 2026

**Changes made:**

- Added `begin_write() -> SeqLockWriteGuard` method to `Cell`
  - Spins until version is even (stable), then CAS to odd (write in progress)
  - Returns RAII guard for exclusive write access
- Added `read_consistent<F, R>(f: F) -> R` method to `Cell`
  - Loops until stable read achieved (version unchanged and even throughout)
  - Uses memory fence to ensure all reads complete before version check
- Added `read_with_version<F, R>(f: F) -> (R, u32)` method to `Cell`
  - Same as `read_consistent` but also returns the version observed
  - Useful for caching version to detect staleness later
- Added `SeqLockWriteGuard` struct with:
  - `write(key, value)` - write new key/value to the cell
  - `key_mut()` - get mutable reference to key
  - `value_mut()` - get mutable reference to value
  - `Drop` impl that bumps version from odd to even (releases write lock)

**Tests added:**

- `test_begin_write_acquires_lock` - verifies version becomes odd when guard is held
- `test_write_guard_releases_on_drop` - verifies version becomes even after drop
- `test_write_guard_write_method` - tests the write() method works correctly
- `test_write_guard_key_mut_and_value_mut` - tests mutable reference methods
- `test_read_consistent_returns_stable_data` - tests consistent reads
- `test_read_consistent_empty_cell` - tests reading empty cell
- `test_read_with_version_returns_version` - tests version is returned correctly
- `test_multiple_writes_increment_version` - verifies version increments by 2 per write cycle
- `test_version_wrapping` - tests u32 wrapping behavior at u32::MAX
- `test_seqlock_write_guard_is_send` - verifies SeqLockWriteGuard is Send

**Notes for next contributor:**

- SeqLock primitives are now available but not yet integrated into existing code
- All 72 tests pass (65 previous + 7 new SeqLock tests)
- Next step: Update `CellGuard` to use SeqLock primitives (replace `cache_version: u16` with `cached_version: u32`, remove `cache_marker_ptr`, add `is_stale()` method)

---

## Step 3: Update `CellGuard` with SeqLock support (Completed)

**Date:** January 12, 2026

**Changes made:**

- Renamed `cache_version` to `cached_version` in `CellGuard` struct (clearer naming)
- Added `is_stale() -> bool` method to `CellGuard`
  - Returns `true` if cell's version differs from `cached_version`
  - Useful for detecting concurrent modifications
- Updated `cache()` method to validate against `cached_version`
  - Checks `is_stale()` before and after reading data
  - Returns `CellReadError` if staleness detected
- Updated `from_raw()` to capture version using `Acquire` ordering
- Updated `btree_map.rs` to use `cached_version` instead of `cache_version`

**Note:** `cache_marker_ptr` field was NOT removed in this step

- The `update()` method still depends on `cache_marker_ptr` for marker CAS logic
- This will be addressed in Step 7 when marker-based writes are replaced with SeqLock writes
- Conservative approach avoids breaking existing functionality

**Tests added:**

- `test_cell_guard_cached_version_is_u32` - verifies cached_version field exists and is initialized correctly
- `test_cell_guard_is_stale_returns_false_when_unchanged` - verifies is_stale() returns false for unchanged cells
- `test_cell_guard_is_stale_returns_true_after_write` - verifies is_stale() detects writes via begin_write()
- `test_cell_guard_is_stale_detects_version_change` - verifies is_stale() detects manual version changes
- `test_cell_guard_from_raw_captures_version` - verifies from_raw() captures current version
- `test_cell_guard_is_filled_correct` - verifies is_filled/is_empty work correctly with SeqLock writes

**Notes for next contributor:**

- CellGuard now has `is_stale()` for detecting concurrent modifications
- The field is now `cached_version` (was `cache_version`)
- `cache_marker_ptr` still exists temporarily for backward compatibility with `update()` method
- All 78 tests pass (72 previous + 6 new CellGuard tests)
- Next step: Update write sites in `btree_map.rs` to use `begin_write()` (Step 4)

---

## Step 4: Update write sites to use `begin_write()` (Completed)

**Date:** January 12, 2026

**Changes made:**

- Updated `insert()` method in [btree_map.rs](../src/cache_oblivious/btree_map.rs):
  - Wrapped key/value writes with `begin_write()` SeqLock guard
  - Version is now managed by SeqLockWriteGuard (increments on drop)
  - Marker is updated after write with the new version for compatibility
- Updated `rebalance()` method in [btree_map.rs](../src/cache_oblivious/btree_map.rs):
  - Wrapped destination cell writes with `begin_write()` guard
  - Wrapped source cell clearing with `begin_write()` guard
  - **Critical fix:** Added marker update for destination cell to keep version/marker in sync
  - Source cell marker is also updated with new version after SeqLock write

**Key insight:** During dual-write transition, both source AND destination cells need their markers updated after SeqLock writes. Otherwise, subsequent rebalance operations fail the `version != marker_version` check.

**Tests added:**

- `test_insert_uses_seqlock_for_writes` - verifies insert uses SeqLock and keeps version/marker in sync
- `test_rebalance_updates_both_cell_markers` - verifies rebalance updates markers for both source and destination
- `test_seqlock_version_increments_by_two_per_write` - verifies SeqLock increments version by 2 per write cycle

**Notes for next contributor:**

- All write operations now use SeqLock for exclusive access
- Marker logic is kept in parallel (dual-write) for backward compatibility
- Version management is now handled by SeqLockWriteGuard drop
- All 81 tests pass (78 previous + 3 new Step 4 tests)
- Next step: Remove `CellGuard::update()` method (Step 5) - marker CAS is no longer needed for correctness

---

## Step 5: Remove `CellGuard::update()` method (Completed)

**Date:** January 12, 2026

**Changes made:**

- Removed `update()` method from `CellGuard` in [cell.rs](../src/cache_oblivious/cell.rs)
  - The method was doing marker CAS, now replaced by SeqLock version CAS
- Removed `cache_marker_ptr` field from `CellGuard` struct
  - This field was only used by `update()` method
- Updated `from_raw()` to no longer read/store marker pointer
- Added `try_begin_write(expected_version)` method to `Cell`
  - Returns `Some(guard)` only if current version matches expected_version
  - Provides atomic claim behavior via version CAS (replaces marker CAS)
- Updated `insert()` in [btree_map.rs](../src/cache_oblivious/btree_map.rs):
  - Replaced `cell.update(marker)` with `cell.inner.try_begin_write(cell.cached_version)`
  - Added logic to refresh `cached_version` after rebalance operations
  - Removed `marker_version` calculation and `Marker::InsertCell` creation

**Key insight:** After rebalance operations, the cell's version changes (because rebalance uses `begin_write()` to clear cells). The `cached_version` must be refreshed after rebalance to ensure `try_begin_write()` succeeds. Without this fix, inserts that trigger rebalancing would fail to write.

**Tests added:**

- `test_try_begin_write_succeeds_with_matching_version` - verifies try_begin_write works with correct version
- `test_try_begin_write_fails_with_mismatched_version` - verifies failure with wrong version
- `test_try_begin_write_fails_with_odd_expected_version` - verifies rejection of odd versions
- `test_try_begin_write_fails_when_write_in_progress` - verifies failure when another write is active
- `test_try_begin_write_releases_lock_on_drop` - verifies proper lock release
- `test_cell_guard_no_cache_marker_ptr_field` - compile-time verification of removed field

**Notes for next contributor:**

- `CellGuard` no longer has `update()` method or `cache_marker_ptr` field
- `Cell` now has `try_begin_write(expected_version)` for conditional write acquisition
- Insert uses version CAS instead of marker CAS for concurrency control
- All 87 tests pass (81 previous + 6 new Step 5 tests)
- Next step: Remove `marker` field from `CellData` (Step 6)
