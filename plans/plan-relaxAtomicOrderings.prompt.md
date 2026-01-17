## Plan: Relax SeqCst Atomic Orderings

Relax `SeqCst` atomic orderings to `Acquire`, `Release`, or `AcqRel` where the synchronization requirements permit, improving performance while preserving the wait-free protocol's correctness.

### Background: Why SeqCst Is Often Overkill

`SeqCst` (sequentially consistent) ordering provides a total global ordering of all atomic operations across all threads. This is the strongest guarantee but comes with performance costs—especially on ARM/AArch64 where it requires full memory barriers.

The wait-free protocol in this codebase uses a **version validation pattern**:

1. Writer claims exclusive access via CAS on `marker_state`
2. Writer modifies `key`/`value` (via `UnsafeCell`)
3. Writer increments `version` and resets `marker_state` to `Empty`
4. Reader loads `version` before reading data, validates it unchanged after

This pattern only requires **pairwise synchronization** between a writer and its readers—not total ordering across unrelated cells. The Release/Acquire model is sufficient: a Release store "publishes" writes, and an Acquire load "observes" those writes.

### Steps

1. [x] **Relax read-path loads to Acquire** in [btree_map.rs](src/cache_oblivious/btree_map.rs) and [cell.rs](src/cache_oblivious/cell.rs): Change `version.load()`, `load_marker_state()`, and `move_dest.load()` calls used for reading cell data from `SeqCst` → `Acquire`.

   **Exception**: Version re-reads used for seqlock validation (e.g., in `from_raw`) must remain `SeqCst`—see "Further Considerations" section.

   **Rationale**: Acquire ordering prevents subsequent reads (of `key`/`value` UnsafeCells) from being reordered before the version check. This is exactly what's needed—we must see the version _before_ we read data to validate consistency. We don't need total ordering with writes to _other_ cells; we only need to synchronize with the writer of _this_ cell.

   **Why Acquire is sufficient (not Relaxed)**: On ARM, without Acquire, the CPU could speculatively execute the UnsafeCell reads before the version load completes, potentially observing partially-written data even when the version check would have failed. Acquire creates a load-acquire barrier ensuring data reads are sequenced after the version is known.

   **Synchronizes with**: The writer's Release stores on `version` and `marker_state`.

2. [x] **Relax write-path stores to Release** in [btree_map.rs](src/cache_oblivious/btree_map.rs): Change `version.store()` and `marker_state.store()` calls after UnsafeCell writes from `SeqCst` → `Release` (lines ~130, ~135, ~149, ~153, ~179, ~183).

   **Rationale**: Release ordering ensures all preceding writes (including the UnsafeCell writes to `key` and `value`) are visible to any thread that subsequently performs an Acquire load on these atomics. This is the "publication" side of the pattern—once a reader sees the new version via Acquire, they're guaranteed to see the data that was written before the Release store.

   **Synchronizes with**: Any reader's Acquire load on `version` or `marker_state`.

3. [x] **Relax CAS operations to AcqRel/Acquire** in both files: Change `compare_exchange` and `compare_exchange_weak` success ordering from `SeqCst` → `AcqRel`, failure ordering from `SeqCst` → `Acquire` for `compare_exchange_marker_state` calls.

   **Rationale**:

   - **Success (AcqRel)**: The Acquire component synchronizes with any prior writer's Release (ensuring we see their completed writes). The Release component publishes our claim so subsequent readers know the cell is in-use.
   - **Failure (Acquire)**: When CAS fails, we need to observe the current value that blocked us. Acquire is sufficient—we don't write anything on failure, so Release semantics add nothing.

   **Why not just Acquire on success?** We need Release to ensure that if we set `Inserting` and later write data, any reader who observes our subsequent version bump will also see that we claimed the cell (in case they retry).

4. [x] **Relax move_dest store to Release** in [btree_map.rs#L381](src/cache_oblivious/btree_map.rs#L381): Change `move_dest.store()` before the Move CAS from `SeqCst` → `Release`, since the subsequent CAS provides the synchronization.

   **Rationale**: The `move_dest` is stored _before_ the CAS that sets `marker_state` to `Move`. The CAS itself uses AcqRel, which provides the synchronization point. However, we use Release on the `move_dest` store to ensure the destination index is visible before any reader observes the Move marker. The CAS's Release component then "carries" this write forward.

   **Critical ordering**: `move_dest` store (Release) → `marker_state` CAS (AcqRel). Any reader who Acquires the Move marker state will observe the `move_dest` value.

   **Why this is safe on ARM**: The Release on `move_dest.store()` ensures it cannot be reordered after the subsequent CAS. The CAS's AcqRel success ordering creates a release sequence that includes the prior `move_dest` store. When a reader performs an Acquire load on `marker_state` and observes `Move`, they synchronize-with this release sequence and are guaranteed to see the `move_dest` value.

   **Reader-side correctness**: In `from_raw`, the reader loads `marker_state` (Acquire) before `move_dest` (Acquire). This order is critical—if the reader sees `Move`, the Acquire on `marker_state` synchronizes with the writer's Release, ensuring the subsequent `move_dest` load sees the stored value. The loads cannot be reordered because each Acquire prevents subsequent operations from moving before it.

5. [ ] **Relax destination cell writes to Release** in rebalance (lines ~407-410): Change destination `version.store()` and `marker_state.store()` from `SeqCst` → `Release`.

   **Rationale**: These stores publish the newly-moved data to the destination cell. Release ensures the preceding UnsafeCell writes (`key.get().write()`, `value.get().write()`) are visible to readers who Acquire-load the destination's `version` or `marker_state`.

   **Synchronizes with**: Readers performing Acquire loads on the destination cell.

6. [ ] **Update cell.rs helper method orderings**: Ensure `update_marker_state()` uses `AcqRel`/`Acquire` for CAS and `Release` for `move_dest.store()`.

   **Rationale**: Centralizes the ordering semantics in the helper, so callers don't need to specify orderings. The helper encapsulates the protocol: store `move_dest` with Release (if Move), then CAS with AcqRel/Acquire.

### Further Considerations

1. [ ] **CellGuard::from_raw ordering**: Uses seqlock pattern with version re-read for validation.

   **Initial loads** (`version`, `marker_state`, `move_dest`): Can be relaxed to `Acquire`—synchronizes with any prior Release from writers.

   **Version re-read for validation**: **MUST remain `SeqCst`** (or use explicit `fence(SeqCst)` before it).

   **Rationale (ARM correctness)**: On ARM/AArch64, `Acquire` only prevents _subsequent_ operations from being reordered _before_ the acquire load. It does NOT prevent _prior_ operations from being reordered _after_ it. Without `SeqCst` on the re-read, the compiler/CPU could reorder the data reads (`key.clone()`, `marker_state`, `move_dest`) to execute _after_ the `version_reread`, breaking the seqlock validation invariant. On x86 (TSO), loads are never reordered with other loads so this wouldn't manifest, but the code must be correct on all architectures.

   **Pattern**:

   ```rust
   let version = cell.version.load(Acquire);        // Initial read: Acquire OK
   let key = (*cell.key.get()).clone();             // Data read
   let marker_state = cell.load_marker_state(Acquire);
   let move_dest = cell.move_dest.load(Acquire);
   let version_reread = cell.version.load(SeqCst);  // Validation: MUST be SeqCst
   if version != version_reread { return Err(...); }
   ```

2. [ ] **Best-effort cleanup CAS**: The source cell cleanup at lines ~414-424 uses `compare_exchange_weak`—can use `Release`/`Acquire` orderings.

   **Rationale**: The source cell's `key`/`value` are already cleared to `None` before these CAS operations. The CAS updates `version` and `marker_state` to reflect the cleared state. Release ensures the `None` writes are visible; Acquire on failure observes if another thread intervened. Since the code tolerates CAS failure (it's "best-effort"), relaxed failure ordering (`Relaxed`) might even suffice, but `Acquire` is conservative and has negligible cost.

   **Design note**: These are two separate CAS operations (version, then marker_state), not atomic together. A reader could theoretically observe `marker_state = Empty` with the old version if the first CAS failed but second succeeded. However, the protocol tolerates this: the reader would see a version mismatch on validation in `cache()`, triggering a retry. This is an inherent property of the current design, not an ordering issue.

3. [ ] **Testing**: Run `cargo test` and benchmarks after changes to verify no regressions in correctness or unexpected behavior under concurrency.

4. [ ] **cache() method ordering**: The version load in `cache()` for validation should also follow the seqlock pattern consideration.

   **Rationale**: `cache()` loads version, then reads UnsafeCell data. However, unlike `from_raw`, it does NOT re-read the version after the data read—it only compares against the cached version from guard creation. This is safe because:

   - The guard's `cache_version` was validated at creation time (via `from_raw`'s seqlock)
   - If a writer completes between guard creation and `cache()`, the version will have changed
   - The check `version != self.cache_version` catches this

   The single version load in `cache()` can be `Acquire`—it only needs to synchronize-with the writer's Release to see if a modification occurred.

### Why This Is Safe: The Synchronization Graph

```
Writer (Insert/Rebalance)                Reader (from_raw + cache)
─────────────────────────                ─────────────────────────
1. CAS marker_state [AcqRel]             1. load version [Acquire]
   (Empty → Inserting/Move)              2. clone key (UnsafeCell)
                                         3. load marker_state [Acquire]
2. write key/value (UnsafeCell)          4. load move_dest [Acquire]
                                         5. load version [SeqCst] ← seqlock validation
3. store version++ [Release] ─────────────→ (compare with step 1)
4. store marker_state=Empty [Release]
                                         Later, in cache():
                                         6. load version [Acquire]
                                            ↑ synchronizes-with step 3
                                         7. clone key/value (UnsafeCell)
```

The key insight: **each cell is an independent synchronization domain**. We only need Release/Acquire pairs between the writer of cell X and readers of cell X. We don't need total ordering between writes to cell X and writes to cell Y—those are independent operations that don't conflict.

`SeqCst` would only be necessary if the correctness depended on observing a global order like "cell A was modified before cell B." The current protocol doesn't have such dependencies—each cell's version validation is self-contained.

**Note on `load_marker()` helper**: This method reads `version`, `marker_state`, and `move_dest` in separate atomic loads, constructing a `Marker` enum. These reads are not atomic as a group, so the returned `Marker` could be inconsistent if a writer modifies the cell between loads. This is acceptable for uses like debug printing, but callers requiring consistency should use `CellGuard` which validates via the seqlock pattern.
