## Plan: Refactor Marker System for Safety & Clarity

This plan restructures the marker-based concurrency mechanism to unify version tracking into a single source of truth, adopt `crossbeam-epoch` for safe memory reclamation, and complete outstanding TODO items before encapsulating everything behind a type-safe API.

### Steps

1. [x] **Add `crossbeam-epoch` dependency** — update [Cargo.toml](Cargo.toml) to include `crossbeam-epoch = "0.9"` and familiarize with `Atomic<T>`, `Owned`, `Shared`, and `Guard` types.

2. [x] **Complete rebalance retry logic** — replace the `todo!()` panics in [btree_map.rs](src/cache_oblivious/btree_map.rs) with proper retry loops that restart the rebalance operation on version mismatch or CAS failure.

3. [ ] **Add version validation to `CellGuard::from_ptr`** — address the TODO at [cell.rs#L255](src/cache_oblivious/cell.rs#L255) by checking that the marker's embedded version matches the cell version before returning a valid guard.

4. [ ] **Update destination cell version/marker during rebalance** — address the TODO at [btree_map.rs#L318](src/cache_oblivious/btree_map.rs#L318) to properly set the version and marker on the newly-written destination cell after moving data.

5. [ ] **Complete old cell cleanup after move** — address the TODO at [btree_map.rs#L339](src/cache_oblivious/btree_map.rs#L339) to ensure the source cell's version is incremented and marker is cleared after its contents are moved.

6. [ ] **Handle overwriting moved records** — address the TODO at [btree_map.rs#L277](src/cache_oblivious/btree_map.rs#L277) to determine if destination cells with existing keys can be safely overwritten during rebalance.

7. [ ] **Fix suspicious `into_iter()` usage in rebalance** — address the todo at [btree_map.rs#L292](src/cache_oblivious/btree_map.rs#L292) where `self.data.into_iter().next()` is used to compute destination index; this may not return the expected base pointer.

8. [ ] **Consider atomic CAS for neighbouring cells** — investigate the TODO at [btree_map.rs#L270](src/cache_oblivious/btree_map.rs#L270) about using compare_and_swap for neighbouring cells to prevent leaving cells unallocated during rebalance.

9. [ ] **Generalize `active_range` calculation** — address the TODO at [packed_memory_array.rs#L23](src/cache_oblivious/packed_memory_array.rs#L23) to extract the active range computation into a reusable helper.

10. [ ] **Replace `AtomicPtr` with `crossbeam_epoch::Atomic`** — change `marker: Option<AtomicPtr<Marker<K, V>>>` to `marker: Atomic<Marker<K, V>>` in [cell.rs](src/cache_oblivious/cell.rs), removing the `Option` wrapper entirely.

11. [ ] **Eliminate marker memory reuse pattern** — remove the unsafe reuse at [cell.rs#L151](src/cache_oblivious/cell.rs#L151) and [packed_memory_array.rs#L319](src/cache_oblivious/packed_memory_array.rs#L319); instead allocate fresh `Owned<Marker>` and defer-destroy old markers via `Guard::defer_destroy()`.

12. [ ] **Thread `Guard` through read/write operations** — update `cache()`, `update()`, and PMA iteration to accept or pin a `crossbeam_epoch::Guard`, ensuring markers aren't reclaimed while readers hold references.

13. [ ] **Split `CellGuard` into `CellReadGuard` / `CellWriteGuard`** — the write guard auto-commits version bump and defers marker destruction on `Drop`; read guard holds an epoch `Guard` to keep marker alive during access.

14. [ ] **Encapsulate `Cell` fields as private** — hide `version`, `marker`, `key`, `value` behind safe methods like `try_claim() -> Option<CellWriteGuard>` and `read(guard) -> Option<CellReadGuard>`.

15. [ ] **Add `#[must_use]`, typed errors, and documentation** — annotate fallible methods, create `CellError` enum, add module-level docs with state-transition diagram and `// SAFETY:` comments for remaining `unsafe` blocks.

### Further Considerations

1. **Unify version tracking?** Currently versions exist in `Cell::version`, inside `Marker` variants, and `CellGuard::cache_version` — consider consolidating into `Cell::version` only once the API is stable.
2. **Pin granularity?** Pin once per high-level operation (insert/lookup) vs. per-cell access — broader pins reduce overhead but delay reclamation longer. -> Yes, let's go with per-operation pinning for simplicity and accept the higher memory usage.
3. **Generalize `active_range` calculation?** The TODO at [packed_memory_array.rs#L23](src/cache_oblivious/packed_memory_array.rs#L23) suggests this helper could be extracted — worth addressing now or defer?
4. **Benchmark before/after?** The existing Criterion benchmarks in [benches/](benches/) can validate that epoch-based reclamation doesn't regress performance.
