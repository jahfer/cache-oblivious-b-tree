## Plan: Refactor Marker System for Safety & Clarity

This plan restructures the marker-based concurrency mechanism to unify version tracking into a single source of truth, adopt `crossbeam-epoch` for safe memory reclamation, and complete outstanding TODO items before encapsulating everything behind a type-safe API.

### Steps

1. [ ] **Add `crossbeam-epoch` dependency** — update [Cargo.toml](Cargo.toml) to include `crossbeam-epoch = "0.9"` and familiarize with `Atomic<T>`, `Owned`, `Shared`, and `Guard` types.

2. [ ] **Complete rebalance retry logic** — replace the six `todo!()` panics in [btree_map.rs#L283-L333](src/cache_oblivious/btree_map.rs#L283-L333) with proper retry loops that restart the rebalance operation on version mismatch or CAS failure.

3. [ ] **Add version validation to `CellGuard::from_ptr`** — address the TODO at [cell.rs#L255](src/cache_oblivious/cell.rs#L255) by checking that the marker's embedded version matches the cell version before returning a valid guard.

4. [ ] **Replace `AtomicPtr` with `crossbeam_epoch::Atomic`** — change `marker: Option<AtomicPtr<Marker<K, V>>>` to `marker: Atomic<Marker<K, V>>` in [cell.rs](src/cache_oblivious/cell.rs), removing the `Option` wrapper entirely.

5. [ ] **Eliminate marker memory reuse pattern** — remove the unsafe reuse at [cell.rs#L151](src/cache_oblivious/cell.rs#L151) and [packed_memory_array.rs#L319](src/cache_oblivious/packed_memory_array.rs#L319); instead allocate fresh `Owned<Marker>` and defer-destroy old markers via `Guard::defer_destroy()`.

6. [ ] **Thread `Guard` through read/write operations** — update `cache()`, `update()`, and PMA iteration to accept or pin a `crossbeam_epoch::Guard`, ensuring markers aren't reclaimed while readers hold references.

7. [ ] **Split `CellGuard` into `CellReadGuard` / `CellWriteGuard`** — the write guard auto-commits version bump and defers marker destruction on `Drop`; read guard holds an epoch `Guard` to keep marker alive during access.

8. [ ] **Encapsulate `Cell` fields as private** — hide `version`, `marker`, `key`, `value` behind safe methods like `try_claim() -> Option<CellWriteGuard>` and `read(guard) -> Option<CellReadGuard>`.

9. [ ] **Add `#[must_use]`, typed errors, and documentation** — annotate fallible methods, create `CellError` enum, add module-level docs with state-transition diagram and `// SAFETY:` comments for remaining `unsafe` blocks.

### Further Considerations

1. **Unify version tracking?** Currently versions exist in `Cell::version`, inside `Marker` variants, and `CellGuard::cache_version` — consider consolidating into `Cell::version` only once the API is stable.
2. **Pin granularity?** Pin once per high-level operation (insert/lookup) vs. per-cell access — broader pins reduce overhead but delay reclamation longer. -> Yes, let's go with per-operation pinning for simplicity and accept the higher memory usage.
3. **Generalize `active_range` calculation?** The TODO at [packed_memory_array.rs#L23](src/cache_oblivious/packed_memory_array.rs#L23) suggests this helper could be extracted — worth addressing now or defer?
4. **Benchmark before/after?** The existing Criterion benchmarks in [benches/](benches/) can validate that epoch-based reclamation doesn't regress performance.
