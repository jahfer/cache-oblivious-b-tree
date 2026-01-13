## Plan: SeqLock for Cell Key/Value Access (Updated)

Replace the marker-based protocol with a pure SeqLock pattern using `AtomicU32`. `CellGuard` will cache the version at creation time to detect if the cell changed since the guard was created.

### What Gets Removed

| Component                                | Location                                                   | Reason                             |
| ---------------------------------------- | ---------------------------------------------------------- | ---------------------------------- |
| `Marker<K, V>` enum                      | [cell.rs#L68-L81](src/cache_oblivious/cell.rs#L68-L81)     | Replaced by SeqLock version        |
| `marker` field in `Cell`                 | [cell.rs#L85](src/cache_oblivious/cell.rs#L85)             | No longer needed                   |
| `marker` field in `CellData`             | [cell.rs#L149](src/cache_oblivious/cell.rs#L149)           | No longer needed                   |
| `cache_marker_ptr` in `CellGuard`        | [cell.rs#L157](src/cache_oblivious/cell.rs#L157)           | Replaced by `cached_version`       |
| `cache_version` (u16) in `CellGuard`     | [cell.rs#L155](src/cache_oblivious/cell.rs#L155)           | Replaced by `cached_version` (u32) |
| `CellGuard::update()` method             | [cell.rs#L192-L212](src/cache_oblivious/cell.rs#L192-L212) | Replaced by SeqLock writes         |
| Marker allocation in `Cell::new/default` | [cell.rs#L94-L111](src/cache_oblivious/cell.rs#L94-L111)   | No marker to allocate              |
| Marker deallocation in `Drop`            | [cell.rs#L113-L118](src/cache_oblivious/cell.rs#L113-L118) | No marker to free                  |

### Steps

1. ~~**Change version to `AtomicU32`** in [cell.rs#L84](src/cache_oblivious/cell.rs#L84) and initialize to `0` (even = stable).~~

2. ~~**Add SeqLock primitives to `Cell`**:~~

   - ~~`begin_write() -> SeqLockWriteGuard` — spins until version is even, CAS to odd~~
   - ~~`read_consistent<F, R>(f: F) -> R` — loop until stable read~~
   - ~~`SeqLockWriteGuard` — RAII guard that bumps version to even on drop~~

3. ~~**Update `CellGuard`** in [cell.rs#L150-L160](src/cache_oblivious/cell.rs#L150-L160):~~

   - ~~Replace `cache_version: u16` with `cached_version: u32`~~
   - ~~Add `is_stale() -> bool` method to check if cell changed since guard creation~~
   - ~~`cache()` validates against `cached_version` before/after reading~~
   - ~~`from_raw()` captures version at guard creation~~
   - _Note: `cache_marker_ptr` kept temporarily for backward compatibility with `update()`_

### Remaining Steps (Revised)

The following steps remove marker-based logic incrementally. Each step should result in a passing test suite.

4. ~~**Update write sites in `btree_map.rs` to use `begin_write()`**:~~

   - ~~Find all `cell.inner.key.get().write()` / `cell.inner.value.get().write()` calls~~
   - ~~Wrap each write with `cell.inner.begin_write()` guard~~
   - ~~Keep marker logic in parallel for now (dual-write) to maintain compatibility~~
   - ~~Update version bump logic to rely on `SeqLockWriteGuard` drop~~

5. **Remove `CellGuard::update()` method**:

   - After Step 4, marker CAS is no longer needed for correctness
   - Remove the `update()` method from `CellGuard`
   - Remove `cache_marker_ptr` field from `CellGuard`
   - Update any call sites that used `update()` to use `begin_write()` directly

6. **Remove `marker` field from `CellData`**:

   - `CellData` should only contain `key` and `value`
   - Update `cache()` method to not read/store marker
   - Remove marker cloning in `cache()`

7. **Remove `marker` field from `Cell`**:

   - Remove `marker: Option<AtomicPtr<Marker<K, V>>>` from `Cell` struct
   - Simplify `Cell::new()` — no marker allocation needed
   - Simplify `Default` impl — no marker allocation needed
   - Remove `Drop` impl for `Cell` — no marker pointer to deallocate

8. **Remove `Marker<K, V>` enum**:

   - Delete the `Marker` enum definition
   - Remove any remaining marker-related imports or dead code
   - Clean up `Debug` impl for `Cell` if it references markers

9. **Final cleanup**:

   - Remove any `#[allow(dead_code)]` that was added during transition
   - Verify no marker-related code remains
   - Update documentation/comments to reflect SeqLock-only approach

### Simplified Structures

```rust
pub struct Cell<K: Clone, V: Clone> {
    pub version: AtomicU32,              // SeqLock version (even = stable)
    pub key: UnsafeCell<Option<K>>,
    pub value: UnsafeCell<Option<V>>,
}

pub struct CellData<K: Clone, V: Clone> {
    pub key: K,
    pub value: V,
}

pub struct CellGuard<'a, K: Clone, V: Clone> {
    pub inner: &'a Cell<K, V>,
    pub is_filled: bool,
    cached_version: u32,                 // Version at guard creation
    cache_data: OnceCell<Option<CellData<K, V>>>,
    _phantom: PhantomData<&'a Cell<K, V>>,
}

impl<K: Clone, V: Clone> CellGuard<'_, K, V> {
    /// Returns true if the cell has been modified since this guard was created.
    pub fn is_stale(&self) -> bool {
        self.inner.version.load(Ordering::Acquire) != self.cached_version
    }
}
```

### SeqLock API

```rust
impl<K: Clone, V: Clone> Cell<K, V> {
    pub fn begin_write(&self) -> SeqLockWriteGuard<'_, K, V> {
        loop {
            let v = self.version.load(Ordering::Acquire);
            if v % 2 == 1 { std::hint::spin_loop(); continue; }
            if self.version.compare_exchange_weak(
                v, v.wrapping_add(1), Ordering::AcqRel, Ordering::Acquire
            ).is_ok() {
                return SeqLockWriteGuard { cell: self };
            }
        }
    }

    pub fn read_consistent<F, R>(&self, f: F) -> R
    where F: Fn(&Option<K>, &Option<V>) -> R, R: Clone {
        loop {
            let v1 = self.version.load(Ordering::Acquire);
            if v1 % 2 == 1 { std::hint::spin_loop(); continue; }

            let result = f(unsafe { &*self.key.get() }, unsafe { &*self.value.get() });

            std::sync::atomic::fence(Ordering::Acquire);
            if v1 == self.version.load(Ordering::Relaxed) {
                return result;
            }
        }
    }

    /// Read and return both the data and the version observed.
    pub fn read_with_version<F, R>(&self, f: F) -> (R, u32)
    where F: Fn(&Option<K>, &Option<V>) -> R, R: Clone {
        loop {
            let v1 = self.version.load(Ordering::Acquire);
            if v1 % 2 == 1 { std::hint::spin_loop(); continue; }

            let result = f(unsafe { &*self.key.get() }, unsafe { &*self.value.get() });

            std::sync::atomic::fence(Ordering::Acquire);
            if v1 == self.version.load(Ordering::Relaxed) {
                return (result, v1);
            }
        }
    }
}

pub struct SeqLockWriteGuard<'a, K: Clone, V: Clone> {
    cell: &'a Cell<K, V>,
}

impl<K: Clone, V: Clone> SeqLockWriteGuard<'_, K, V> {
    pub fn write(&self, key: Option<K>, value: Option<V>) {
        unsafe {
            *self.cell.key.get() = key;
            *self.cell.value.get() = value;
        }
    }
}

impl<K: Clone, V: Clone> Drop for SeqLockWriteGuard<'_, K, V> {
    fn drop(&mut self) {
        self.cell.version.fetch_add(1, Ordering::Release);
    }
}
```

### Further Considerations

1. **Backoff strategy?** Current plan uses `spin_loop()`. Could add `thread::yield_now()` after N iterations for better fairness under contention.
