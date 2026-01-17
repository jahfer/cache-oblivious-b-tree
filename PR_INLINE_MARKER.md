# Inline Cell Marker for Improved Performance

## Summary

This PR eliminates heap allocation for cell markers by inlining atomic fields directly into the `Cell` struct. Previously, each cell stored a pointer to a heap-allocated `Marker<K, V>` enum, requiring an indirection on every read/write operation. Now, the marker state is stored as inline atomic fields within the cell itself.

## Motivation

Profiling revealed that the heap-allocated marker was a significant source of overhead:

1. **Pointer indirection**: Every cell access required dereferencing an `AtomicPtr` to read the marker state
2. **Memory allocation**: Creating/updating markers required `Box::new()` and `Box::into_raw()` calls
3. **Cache inefficiency**: Marker data was scattered across the heap rather than co-located with cell data
4. **Memory overhead**: Each marker allocation included allocator metadata overhead

## Changes

### Before

```rust
pub struct Cell<K: Clone, V: Clone> {
    pub version: AtomicU16,
    pub marker: Option<AtomicPtr<Marker<K, V>>>,  // Heap-allocated pointer
    pub key: UnsafeCell<Option<K>>,
    pub value: UnsafeCell<Option<V>>,
}

pub enum Marker<K: Clone, V: Clone> {
    Empty(u16),
    Move(u16, isize),
    InsertCell(u16, K, V),   // Stored K, V redundantly
    DeleteCell(u16, K),       // Stored K redundantly
}
```

### After

```rust
pub struct Cell<K: Clone, V: Clone> {
    pub version: AtomicU16,
    pub marker_state: AtomicU8,    // Inline: Empty=0, Move=1, Inserting=2, Deleting=3
    pub move_dest: AtomicIsize,    // Inline: destination for Move operations
    pub key: UnsafeCell<Option<K>>,
    pub value: UnsafeCell<Option<V>>,
}

#[repr(u8)]
pub enum MarkerState {
    Empty = 0,
    Move = 1,
    Inserting = 2,
    Deleting = 3,
}
```

### Key Insight

The `InsertCell(version, K, V)` and `DeleteCell(version, K)` marker variants stored key/value data that was **never read by other threads**—they only served as intent markers. Only the operation state and the `move_dest` index (for rebalancing) need to be atomically visible. This allows us to replace the polymorphic heap-allocated enum with fixed-size inline atomics.

## Benchmark Results

All benchmarks run on the same hardware, comparing before/after this change.

### Point Lookups (Read Performance)

| Dataset Size | Before   | After    | Improvement |
| ------------ | -------- | -------- | ----------- |
| 100          | 81.7 ns  | 78.7 ns  | —           |
| 500          | 92.5 ns  | 87.6 ns  | **+6.1%**   |
| 1000         | 91.9 ns  | 87.9 ns  | **+5.3%**   |
| 2000         | 100.2 ns | 91.9 ns  | **+9.0%**   |
| 5000         | 113.9 ns | 104.7 ns | **+8.9%**   |

### Sequential Inserts

| Dataset Size | Before  | After   | Improvement |
| ------------ | ------- | ------- | ----------- |
| 100          | 3.70 ms | 2.80 ms | **+32%**    |
| 500          | 14.2 ms | 13.4 ms | **+5.7%**   |
| 1000         | 27.1 ms | 26.5 ms | **+2.3%**   |

### Random Inserts (Most Dramatic Improvement)

| Dataset Size | Before  | After   | Improvement  |
| ------------ | ------- | ------- | ------------ |
| 100          | 1.59 ms | 417 µs  | **+281%** 🚀 |
| 500          | 2.01 ms | 725 µs  | **+177%** 🚀 |
| 1000         | 2.33 ms | 888 µs  | **+160%** 🚀 |
| 2000         | 3.42 ms | 1.71 ms | **+100%** 🚀 |
| 5000         | 5.62 ms | 3.44 ms | **+63%**     |

### Mixed Workload (80% Read / 20% Write)

| Dataset Size | Before | After   | Improvement |
| ------------ | ------ | ------- | ----------- |
| 500          | 104 µs | 96 µs   | **+8.5%**   |
| 1000         | 107 µs | 96.5 µs | **+11%**    |
| 2000         | 116 µs | 102 µs  | **+14%**    |

### Strided Access

| Structure | Before | After  | Improvement |
| --------- | ------ | ------ | ----------- |
| COBTree   | 989 ns | 929 ns | **+6.8%**   |

## Why Random Inserts Improved So Dramatically

Random inserts trigger frequent rebalancing operations. The old implementation:

1. Allocated a new `Marker::Move(version, dest_index)` via `Box::new()`
2. Performed CAS to swap the pointer
3. On failure, deallocated the marker and retried
4. On success, copied data, then allocated another marker for the destination cell

The new implementation:

1. Performs a single `compare_exchange` on the `AtomicU8` marker state
2. Stores the destination index in the inline `AtomicIsize`
3. No heap allocation, no deallocation on retry

This eliminates **all heap allocations** from the hot path of cell operations.

## Testing

All 103 existing tests pass. New tests added for:

- `MarkerState` enum conversion and methods
- `Cell::load_marker()` reconstruction
- `CellGuard` version change detection on cache access

## Breaking Changes

- `Cell::new()` no longer takes a marker pointer argument
- `CellGuard::update()` replaced with `update_marker_state(MarkerState, isize)`
- `Marker` enum is now non-generic (no longer `Marker<K, V>`) - type parameters removed entirely since only the state matters, not the key/value types

## Atomicity Improvements

The `move_dest` field is now stored **before** the marker state CAS during Move operations. This ensures that any reader who observes a `Move` marker state will always see a valid `move_dest` value. If the CAS fails, the stale `move_dest` is harmless since the marker state won't indicate `Move`.
