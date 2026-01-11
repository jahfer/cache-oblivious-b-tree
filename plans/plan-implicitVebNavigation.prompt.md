# Plan: Implicit vEB Navigation and Incremental Index Updates

The cache-oblivious B-tree design uses **two separate structures**:

1. A **static search tree** in vEB layout for indexing
2. A **packed-memory array (PMA)** for data storage

The cache-oblivious property comes from the vEB layout of the search tree itself — not from unifying it with the PMA. The current implementation has the right architecture but uses explicit pointers and full rebuilds unnecessarily.

## Current Problems

| Issue           | Current State                          | Should Be                          |
| --------------- | -------------------------------------- | ---------------------------------- |
| Tree navigation | `NonNull` pointer fields in each node  | Implicit position arithmetic       |
| Node size       | `min_rhs` + 2 pointers (24+ bytes)     | `min_rhs` only (8 bytes)           |
| Index updates   | Full O(N) rebuild on every insert      | Incremental O(affected leaves)     |
| Update timing   | Async 50ms delay, eventual consistency | Synchronous, immediate consistency |

## Part 1: Implicit Addressing in BlockSearchTree

Remove explicit `left`/`right` pointers from `Node::Internal`. Compute child positions from the current index using the vEB recursive structure.

### Current Node Structure

```rust
enum Node<K, V> {
    Leaf(Key<K>, Block<K, V>),
    Internal {
        min_rhs: Key<K>,
        left: MaybeUninit<NonNull<UnsafeCell<Node<K, V>>>>,   // Remove
        right: MaybeUninit<NonNull<UnsafeCell<Node<K, V>>>>,  // Remove
    },
}
```

### Proposed Node Structure

```rust
enum Node<K, V> {
    Leaf { min_key: Key<K>, block: Block<K, V> },
    Internal { min_rhs: Key<K> },  // No pointers!
}
```

### Navigation via Position Arithmetic

The vEB layout places nodes such that child positions can be computed from:

- Current position in the array
- Current depth in the tree
- Precomputed subtree sizes at each recursion level

```rust
struct BlockSearchTree<K, V> {
    nodes: Box<[Node<K, V>]>,        // No UnsafeCell needed without pointers
    subtree_sizes: Box<[usize]>,     // Precomputed sizes at each vEB depth
    height: u8,
}

impl<K, V> BlockSearchTree<K, V> {
    fn left_child(&self, pos: usize, veb_depth: usize) -> usize {
        // Within upper subtree: simple offset
        // Crossing to lower subtree: jump to lower region
        self.compute_child_offset(pos, veb_depth, false)
    }

    fn right_child(&self, pos: usize, veb_depth: usize) -> usize {
        self.compute_child_offset(pos, veb_depth, true)
    }

    fn compute_child_offset(&self, pos: usize, veb_depth: usize, is_right: bool) -> usize {
        // Use subtree_sizes table to determine offset
        // Track whether we're in upper or lower subtree region
        todo!()
    }
}
```

### vEB Subtree Size Precomputation

At construction time, build a small lookup table ($O(\log \log N)$ entries):

```rust
fn precompute_subtree_sizes(total_height: u8) -> Box<[usize]> {
    let mut sizes = Vec::new();
    let mut h = total_height as usize;

    while h > 0 {
        let lower_h = (h / 2).next_power_of_two().max(1);
        let upper_h = h - lower_h;
        sizes.push((1 << upper_h) - 1);  // Upper subtree node count
        sizes.push((1 << lower_h) - 1);  // Lower subtree node count
        h = lower_h;
    }

    sizes.into_boxed_slice()
}
```

## Part 2: Incremental Leaf Updates

Instead of rebuilding the entire index when the PMA rebalances, update only the affected leaf nodes.

### Track Dirty Leaves

```rust
struct BlockSearchTree<K, V> {
    nodes: Box<[Node<K, V>]>,
    subtree_sizes: Box<[usize]>,
    height: u8,
    first_leaf_index: usize,  // Position of leftmost leaf in nodes[]
}

impl<K, V> BlockSearchTree<K, V> {
    /// Update a single leaf's min_key after PMA rebalancing
    fn update_leaf(&mut self, leaf_index: usize, new_min_key: Key<K>) {
        let pos = self.first_leaf_index + leaf_index;
        if let Node::Leaf { ref mut min_key, .. } = self.nodes[pos] {
            *min_key = new_min_key;
        }
        // Optionally propagate min_rhs changes up the tree
        self.propagate_key_change(pos);
    }

    /// Propagate key changes up to ancestors if needed
    fn propagate_key_change(&mut self, leaf_pos: usize) {
        // Only internal nodes storing min_rhs of affected subtrees need updates
        // Walk up parent chain, updating min_rhs where this leaf is the minimum
    }
}
```

### Hook PMA Rebalancing to Index

Modify PMA to report which block ranges were affected:

```rust
struct RebalanceResult {
    affected_blocks: Range<usize>,  // Which leaf indices need min_key refresh
}

impl<T> PackedMemoryArray<T> {
    fn rebalance(&mut self, ...) -> RebalanceResult {
        // ... existing rebalancing logic ...
        RebalanceResult { affected_blocks: start..end }
    }
}
```

Then in `BTreeMap::insert`:

```rust
fn insert(&mut self, key: K, value: V) {
    // ... insert into PMA ...
    let result = self.data.rebalance(...);

    // Update only affected leaves
    for block_idx in result.affected_blocks {
        let new_min = self.compute_block_min_key(block_idx);
        self.index.update_leaf(block_idx, new_min);
    }
}
```

## Part 3: Remove Async Indexing

Delete the background thread infrastructure entirely:

### Remove These Components

- `INDEX_UPDATE_DELAY` constant
- `tx: Sender<...>` channel
- `index_updating: Arc<AtomicBool>` flag
- `start_indexing_thread()` function
- `request_reindex()` method

### Replace With Synchronous Updates

```rust
pub struct BTreeMap<K, V> {
    data: PackedMemoryArray<Cell<K, V>>,      // No Arc needed if single-threaded
    index: BlockSearchTree<K, V>,              // No RwLock needed
}

impl<K, V> BTreeMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) {
        // 1. Find block via index
        // 2. Insert into PMA
        // 3. If rebalance occurred, update affected leaves immediately
        // No async, no delay, reads see writes instantly
    }
}
```

## Implementation Steps

1. **[x] Add subtree size precomputation** — Implement `precompute_subtree_sizes()` and store in `BlockSearchTree`

2. **[x] Implement implicit child navigation** — Add `left_child()`, `right_child()` methods using position arithmetic instead of pointer derefs

3. **[x] Remove pointer fields from Node** — Change `Node::Internal` to only store `min_rhs`, remove `left`/`right` `NonNull` fields

4. **[ ] Implement `update_leaf()`** — Single-leaf min_key update with optional ancestor propagation

5. **[ ] Modify PMA rebalance to return affected range** — Return `RebalanceResult` indicating which blocks shifted

6. **[ ] Remove async indexing** — Delete channel, thread, delay constant, `request_reindex()`

7. **[ ] Wire incremental updates into insert** — After PMA operations, call `update_leaf()` for affected blocks

## Complexity Analysis

| Operation             | Current                 | After Changes        |
| --------------------- | ----------------------- | -------------------- |
| Insert (index update) | O(N) rebuild            | O(log N) leaf update |
| Get after insert      | 50ms wait               | Immediate            |
| Node memory           | 24+ bytes               | 8 bytes              |
| Search                | O(log N) pointer derefs | O(log N) arithmetic  |

## Open Questions

1. **Concurrency model**: With synchronous updates, should `BTreeMap` use internal mutability (`RefCell`/`RwLock`) for concurrent reads during writes, or require `&mut self` for all mutations?

2. **Parent pointers**: For `propagate_key_change()`, should we store parent indices, or recompute the path from root each time?

3. **Block size tuning**: Current formula is `slots = log₂(capacity)`. Should this be configurable for different workloads?
