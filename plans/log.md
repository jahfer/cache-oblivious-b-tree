# Summary of work completed

## Step 1: Add subtree size precomputation (Jan 11, 2026)

### What was implemented

Added infrastructure for implicit vEB navigation by precomputing subtree sizes at each recursion level.

**New types added to `btree_map.rs`:**

- `SubtreeSizes` struct: Stores `upper_size` and `lower_size` for each vEB recursion level

**New functions:**

- `precompute_subtree_sizes(total_height: u8) -> Box<[SubtreeSizes]>`: Computes the upper and lower subtree node counts at each vEB recursion level. Returns O(log log N) entries.
- `compute_tree_height(node_count: usize) -> u8`: Helper to determine tree height from node count.

**Modified `BlockSearchTree` struct:**

- Added `subtree_sizes: Box<[SubtreeSizes]>` field
- Added `height: u8` field
- Updated `new()` to compute and store these values

### How it works

The vEB layout recursively splits a tree of height h into:

- An "upper" subtree of height ceil(h/2) containing the top portion
- Multiple "lower" subtrees of height floor(h/2) hanging off the upper leaves

For power-of-2 heights, the split is even. For other heights, the lower height is rounded down to the nearest power of 2 to maintain cache-oblivious properties.

Example for height 8:

- Level 0: upper_size=15, lower_size=15 (h=8 → 4+4)
- Level 1: upper_size=3, lower_size=3 (h=4 → 2+2)
- Level 2: upper_size=1, lower_size=1 (h=2 → 1+1)

### Tests added

9 unit tests in `cache_oblivious::btree_map::tests`:

- `test_compute_tree_height`: Verifies height computation for various node counts
- `test_precompute_subtree_sizes_*`: Tests empty, height 1-4, and height 8 cases
- `test_subtree_sizes_count_matches_expected`: Confirms O(log log N) recursion depth
- `test_subtree_sizes_total_matches_tree`: Validates that upper + lower sizes reconstruct full tree

### Next steps

Step 2 will use these precomputed sizes to implement `left_child()` and `right_child()` methods that compute child positions arithmetically instead of following pointers.

## Step 2: Implement implicit child navigation (Jan 11, 2026)

### What was implemented

Added `VebNavigator` struct and methods to compute child positions using arithmetic instead of pointer dereferencing.

**New types added to `btree_map.rs`:**

- `VebNavigator` struct: Tracks navigation state within the vEB-layout tree, including:
  - `position`: Current position in the flattened node array
  - `subtree_base`: Base offset of the current vEB subtree
  - `veb_depth`: Current depth within the vEB recursion (index into subtree_sizes)
  - `local_position`: Position within the current upper subtree
  - `current_subtree_size`: Size of the current subtree (for bounds checking)

**New methods on `VebNavigator`:**

- `at_root(tree_size: usize) -> Self`: Creates a navigator starting at the root
- `left_child(&self, subtree_sizes: &[SubtreeSizes]) -> VebNavigator`: Computes left child position
- `right_child(&self, subtree_sizes: &[SubtreeSizes]) -> VebNavigator`: Computes right child position
- `is_valid(&self) -> bool`: Checks if the current position is within subtree bounds

**New methods on `BlockSearchTree`:**

- `root_navigator(&self) -> VebNavigator`: Creates a navigator at the root
- `node_at(&self, nav: &VebNavigator) -> &Node`: Gets the node at a navigator position
- `left_child(&self, nav: &VebNavigator) -> VebNavigator`: Wrapper for navigation
- `right_child(&self, nav: &VebNavigator) -> VebNavigator`: Wrapper for navigation

### How it works

The `VebNavigator::child()` method handles two cases:

1. **Within upper subtree**: Uses standard binary tree indexing (`2*p+1` for left, `2*p+2` for right)
2. **Transition to lower subtree**: When `child_local >= upper_size`, computes:
   - `leaf_index = child_local - upper_size` (which lower subtree to enter)
   - `lower_subtree_base = subtree_base + upper_size + leaf_index * lower_size`
   - Increments `veb_depth` and resets `local_position` to 0

The `is_valid()` method checks if `local_position < current_subtree_size`, which correctly identifies when children would be out of bounds (e.g., leaves have no valid children).

### Tests added

8 new unit tests in `cache_oblivious::btree_map::tests`:

- `test_veb_navigator_at_root`: Verifies initial navigator state
- `test_veb_navigator_height_2`: Tests navigation in a 3-node tree
- `test_veb_navigator_height_3`: Tests navigation with upper/lower subtree transitions
- `test_veb_navigator_height_4`: Tests multi-level recursion
- `test_veb_navigator_all_positions_reachable`: BFS traversal reaches all N positions
- `test_veb_navigator_no_duplicate_positions`: Left and right children are always distinct
- `test_veb_navigator_leaf_count`: Correct number of leaves (2^(h-1)) detected
- `test_veb_navigator_is_valid`: Validates bounds checking for leaf children

### Next steps

Step 3 will remove the `left` and `right` `NonNull` pointer fields from `Node::Internal`, relying entirely on the implicit navigation implemented in this step.

## Step 3: Remove pointer fields from Node (Jan 11, 2026)

### What was implemented

Removed explicit pointer fields from `Node::Internal` and updated all tree construction and search logic to use the implicit `VebNavigator`-based navigation.

**Modified `Node` enum:**

Before:

```rust
enum Node<K, V> {
    Leaf(Key<K>, Block<K, V>),
    Internal {
        min_rhs: Key<K>,
        left: MaybeUninit<NonNull<UnsafeCell<Node<K, V>>>>,
        right: MaybeUninit<NonNull<UnsafeCell<Node<K, V>>>>,
    },
}
```

After:

```rust
enum Node<K, V> {
    Leaf(Key<K>, Block<K, V>),
    Internal {
        min_rhs: Key<K>,
    },
}
```

**Modified `BlockSearchTree` struct:**

- Changed `nodes` field from `Box<[UnsafeCell<Node<K, V>>]>` to `Box<[Node<K, V>]>` (no more interior mutability needed)
- Removed `allocate()`, `initialize_nodes()`, `assign_node_values()`, and `split_tree_memory()` helper functions
- Added `finalize_leaves_veb()`, `collect_leaf_positions_inorder()`, and `inorder_collect_leaves()` for navigator-based tree construction
- Updated `find()` to use `VebNavigator` loop instead of calling `search_to_block()` on nodes

**Removed from `Node`:**

- `search()` method (used pointer fields)
- `search_to_block()` method (used pointer fields)

**Removed from `SearchResult`:**

- `Internal` variant (no longer needed since navigation is external)

**Removed imports:**

- `std::cell::UnsafeCell`
- `std::mem::MaybeUninit`
- `std::ptr::NonNull`

### How it works

Tree construction now uses a two-phase approach:

1. Allocate all nodes as `Internal { min_rhs: Key::Supremum }`
2. Use `VebNavigator` to traverse in-order and identify leaf positions, then convert those nodes to `Leaf` variants with their corresponding PMA block references

Search now happens entirely in `BlockSearchTree::find()`:

1. Start with `root_navigator()`
2. Loop: get node at current position, if `Leaf` return result, if `Internal` compute next navigator position based on `min_rhs` comparison
3. Navigation uses purely arithmetic operations via `VebNavigator::left_child()` / `right_child()`

### Memory savings

The `Node::Internal` variant now only stores `min_rhs` (a `Key<K>`), eliminating the 16 bytes previously used for `left` and `right` pointer fields. For trees with many internal nodes, this represents significant memory reduction.

### Tests added

3 new unit tests in `cache_oblivious::btree_map::tests`:

- `test_node_internal_has_no_pointers`: Verifies the new `Node::Internal` structure compiles and works without pointer fields
- `test_leaf_positions_collected_in_order`: Validates that leaf collection via navigator returns the correct count of unique, valid positions
- `test_navigator_based_tree_traversal_matches_leaf_count`: Confirms the navigator correctly finds 2^(h-1) leaves for trees of various heights

### Next steps

Step 4 will add leaf position tracking (`first_leaf_index`) to enable O(1) leaf-to-position mapping for incremental updates.

## Step 4: Implement update_leaf() (Jan 11, 2026)

### What was implemented

Added infrastructure for incremental leaf updates, enabling O(log N) index updates instead of full O(N) rebuilds.

**New field in `BlockSearchTree`:**

- `leaf_positions: Box<[usize]>`: Precomputed mapping from leaf index (0..num_leaves) to node array position. Computed once at tree construction via `collect_leaf_positions_inorder()`.

**New methods on `BlockSearchTree`:**

- `leaf_count(&self) -> usize`: Returns the number of leaves in the tree
- `leaf_index_to_position(&self, leaf_index: usize) -> Option<usize>`: O(1) lookup of node array position from leaf index
- `update_leaf(&mut self, leaf_index: usize, new_min_key: Key<K>) -> bool`: Updates a single leaf's min_key and propagates changes to ancestors
- `propagate_key_change(&mut self, leaf_pos: usize, new_key: &Key<K>)`: Updates internal node `min_rhs` values when a leaf's key changes

### How it works

**Leaf position mapping:**
The `leaf_positions` array is computed at tree construction time using the existing `collect_leaf_positions_inorder()` function. This provides O(1) mapping from a logical leaf index (e.g., which PMA block) to the physical position in the vEB-layout node array.

**Leaf updates:**
`update_leaf()` takes a leaf index and new key, looks up the node position via `leaf_positions`, and updates the `Node::Leaf`'s `min_key` field directly. Returns `false` for out-of-bounds indices.

**Ancestor propagation:**
When a leaf's min_key changes, internal nodes may need their `min_rhs` updated. An internal node's `min_rhs` stores the minimum key in its right subtree. `propagate_key_change()`:

1. Walks from root to the updated leaf, recording which direction (left/right) was taken at each step
2. For any node where we went right AND then only left to reach the leaf, that node's `min_rhs` is updated (since the leaf is the minimum of its right subtree)

### Tests added

8 new unit tests in `cache_oblivious::btree_map::tests`:

- `test_leaf_positions_stored_in_tree`: Verifies leaf_positions are computed correctly for various tree heights
- `test_leaf_index_to_position_mapping`: Tests O(1) lookup and out-of-bounds handling
- `test_leaf_count`: Validates leaf_count() returns 2^(h-1)
- `test_update_leaf_basic`: Tests updating leaves with new keys
- `test_update_leaf_out_of_bounds`: Confirms false return for invalid indices
- `test_update_leaf_with_supremum`: Tests updating to Key::Supremum (empty block)
- `test_propagate_key_change_updates_ancestors`: Tests ancestor min_rhs updates
- `test_update_all_leaves_sequentially`: Tests sequential updates to all leaves

### Next steps

Step 5 will modify the PMA rebalance operation to return a `RebalanceResult` indicating which blocks were affected, enabling the caller to selectively update only those leaves.

## Step 5: Modify PMA rebalance to return affected range (Jan 11, 2026)

### What was implemented

Added infrastructure for tracking which blocks are affected during rebalancing, enabling incremental index updates.

**New type added to `btree_map.rs`:**

- `RebalanceResult` struct: Contains `affected_blocks: Range<usize>` indicating which leaf indices need their min_key refreshed after a rebalance

**New methods on `RebalanceResult`:**

- `new(affected_blocks: Range<usize>) -> Self`: Creates a new result with the given affected range
- `none() -> Self`: Creates a result indicating no blocks were affected (empty range)
- `has_affected_blocks(&self) -> bool`: Returns true if any blocks were affected

**New methods on `BTreeMap`:**

- `cell_ptr_to_block_index(&self, cell_ptr: *const Cell<K, V>) -> Option<usize>`: Computes block index from cell pointer using `cell_offset / slot_size`
- `compute_affected_blocks(&self, dest_ptr, source_end_ptr) -> RebalanceResult`: Computes the range of blocks affected by a rebalance

**Modified `BTreeMap::rebalance()`:**

- Changed return type from `()` to `RebalanceResult`
- Tracks the source range boundaries (`last_cell_ptr`) during cell scanning
- After moving cells, computes and returns the affected block range

### How it works

**Block index computation:**
Each block (leaf in the search tree) corresponds to `slot_size` cells in the PMA, where `slot_size = log2(requested_capacity)`. A cell at offset `n` belongs to block `n / slot_size`.

**Affected range tracking:**
During rebalance, cells move from a source region toward a destination region. The function tracks:

1. The destination pointer (moves backward as cells are placed)
2. The source end pointer (the last cell examined during density scanning)

The affected blocks are those spanning from the destination block to the source block (inclusive).

### Tests added

6 new unit tests in `cache_oblivious::btree_map::tests`:

- `test_rebalance_result_new`: Verifies construction with a specific range
- `test_rebalance_result_none`: Tests empty/none result creation
- `test_rebalance_result_has_affected_blocks`: Tests empty vs non-empty range detection
- `test_rebalance_result_equality`: Tests PartialEq implementation
- `test_rebalance_result_debug`: Tests Debug formatting
- `test_rebalance_result_clone`: Tests Clone implementation

### Next steps

Step 6 will remove the async indexing infrastructure (channel, thread, delay constant, `request_reindex()`), preparing for synchronous incremental updates in Step 7.

## Step 6: Remove async indexing (Jan 11, 2026)

### What was implemented

Removed the entire background thread infrastructure for index updates, transitioning to a synchronous model that will enable immediate consistency in Step 7.

**Removed imports:**

- `std::sync::atomic::AtomicBool` (no longer needed for `index_updating` flag)
- `std::sync::mpsc::{channel, Receiver, Sender}` (no channel communication needed)
- `std::sync::Weak` (no weak references to PMA needed)
- `std::thread` (no background thread)
- `std::time` (no sleep delay)

**Removed constant:**

- `INDEX_UPDATE_DELAY: time::Duration` (the 50ms delay between index updates)

**Removed fields from `BTreeMap` struct:**

- `tx: Sender<Weak<PackedMemoryArray<Cell<K, V>>>>` (channel sender for signaling reindex)
- `index_updating: Arc<AtomicBool>` (flag indicating if background thread is updating)

**Removed methods:**

- `request_reindex(&self)`: Was called after insert() to signal the background thread
- `start_indexing_thread(...)`: Spawned and managed the background indexing thread

**Modified `BTreeMap::new()`:**

Simplified constructor now just creates data and index without spawning any threads:

```rust
pub fn new(capacity: u32) -> BTreeMap<K, V> {
    let packed_cells = PackedMemoryArray::with_capacity(capacity);
    let data = Arc::new(packed_cells);
    let raw_index = Self::generate_index(Arc::clone(&data));
    let index = Arc::new(RwLock::new(raw_index));
    BTreeMap { index, data }
}
```

**Modified `insert()` method:**

Removed the call to `request_reindex()` at the end. Added a TODO comment indicating that Step 7 will wire incremental updates here using `update_leaf()`.

### How it works

Previously, the indexing model was:

1. `insert()` would modify the PMA and call `request_reindex()`
2. `request_reindex()` would send a weak reference through a channel (if not already updating)
3. A background thread would receive the reference, wait 50ms for debouncing, rebuild the entire index
4. Reads during this window would see stale data

Now the model is:

1. `insert()` modifies the PMA (no background work)
2. Index remains unchanged (temporarily stale until Step 7 wires `update_leaf()`)
3. No threads, no channels, no delays

### Benefits of this change

- **Simplified architecture**: No thread synchronization complexity
- **Reduced memory**: No channel buffers, atomic flags, or thread stacks
- **Eliminated 50ms latency**: Reads no longer need to wait for background updates
- **Foundation for Step 7**: Clean slate for wiring incremental `update_leaf()` calls

### Tests added

4 new unit tests in `cache_oblivious::btree_map::tests`:

- `test_btreemap_struct_has_no_async_fields`: Verifies BTreeMap compiles without async-related type fields
- `test_btreemap_new_returns_immediately`: Confirms construction completes in <10ms (no thread spawn delay)
- `test_btreemap_no_index_update_delay`: Verifies multiple creations complete in <50ms (no INDEX_UPDATE_DELAY)
- `test_generate_index_is_synchronous`: Confirms generate_index produces a valid BlockIndex synchronously

### Known failing tests

Three integration tests in `src/lib.rs` now fail (`add_ordered_values`, `add_unordered_values`, `add_100_values`). These tests relied on the async indexing to rebuild the entire index after a 50ms sleep. Since the async infrastructure is removed but Step 7 (incremental updates) is not yet implemented, the index is never updated after inserts.

**This is expected behavior for Step 6.** These tests will pass again after Step 7 wires `update_leaf()` into the `insert()` method to provide immediate, synchronous index updates.

### Next steps

Step 7 will wire incremental updates into `insert()` by calling `update_leaf()` for affected blocks after PMA operations, completing the transition from O(N) rebuilds to O(log N) incremental updates.

## Step 7: Wire incremental updates into insert (Jan 11, 2026)

### What was implemented

Completed the transition to synchronous incremental index updates by wiring the `update_leaf()` method into the `insert()` flow, eliminating the need for full index rebuilds.

**New methods on `BlockIndex`:**

- `update_leaf(&mut self, leaf_index: usize, new_min_key: Key<K>) -> bool`: Delegates to `BlockSearchTree::update_leaf()` for updating a single leaf's min_key
- `slot_size(&self) -> usize`: Returns the number of cells per block (log2 of requested capacity)
- `compute_block_min_key(&self, block_index: usize) -> Key<K>`: Computes the minimum key for a given block by scanning its cells

**New method on `BTreeMap`:**

- `update_index_for_affected_blocks(&mut self, affected_blocks: &[Range<usize>], inserted_cell_ptr: Option<*const Cell<K, V>>)`: Collects all affected block indices from rebalance operations and the insertion point, then updates the index for each

**Modified `BTreeMap::insert()`:**

The insert method was refactored to:

1. Track `RebalanceResult` from each rebalance call during the loop
2. Track the cell pointer where the value was actually inserted
3. Scope the read lock on the index to just the search/insert phase
4. After the insert completes, call `update_index_for_affected_blocks()` with a write lock to update affected leaves

### How it works

**Affected block collection:**
During insert, rebalance operations return `RebalanceResult` containing the range of blocks that were affected. These ranges are collected into a vector. Additionally, the block containing the inserted cell is tracked.

**Index update flow:**
After the insert loop completes:

1. The read lock on `index` is dropped (via scoping)
2. All affected block indices are merged into a `HashSet` to deduplicate
3. A write lock is acquired on `index`
4. For each affected block, `compute_block_min_key()` scans the cells to find the current minimum
5. `update_leaf()` updates the leaf's min_key and propagates changes to ancestors

**Lock ordering:**
The read lock is explicitly scoped with `{ ... }` so it's dropped before acquiring the write lock. This prevents deadlocks and allows the index to be updated after the PMA modifications are complete.

### Benefits of this change

- **Immediate consistency**: No 50ms delay between inserts and reads
- **O(log N) updates**: Only affected leaves are updated, not the entire index
- **No background threads**: All work happens synchronously in the insert call
- **Memory efficiency**: No channel buffers or atomic flags needed

### Tests added

10 new unit tests in `cache_oblivious::btree_map::tests`:

- `test_block_index_update_leaf_delegates_to_tree`: Verifies BlockIndex properly delegates to BlockSearchTree
- `test_block_index_update_leaf_out_of_bounds`: Tests that out-of-bounds indices return false
- `test_block_index_slot_size`: Verifies slot_size computation
- `test_block_index_compute_block_min_key_empty_block`: Tests Key::Supremum for empty blocks
- `test_block_index_compute_block_min_key_out_of_bounds`: Tests bounds checking
- `test_update_index_for_affected_blocks_empty`: Tests no-op behavior with empty inputs
- `test_insert_updates_index_synchronously`: Confirms immediate value retrieval after insert
- `test_insert_multiple_values_index_stays_consistent`: Tests multiple sequential inserts
- `test_insert_updates_index_without_delay`: Verifies inserts complete in <50ms

### Previously failing tests now pass

The three integration tests that were failing after Step 6 now pass:

- `add_ordered_values`
- `add_unordered_values`
- `add_100_values`

These tests no longer need their `thread::sleep(50ms)` calls (though they still have them), because the index is updated synchronously during insert.

### Completion notes

This completes the implementation plan for implicit vEB navigation and incremental index updates. All 7 steps are now complete:

1. ✅ Subtree size precomputation
2. ✅ Implicit child navigation
3. ✅ Pointer-free Node structure
4. ✅ Single-leaf update method
5. ✅ Rebalance result tracking
6. ✅ Async infrastructure removal
7. ✅ Incremental update wiring

The cache-oblivious B-tree now has:

- O(log N) search using implicit vEB navigation (no pointer derefs)
- O(log N) incremental index updates (no full rebuilds)
- Immediate consistency (no 50ms async delays)
- Reduced memory footprint (no pointer fields in internal nodes)

### Cache-oblivious properties preserved

The implementation maintains cache-oblivious properties through:

1. **VebNavigator for tree traversal**: All tree navigation uses the `VebNavigator` struct with `left_child()` and `right_child()` methods that compute positions arithmetically using the precomputed `subtree_sizes` table. This avoids pointer chasing and maintains vEB layout benefits.

2. **Sorted block updates**: The `update_index_for_affected_blocks()` method sorts affected block indices before updating, ensuring sequential access to the `leaf_positions` array and underlying PMA. This maximizes cache locality.

3. **Sequential cell access**: The `compute_block_min_key()` method accesses cells sequentially within a contiguous slice of the PMA, which is optimal for cache line utilization.

4. **Precomputed leaf positions**: The `leaf_positions` array provides O(1) mapping from logical leaf index to vEB node position, avoiding repeated tree traversals.

5. **Propagation via vEB navigation**: The `propagate_key_change()` method uses `VebNavigator` to walk from root to leaf, following the vEB layout structure for cache-efficient ancestor updates.
