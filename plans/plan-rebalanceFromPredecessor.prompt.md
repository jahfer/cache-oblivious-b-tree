# Plan: Include Predecessor in Rebalance Range

## Problem Statement

When inserting between two adjacent cells (e.g., inserting key 1 between cells with keys 0 and 2), the current algorithm:

1. Finds the predecessor (cell with largest key < insert_key)
2. Finds the first cell with key > insert_key
3. If no gap exists between them, calls `rebalance(cell_with_key_gt_insert_key)`
4. **Problem**: Rebalance spreads cells WITHIN its range rightward, creating gaps between cells in that range, but NOT before the first cell
5. After rebalance, there's still no gap between predecessor and the first cell > insert_key
6. Insert retries, finds no gap, rebalances again → excessive rebalances

## Root Cause

When we call `rebalance(cell_2)` for cells `[0, 2, 4, 6, ...]`:

- Cells `[2, 4, 6, ...]` are spread rightward
- Gaps are created BETWEEN 2 and 4, BETWEEN 4 and 6, etc.
- But cell 2 stays at its position (or moves right) - no gap is created BEFORE cell 2
- We need a gap between cell 0 (predecessor) and cell 2 (first cell > insert_key)

## Proposed Solution: Option A

**Include the predecessor in the rebalance range** by calling `rebalance(predecessor)` instead of `rebalance(cell_with_key_gt_insert_key)`.

When we rebalance starting from the predecessor:

- The predecessor moves rightward
- A gap is created at the predecessor's OLD position
- This gap is exactly where we need to insert the new key

### Why This Works

Before: `[..., pred(key=0), next(key=2), ...]` - no gap between pred and next

After `rebalance(pred)`:

- Cells `[0, 2, 4, ...]` spread rightward
- Cell 0 moves to a new position (e.g., position+1 or further)
- Gap created at cell 0's old position
- Layout: `[..., gap, pred(key=0), gap, next(key=2), ...]`

Now insert can use the gap that's before the (moved) predecessor, which maintains sorted order since the new key is > predecessor's key.

### Constraint: Cells Can Only Move Rightward

The plan must respect the invariant that cells can only move rightward during rebalance. This is required because:

- `AtomicPtr` readers that find a moved cell scan RIGHT to find the destination
- Moving cells leftward would break this recovery mechanism

Rebalancing from the predecessor still moves cells rightward, so this constraint is satisfied.

## Implementation Steps

### Step 1: Change rebalance call site when key > insert_key is found

Current code (btree_map.rs, around line 185):

```rust
// No empty cell available - need to rebalance to create space.
// Rebalance from this cell (the first cell with key >= insert_key)
drop(index);
let result = self.rebalance(cell_guard.inner, true);
```

Change to:

```rust
// No empty cell available - need to rebalance to create space.
// Rebalance from the PREDECESSOR (if exists) to create a gap at its old position.
// This gap will be between the predecessor and the cell with key > insert_key.
drop(index);
let rebalance_start = if let Some(ref pred) = predecessor_cell {
    pred.inner
} else {
    // No predecessor - rebalance from the cell with key > insert_key
    cell_guard.inner
};
let result = self.rebalance(rebalance_start, true);
```

### Step 2: Modify rebalance to place gaps at targeted locations

The current rebalance spreads cells evenly based on density thresholds, placing gaps arbitrarily between cells. Since we're already creating gaps to satisfy density requirements, we should **place them where they're needed** rather than randomly.

**Enhancement**: Add a `gap_after` parameter to hint where the first gap should be placed.

```rust
fn rebalance(
    &self,
    cell_ptr_start: *const Cell<K, V>,
    for_insertion: bool,
    gap_after: Option<*const Cell<K, V>>,  // NEW: hint for gap placement
) -> RebalanceResult
```

**Behavior**:

- When `gap_after` is `Some(ptr)`, ensure at least one gap is placed immediately after that cell
- Still respect density thresholds - don't distort balance
- If density requires N gaps, place the first one at the targeted location, distribute the rest evenly
- If `gap_after` is `None`, use current even distribution (backwards compatible)

**Why this is safe**:

- We're not adding extra gaps, just controlling WHERE existing gaps go
- Density invariants are still satisfied
- The total number of cells and gaps remains the same

**Call site change**:

```rust
// In insert(), when rebalancing for insertion:
let rebalance_start = predecessor_cell
    .as_ref()
    .map(|pred| pred.inner)
    .unwrap_or(cell_guard.inner);
let gap_hint = predecessor_cell.as_ref().map(|pred| pred.inner);
let result = self.rebalance(rebalance_start, true, gap_hint);
```

### Step 3: Update threshold in scaling test

Change `test_insert_scaling_is_sublinear` threshold from 20x back to 3x:

```rust
assert!(
    ratio < 3.0,
    "Insert time scaled too much with size: {:.2}x (expected <3x for sublinear behavior).",
    ...
);
```

## Code Changes Summary

```rust
// btree_map.rs, in the insert() function, around line 185

// BEFORE:
} else {
    // This cell's key > our key - we've found where to insert.
    if let Some(mut empty_cell) = first_empty_after_predecessor {
        // ... use the gap ...
    }

    // No empty cell available - need to rebalance to create space.
    drop(index);
    let result = self.rebalance(cell_guard.inner, true);  // <-- CHANGE THIS
    // ...
}

// AFTER:
} else {
    // This cell's key > our key - we've found where to insert.
    if let Some(mut empty_cell) = first_empty_after_predecessor {
        // ... use the gap ...
    }

    // No empty cell available - need to rebalance to create space.
    // Rebalance from predecessor to create gap at its old position,
    // which is exactly where we need to insert (between pred and this cell).
    drop(index);
    let rebalance_start = predecessor_cell
        .as_ref()
        .map(|pred| pred.inner)
        .unwrap_or(cell_guard.inner);
    let result = self.rebalance(rebalance_start, true);
    // ...
}
```

## Testing Strategy

1. **Run existing tests** - all 98 should pass
2. **Run scaling test with 3x threshold** - should pass after fix
3. **Add interleaved insert test**:

   ```rust
   #[test]
   fn test_interleaved_even_odd_inserts() {
       let mut tree: BTreeMap<usize, usize> = BTreeMap::new(200);

       // Insert even numbers
       for i in 0..100 {
           tree.insert(i * 2, i);
       }

       // Insert odd numbers between them
       for i in 0..100 {
           tree.insert(i * 2 + 1, i);
       }

       // All values should be retrievable
       for i in 0..200 {
           assert_eq!(tree.get(&i), Some(i / 2));
       }
   }
   ```

## Risks and Mitigations

| Risk                                                | Mitigation                                                                                    |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Infinite loop if rebalance doesn't move predecessor | Targeted gap placement guarantees a gap after predecessor; no reliance on random distribution |
| predecessor_cell is None when we need it            | Fall back to `cell_guard.inner` (original behavior) with `gap_after: None`                    |
| Targeted gap violates density threshold             | We only control WHERE gaps go, not how many; density invariants preserved                     |
| Gap hint points to cell outside rebalance range     | Validate `gap_after` is within `[cell_ptr_start, current_cell_ptr]`; ignore hint if invalid   |
| Performance regression from larger rebalance range  | Monitor; the extra cells in range should be minimal                                           |

## Success Criteria

- [ ] All 98 existing tests pass
- [ ] `test_insert_scaling_is_sublinear` passes with 3x threshold (down from 20x)
- [ ] No infinite loops on any insert pattern
- [ ] Interleaved insert pattern (evens then odds) completes efficiently
- [ ] New test `test_interleaved_even_odd_inserts` passes
