//! # Cache-Oblivious B-Tree
//!
//! This crate provides a cache-oblivious B-tree implementation using a packed memory array (PMA)
//! as the underlying storage structure. The design optimizes for cache efficiency without
//! requiring knowledge of the cache hierarchy parameters.
//!
//! ## Overview
//!
//! The data structure consists of three main components:
//!
//! - **[`BTreeMap`]**: The main map interface providing `get()` and `insert()` operations.
//! - **`Cell`**: Individual storage units within the PMA that hold key-value pairs with
//!   versioning and markers for lock-free concurrent access.
//! - **`PackedMemoryArray`**: The underlying "gapped array" storage that maintains
//!   density invariants and supports efficient rebalancing.
//!
//! ## Lock-Free Concurrency Model
//!
//! The implementation uses a marker-based protocol for lock-free operations:
//!
//! ```text
//! Cell State Transitions:
//! ┌─────────────────────────────────────────────────────────────┐
//! │                                                             │
//! │   ┌─────────┐                        ┌──────────────┐       │
//! │   │  Empty  │───── Insert ──────────>│  InsertCell  │       │
//! │   │  (v)    │                        │  (v, k, val) │       │
//! │   └────┬────┘                        └──────┬───────┘       │
//! │        │                                    │               │
//! │        │<──────── Commit ───────────────────┘               │
//! │        │          (v+1)                                     │
//! │        │                                                    │
//! │        │                             ┌─────────────┐        │
//! │        ├─────── Rebalance ──────────>│     Move    │        │
//! │        │                             │  (v, dest)  │        │
//! │        │                             └──────┬──────┘        │
//! │        │                                    │               │
//! │        │<──────── Complete ─────────────────┘               │
//! │        │          (v+1)                                     │
//! │        │                                                    │
//! │        │                             ┌─────────────┐        │
//! │        └─────── Delete ─────────────>│ DeleteCell  │        │
//! │                                      │   (v, k)    │        │
//! │                                      └─────────────┘        │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//!
//! Legend:
//!   v    = version number
//!   k    = key
//!   dest = destination index (for Move operations)
//! ```
//!
//! ### Marker Types
//!
//! - **`Empty(version)`**: Cell is available or contains committed data
//! - **`InsertCell(version, key, value)`**: Insert operation in progress
//! - **`DeleteCell(version, key)`**: Delete operation in progress
//! - **`Move(version, dest_index)`**: Cell being relocated during rebalance
//!
//! ### Version Validation
//!
//! Each cell has a version number that must match the marker's embedded version
//! for a read to be considered valid. This detects concurrent modifications:
//!
//! 1. Reader loads `cell.version` (e.g., 5)
//! 2. Reader loads `cell.marker` and extracts `marker.version()` (should be 5)
//! 3. If versions match, the data is consistent; if not, retry the read
//!
//! ## Packed Memory Array
//!
//! The PMA divides its capacity into three regions:
//!
//! ```text
//! ┌────────────┬──────────────────────────────┬────────────┐
//! │   Buffer   │        Active Range          │   Buffer   │
//! │   (1/4)    │           (1/2)              │   (1/4)    │
//! └────────────┴──────────────────────────────┴────────────┘
//! ```
//!
//! The buffer regions provide space for rebalancing operations without
//! reallocating the entire array.
//!
//! ## Example
//!
//! ```rust
//! use cache_oblivious_b_tree::BTreeMap;
//!
//! let mut map: BTreeMap<u32, String> = BTreeMap::new(100);
//! map.insert(1, "one".to_string());
//! map.insert(2, "two".to_string());
//!
//! assert_eq!(map.get(&1), Some("one".to_string()));
//! assert_eq!(map.get(&3), None);
//! ```

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod cache_oblivious;
pub use cache_oblivious::BTreeMap;

#[cfg(test)]
mod tests {
    use crate::BTreeMap;

    #[test]
    fn find_missing() {
        let tree = BTreeMap::<u8, String>::new(3);
        assert_eq!(tree.get(&4), None);
    }

    #[test]
    fn add_existing() {
        let mut tree = BTreeMap::<u8, String>::new(3);
        tree.insert(5, String::from("Test"));
        assert_eq!(tree.get(&5), Some(String::from("Test")));
        tree.insert(5, String::from("Double"));
        assert_eq!(tree.get(&5), Some(String::from("Double")));
    }

    #[test]
    fn add_ordered_values() {
        let mut tree = BTreeMap::<u8, String>::new(3);
        tree.insert(3, String::from("Hello"));
        tree.insert(8, String::from("World"));
        tree.insert(12, String::from("!"));

        assert_eq!(tree.get(&3), Some(String::from("Hello")));
        assert_eq!(tree.get(&8), Some(String::from("World")));
        assert_eq!(tree.get(&12), Some(String::from("!")));
    }

    #[test]
    fn add_unordered_values() {
        let mut tree = BTreeMap::<u8, String>::new(16);
        tree.insert(5, String::from("Hello"));
        tree.insert(3, String::from("World"));
        tree.insert(2, String::from("!"));

        assert_eq!(tree.get(&5), Some(String::from("Hello")));
        assert_eq!(tree.get(&4), None);
        assert_eq!(tree.get(&3), Some(String::from("World")));
        assert_eq!(tree.get(&2), Some(String::from("!")));
    }

    #[test]
    fn add_100_values() {
        let mut tree = BTreeMap::<u8, u8>::new(100);
        for i in 1..100u8 {
            tree.insert(i, i + 1);
        }

        assert_eq!(tree.get(&99), Some(100));
    }
}
