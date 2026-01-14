//! # Cache-Oblivious B-Tree
//!
//! This module provides a cache-oblivious B-tree implementation using a packed memory array (PMA)
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

// mod binary_tree;
// mod packed_data;
mod btree_map;
mod cell;
mod packed_memory_array;

pub use btree_map::BTreeMap;
