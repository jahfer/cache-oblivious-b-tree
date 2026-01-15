use std::cell::OnceCell;
use std::cell::UnsafeCell;
use std::cmp::{Ord, Ordering};
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicIsize, AtomicU16, AtomicU8, Ordering as AtomicOrdering};

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy)]
pub enum Key<T: Ord> {
    Infimum,
    Value(T),
    Supremum,
}

impl<'a, T: Ord> Key<T> {
    pub fn as_ref(&self) -> Key<&T> {
        match *self {
            Key::Value(ref v) => Key::Value(v),
            Key::Infimum => Key::Infimum,
            Key::Supremum => Key::Supremum,
        }
    }

    pub fn unwrap(self) -> T {
        match self {
            Key::Value(val) => val,
            _ => panic!("Called Key::unwrap() on an infinite value"),
        }
    }

    #[allow(dead_code)]
    pub fn is_infimum(&self) -> bool {
        match self {
            Key::Infimum => true,
            _ => false,
        }
    }

    pub fn is_supremum(&self) -> bool {
        match self {
            Key::Supremum => true,
            _ => false,
        }
    }
}

impl<T: Ord> Ord for Key<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (x, y) if x == y => Ordering::Equal,
            (Key::Infimum, _) | (_, Key::Supremum) => Ordering::Less,
            (Key::Supremum, _) | (_, Key::Infimum) => Ordering::Greater,
            (Key::Value(a), Key::Value(b)) => a.cmp(b),
        }
    }
}

impl<'a, T: Ord> From<&'a Key<T>> for Key<&'a T> {
    fn from(k: &'a Key<T>) -> Key<&'a T> {
        k.as_ref()
    }
}

/// Marker indicating the current operation state of a cell.
///
/// Markers are used for lock-free synchronization: before modifying a cell,
/// a thread sets a marker indicating the operation in progress. Other threads
/// can observe this marker to coordinate access.
///
/// # State Transitions
///
/// - `Empty` → `Inserting` → `Empty`: Insert operation
/// - `Empty` → `Move` → `Empty`: Rebalance operation
/// - `Empty` → `Deleting` → `Empty`: Delete operation
///
/// The version number is stored separately in the Cell's `version` field.
/// The `move_dest` field stores the destination index for Move operations.
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MarkerState {
    /// Cell is idle and available for operations.
    /// Contains committed data if key/value are `Some`, otherwise empty.
    Empty = 0,
    /// Cell's data is being moved during rebalance.
    /// The destination index is stored in the Cell's `move_dest` field.
    Move = 1,
    /// An insert operation is in progress.
    Inserting = 2,
    /// A delete operation is in progress.
    Deleting = 3,
}

impl MarkerState {
    /// Converts a u8 to MarkerState. Returns Empty for invalid values.
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => MarkerState::Empty,
            1 => MarkerState::Move,
            2 => MarkerState::Inserting,
            3 => MarkerState::Deleting,
            _ => MarkerState::Empty,
        }
    }
}

/// Legacy Marker enum for backwards compatibility with existing code.
/// This is now derived from the Cell's inline atomic fields.
///
/// Note: This enum no longer has type parameters since the marker state
/// is now stored inline in the Cell and doesn't need to carry type information.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Marker {
    /// Cell is idle and available for operations.
    Empty(u16),
    /// Cell's data is being moved to `dest_index` during rebalance.
    Move(u16, isize),
    /// An insert operation is in progress (key/value stored in cell fields).
    InsertCell(u16),
    /// A delete operation is in progress (key stored in cell fields).
    DeleteCell(u16),
}

impl Marker {
    /// Returns a reference to the version number embedded in this marker.
    pub fn version(&self) -> &u16 {
        match self {
            Marker::Empty(v)
            | Marker::Move(v, _)
            | Marker::InsertCell(v)
            | Marker::DeleteCell(v) => v,
        }
    }

    /// Returns the MarkerState for this marker.
    pub fn state(&self) -> MarkerState {
        match self {
            Marker::Empty(_) => MarkerState::Empty,
            Marker::Move(_, _) => MarkerState::Move,
            Marker::InsertCell(_) => MarkerState::Inserting,
            Marker::DeleteCell(_) => MarkerState::Deleting,
        }
    }

    /// Returns true if this is an Empty or Move marker (safe to overwrite during rebalance).
    pub fn is_idle_or_move(&self) -> bool {
        matches!(self, Marker::Empty(_) | Marker::Move(_, _))
    }
}

/// A single cell in the packed memory array.
///
/// Each cell stores an optional key-value pair along with synchronization
/// primitives for lock-free concurrent access:
///
/// - `version`: Monotonically increasing counter, incremented on each modification
/// - `marker_state`: The current operation state (Empty, Move, Inserting, Deleting)
/// - `move_dest`: Destination index for Move operations (only valid when marker_state == Move)
/// - `key`/`value`: The actual data stored in `UnsafeCell` for interior mutability
///
/// # Thread Safety
///
/// `Cell` implements `Send` and `Sync` because:
/// - `version`, `marker_state`, and `move_dest` are atomics with proper operations
/// - `key`/`value` access is guarded by version validation and marker checks
///
/// The lock-free protocol ensures that concurrent readers and writers
/// can operate safely without data races:
/// 1. Writer sets marker_state to indicate operation in progress
/// 2. Writer modifies key/value
/// 3. Writer increments version and updates marker_state to `Empty`
/// 4. Readers validate version before and after reading to detect concurrent modifications
pub struct Cell<K: Clone, V: Clone> {
    pub version: AtomicU16,
    pub marker_state: AtomicU8,
    pub move_dest: AtomicIsize,
    pub key: UnsafeCell<Option<K>>,
    pub value: UnsafeCell<Option<V>>,
}

// SAFETY: Cell is Send because all its fields can be safely transferred between threads:
// - AtomicU16, AtomicU8, and AtomicIsize are Send
// - UnsafeCell contents are protected by the version/marker protocol
unsafe impl<K: Clone + Send, V: Clone + Send> Send for Cell<K, V> {}

// SAFETY: Cell is Sync because concurrent access is safe through:
// - Atomic operations on version, marker_state, and move_dest
// - Version validation preventing observation of partially-written data
unsafe impl<K: Clone + Sync, V: Clone + Sync> Sync for Cell<K, V> {}

impl<K: Clone, V: Clone> Cell<K, V> {
    /// Creates a new empty cell with version 1 and Empty marker state.
    pub fn new() -> Cell<K, V> {
        Cell {
            version: AtomicU16::new(1),
            marker_state: AtomicU8::new(MarkerState::Empty as u8),
            move_dest: AtomicIsize::new(0),
            key: UnsafeCell::new(None),
            value: UnsafeCell::new(None),
        }
    }

    /// Loads the current marker state.
    #[inline]
    pub fn load_marker_state(&self, ordering: AtomicOrdering) -> MarkerState {
        MarkerState::from_u8(self.marker_state.load(ordering))
    }

    /// Atomically sets the marker state, returning the previous state.
    #[inline]
    pub fn swap_marker_state(
        &self,
        new_state: MarkerState,
        ordering: AtomicOrdering,
    ) -> MarkerState {
        MarkerState::from_u8(self.marker_state.swap(new_state as u8, ordering))
    }

    /// Atomically compares and exchanges the marker state.
    /// Returns Ok(old_state) on success, Err(current_state) on failure.
    #[inline]
    pub fn compare_exchange_marker_state(
        &self,
        expected: MarkerState,
        new: MarkerState,
        success: AtomicOrdering,
        failure: AtomicOrdering,
    ) -> Result<MarkerState, MarkerState> {
        self.marker_state
            .compare_exchange(expected as u8, new as u8, success, failure)
            .map(MarkerState::from_u8)
            .map_err(MarkerState::from_u8)
    }

    /// Reconstructs a Marker enum from the cell's atomic fields.
    /// This is for backwards compatibility with code expecting the Marker enum.
    pub fn load_marker(&self, ordering: AtomicOrdering) -> Marker {
        let version = self.version.load(ordering);
        let state = self.load_marker_state(ordering);
        match state {
            MarkerState::Empty => Marker::Empty(version),
            MarkerState::Move => Marker::Move(version, self.move_dest.load(ordering)),
            MarkerState::Inserting => Marker::InsertCell(version),
            MarkerState::Deleting => Marker::DeleteCell(version),
        }
    }
}

impl<K, V> Default for Cell<K, V>
where
    K: Clone,
    V: Clone,
{
    /// Creates a default empty cell with version 1 and Empty marker state.
    fn default() -> Self {
        Cell::new()
    }
}

impl<K: Debug + Clone, V: Debug + Clone> Debug for Cell<K, V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let version = self.version.load(AtomicOrdering::Acquire);
        let marker_state = self.load_marker_state(AtomicOrdering::Acquire);
        let move_dest = self.move_dest.load(AtomicOrdering::Acquire);
        // SAFETY: Debug formatting only reads data; we accept potential inconsistency
        // as this is for debugging purposes only, not for correctness-critical reads
        let key = unsafe { &*self.key.get() };
        // SAFETY: Same as above - debug read only
        let value = unsafe { &*self.value.get() };

        let mut dbg_struct = formatter.debug_struct("Cell");

        dbg_struct
            .field("version", &version)
            .field("marker_state", &marker_state)
            .field("move_dest", &move_dest)
            .field("key", key)
            .field("value", value);

        dbg_struct.finish()
    }
}

/// Cached data from a cell, containing the key, value, and marker at read time.
///
/// This struct is returned by [`CellGuard::cache()`] and represents a consistent
/// snapshot of the cell's contents at a point in time.
#[derive(Debug, Copy, Clone)]
pub struct CellData<K: Clone, V: Clone> {
    pub key: K,
    pub value: V,
    pub marker: Marker,
}

/// A guard that provides safe access to a cell's data.
///
/// `CellGuard` captures a consistent snapshot of a cell's state at creation time,
/// including the version number and marker state. It provides lazy caching of
/// the cell's key/value data with version validation on access.
///
/// # Thread Safety
///
/// While `CellGuard` itself is not `Send` or `Sync`, it safely guards against
/// concurrent modifications by validating version consistency before returning data.
pub struct CellGuard<'a, K: 'a + Clone, V: 'a + Clone> {
    pub inner: &'a Cell<K, V>,
    pub cache_version: u16,
    pub is_filled: bool,
    cache_data: OnceCell<Option<CellData<K, V>>>,
    cache_marker_state: MarkerState,
    cache_move_dest: isize,
    _phantom: PhantomData<&'a Cell<K, V>>,
}

impl<K: Clone, V: Clone> CellGuard<'_, K, V> {
    /// Returns `true` if this cell contains no key-value data.
    pub fn is_empty(&self) -> bool {
        !self.is_filled
    }

    /// Returns the cached marker state.
    pub fn marker_state(&self) -> MarkerState {
        self.cache_marker_state
    }

    /// Returns the cached move destination (only valid if marker_state is Move).
    pub fn move_dest(&self) -> isize {
        self.cache_move_dest
    }

    /// Retrieves and caches the cell's data with version validation.
    ///
    /// On first access, this method loads the cell's key, value, and marker,
    /// validating that the version hasn't changed since the guard was created.
    /// Subsequent calls return the cached data without re-validation.
    ///
    /// # Errors
    ///
    /// Returns [`CellReadError`] if the cell's version changed since the guard
    /// was created, indicating concurrent modification.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Ok(Some(data)) = guard.cache() {
    ///     println!("Key: {:?}, Value: {:?}", data.key, data.value);
    /// }
    /// ```
    #[must_use = "this returns a Result that should be checked for errors"]
    pub fn cache(&self) -> Result<&Option<CellData<K, V>>, CellReadError> {
        // Manual implementation of get_or_try_init for stable Rust
        if let Some(cached) = self.cache_data.get() {
            return Ok(cached);
        }

        let version = self.inner.version.load(AtomicOrdering::SeqCst);

        // Check version consistency BEFORE cloning any data
        if version != self.cache_version {
            return Err(CellReadError {});
        }

        // Only clone after validation passes
        // SAFETY: We have validated version consistency, so the cell data is consistent.
        // The UnsafeCell access is safe because:
        // - We only read (clone) the data, never write
        // - Version validation ensures no concurrent write is in progress
        let key = unsafe { (*self.inner.key.get()).clone() };
        let result = if let Some(k) = key {
            // SAFETY: Same reasoning as above for key - validated consistent state
            let value = unsafe { (*self.inner.value.get()).clone() };
            // Reconstruct marker from cached state
            let marker = match self.cache_marker_state {
                MarkerState::Empty => Marker::Empty(self.cache_version),
                MarkerState::Move => Marker::Move(self.cache_version, self.cache_move_dest),
                MarkerState::Inserting => Marker::InsertCell(self.cache_version),
                MarkerState::Deleting => Marker::DeleteCell(self.cache_version),
            };
            Some(CellData {
                key: k,
                value: value.unwrap(),
                marker,
            })
        } else {
            None
        };

        // set() will fail if already set (race condition), but get() will return the value
        let _ = self.cache_data.set(result);
        Ok(self.cache_data.get().unwrap())
    }

    /// Atomically updates the cell's marker state using compare-and-swap.
    ///
    /// This operation will succeed only if the marker state hasn't been modified
    /// since the guard was created.
    ///
    /// # Arguments
    /// * `new_state` - The new marker state to set
    /// * `new_move_dest` - The move destination (only used if new_state is Move)
    ///
    /// # Errors
    ///
    /// Returns [`CellWriteError`] if the CAS operation fails due to concurrent
    /// modification. The caller should typically retry the entire operation.
    #[must_use = "this returns a Result that should be checked for errors"]
    pub fn update_marker_state(
        &mut self,
        new_state: MarkerState,
        new_move_dest: isize,
    ) -> Result<(), Box<dyn Error>> {
        // Store move_dest BEFORE the CAS on marker_state.
        // This ensures readers who see the Move state will always find a valid
        // move_dest value - they may see a stale value if CAS fails, but that's
        // harmless since the marker state won't indicate Move.
        if new_state == MarkerState::Move {
            self.inner
                .move_dest
                .store(new_move_dest, AtomicOrdering::SeqCst);
        }

        let result = self.inner.compare_exchange_marker_state(
            self.cache_marker_state,
            new_state,
            AtomicOrdering::SeqCst,
            AtomicOrdering::SeqCst,
        );

        if result.is_err() {
            return Err(Box::new(CellWriteError {}));
        }

        self.cache_marker_state = new_state;
        self.cache_move_dest = new_move_dest;
        Ok(())
    }
}

/// Errors that can occur during cell operations.
///
/// This enum provides typed errors for cell read and write failures,
/// enabling callers to distinguish between different failure modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellError {
    /// A read operation failed due to concurrent modification.
    ///
    /// This occurs when the cell's version number doesn't match the
    /// marker's embedded version, indicating another thread modified
    /// the cell between loading the version and reading the data.
    VersionMismatch,
    /// A write operation failed due to a CAS (compare-and-swap) failure.
    ///
    /// This occurs when the marker pointer changed between reading it
    /// and attempting to update it, indicating concurrent modification.
    CasFailed,
}

impl Display for CellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellError::VersionMismatch => {
                write!(
                    f,
                    "Cell read failed: version mismatch detected (concurrent modification)"
                )
            }
            CellError::CasFailed => {
                write!(
                    f,
                    "Cell write failed: CAS operation failed (concurrent modification)"
                )
            }
        }
    }
}

impl Error for CellError {}

// Legacy error types for backwards compatibility
#[derive(Debug)]
pub struct CellReadError;
#[derive(Debug)]
pub struct CellWriteError;

impl Display for CellReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CellReadError - Unable to read cell!")
    }
}

impl Display for CellWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CellWriteError - Unable to write cell!")
    }
}

impl Error for CellReadError {}
impl Error for CellWriteError {}

impl<'a, K: Clone, V: Clone> CellGuard<'a, K, V> {
    /// Creates a `CellGuard` from a raw pointer to a `Cell`.
    ///
    /// This function captures the cell's current state (version, marker state,
    /// move destination) for later validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to a valid, properly aligned `Cell<K, V>`
    /// - The `Cell` will remain valid for the lifetime `'a`
    #[must_use = "this returns a Result that should be checked for errors"]
    pub unsafe fn from_raw(ptr: *const Cell<K, V>) -> Result<CellGuard<'a, K, V>, Box<dyn Error>> {
        // SAFETY: Caller guarantees ptr is valid and properly aligned
        let cell = &*ptr;
        let version = cell.version.load(AtomicOrdering::SeqCst);
        // SAFETY: Caller guarantees cell is valid, so its key UnsafeCell is valid
        let key = (*cell.key.get()).clone();
        let marker_state = cell.load_marker_state(AtomicOrdering::SeqCst);
        let move_dest = cell.move_dest.load(AtomicOrdering::SeqCst);

        Ok(CellGuard {
            inner: cell,
            is_filled: key.is_some(),
            cache_version: version,
            cache_marker_state: marker_state,
            cache_move_dest: move_dest,
            cache_data: OnceCell::new(),
            _phantom: PhantomData,
        })
    }
}

/// An iterator over cells in a packed memory array region.
///
/// `CellIterator` yields [`CellGuard`]s for each cell in a contiguous range,
/// allowing safe iteration over cells while respecting version consistency.
///
/// # Panics
///
/// The iterator panics if `CellGuard::from_raw` fails due to version mismatch.
/// In concurrent scenarios where this is possible, consider using a retry loop
/// around the iteration logic.
pub struct CellIterator<'a, K: Ord + Clone, V: Clone> {
    count: usize,
    address: *const Cell<K, V>,
    end_address: *const Cell<K, V>,
    _phantom: PhantomData<&'a Cell<K, V>>,
}

impl<'a, K: Clone + Ord, V: Clone> CellIterator<'a, K, V> {
    /// Creates a new cell iterator starting at `ptr` and ending at `last_cell_address`.
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure that all pointers in the range `[ptr, last_cell_address]`
    /// point to valid `Cell<K, V>` instances that will remain valid for lifetime `'a`.
    pub fn new(
        ptr: *const Cell<K, V>,
        last_cell_address: *const Cell<K, V>,
    ) -> CellIterator<'a, K, V> {
        CellIterator {
            count: 0,
            address: ptr,
            end_address: last_cell_address,
            _phantom: PhantomData,
        }
    }
}

impl<'a, K: Ord + Clone, V: Clone> Iterator for CellIterator<'a, K, V> {
    type Item = CellGuard<'a, K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count > 0 {
            // SAFETY: We stay within bounds checked against end_address,
            // and the caller of new() guarantees all cells in range are valid
            self.address = unsafe { self.address.add(1) };
            if self.address > self.end_address {
                return None;
            }
        }

        self.count += 1;

        // SAFETY: address is within the valid range [start, end_address]
        // guaranteed by the bounds check above and caller's contract in new()
        let guard = unsafe { CellGuard::from_raw(self.address) }.unwrap();
        Some(guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering as AtomicOrdering;

    #[test]
    fn test_cell_guard_from_raw_with_matching_versions() {
        // Create a cell with default state
        let cell: Cell<u32, String> = Cell::default();

        // Default cell has version 1 and Empty marker state
        let guard_result = unsafe { CellGuard::from_raw(&cell as *const Cell<u32, String>) };

        assert!(guard_result.is_ok(), "CellGuard::from_raw should succeed");
        let guard = guard_result.unwrap();
        assert_eq!(guard.cache_version, 1);
        assert!(guard.is_empty());
        assert_eq!(guard.marker_state(), MarkerState::Empty);
    }

    #[test]
    fn test_cell_guard_detects_version_change_on_cache() {
        // Create a cell with default state
        let cell: Cell<u32, String> = Cell::default();

        // Create guard while version is 1
        let guard_result = unsafe { CellGuard::from_raw(&cell as *const Cell<u32, String>) };
        assert!(guard_result.is_ok());
        let guard = guard_result.unwrap();

        // Now change the version (simulating concurrent modification)
        cell.version.store(2, AtomicOrdering::SeqCst);

        // Trying to cache should fail due to version mismatch
        let cache_result = guard.cache();
        assert!(
            cache_result.is_err(),
            "cache() should fail when version changed"
        );
    }

    #[test]
    fn test_marker_state_conversion() {
        assert_eq!(MarkerState::from_u8(0), MarkerState::Empty);
        assert_eq!(MarkerState::from_u8(1), MarkerState::Move);
        assert_eq!(MarkerState::from_u8(2), MarkerState::Inserting);
        assert_eq!(MarkerState::from_u8(3), MarkerState::Deleting);
        assert_eq!(MarkerState::from_u8(255), MarkerState::Empty); // Invalid defaults to Empty
    }

    #[test]
    fn test_marker_version_accessor() {
        let marker_empty = Marker::Empty(42);
        assert_eq!(*marker_empty.version(), 42);

        let marker_move = Marker::Move(17, 5);
        assert_eq!(*marker_move.version(), 17);

        let marker_insert = Marker::InsertCell(99);
        assert_eq!(*marker_insert.version(), 99);

        let marker_delete = Marker::DeleteCell(3);
        assert_eq!(*marker_delete.version(), 3);
    }

    #[test]
    fn test_marker_state_method() {
        let marker_empty = Marker::Empty(1);
        assert_eq!(marker_empty.state(), MarkerState::Empty);

        let marker_move = Marker::Move(1, 5);
        assert_eq!(marker_move.state(), MarkerState::Move);

        let marker_insert = Marker::InsertCell(1);
        assert_eq!(marker_insert.state(), MarkerState::Inserting);

        let marker_delete = Marker::DeleteCell(1);
        assert_eq!(marker_delete.state(), MarkerState::Deleting);
    }

    #[test]
    fn test_cell_load_marker() {
        let cell: Cell<u32, String> = Cell::default();

        // Default state
        let marker = cell.load_marker(AtomicOrdering::SeqCst);
        assert!(matches!(marker, Marker::Empty(1)));

        // Set to Move state
        cell.marker_state
            .store(MarkerState::Move as u8, AtomicOrdering::SeqCst);
        cell.move_dest.store(42, AtomicOrdering::SeqCst);
        let marker = cell.load_marker(AtomicOrdering::SeqCst);
        assert!(matches!(marker, Marker::Move(1, 42)));
    }

    #[test]
    fn test_cell_error_enum_variants() {
        // Test CellError::VersionMismatch
        let version_error = CellError::VersionMismatch;
        assert_eq!(version_error, CellError::VersionMismatch);
        assert_ne!(version_error, CellError::CasFailed);
        assert!(format!("{}", version_error).contains("version mismatch"));

        // Test CellError::CasFailed
        let cas_error = CellError::CasFailed;
        assert_eq!(cas_error, CellError::CasFailed);
        assert_ne!(cas_error, CellError::VersionMismatch);
        assert!(format!("{}", cas_error).contains("CAS"));

        // Test that CellError implements Debug
        let debug_str = format!("{:?}", CellError::VersionMismatch);
        assert!(debug_str.contains("VersionMismatch"));

        // Test Copy trait
        let err1 = CellError::CasFailed;
        let err2 = err1; // Copy
        assert_eq!(err1, err2);

        // Test Clone trait
        let err3 = err1.clone();
        assert_eq!(err1, err3);
    }

    #[test]
    fn test_cell_error_is_std_error() {
        use std::error::Error;

        let error: Box<dyn Error> = Box::new(CellError::VersionMismatch);
        assert!(error.to_string().contains("version mismatch"));

        let error: Box<dyn Error> = Box::new(CellError::CasFailed);
        assert!(error.to_string().contains("CAS"));
    }
}
