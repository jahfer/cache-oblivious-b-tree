use std::cell::OnceCell;
use std::cell::UnsafeCell;
use std::cmp::{Ord, Ordering};
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU16, Ordering as AtomicOrdering};

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
/// Each marker variant contains a version number that must match the cell's
/// version for a read to be considered valid.
///
/// # State Transitions
///
/// - `Empty(v)` → `InsertCell(v, k, val)` → `Empty(v+1)`: Insert operation
/// - `Empty(v)` → `Move(v, dest)` → `Empty(v+1)`: Rebalance operation
/// - `Empty(v)` → `DeleteCell(v, k)` → `Empty(v+1)`: Delete operation
#[derive(Debug, Copy, Clone)]
pub enum Marker<K: Clone, V: Clone> {
    /// Cell is idle and available for operations.
    /// Contains committed data if key/value are `Some`, otherwise empty.
    Empty(u16),
    /// Cell's data is being moved to `dest_index` during rebalance.
    /// Readers should follow the destination index to find the data.
    Move(u16, isize),
    /// An insert operation is in progress with the given key and value.
    InsertCell(u16, K, V),
    /// A delete operation is in progress for the given key.
    DeleteCell(u16, K),
}

impl<K: Clone, V: Clone> Marker<K, V> {
    /// Returns a reference to the version number embedded in this marker.
    pub fn version(&self) -> &u16 {
        match self {
            Marker::Empty(v)
            | Marker::Move(v, _)
            | Marker::InsertCell(v, _, _)
            | Marker::DeleteCell(v, _) => v,
        }
    }
}

/// A single cell in the packed memory array.
///
/// Each cell stores an optional key-value pair along with synchronization
/// primitives for lock-free concurrent access:
///
/// - `version`: Monotonically increasing counter, incremented on each modification
/// - `marker`: Points to a [`Marker`] indicating the current operation state
/// - `key`/`value`: The actual data stored in `UnsafeCell` for interior mutability
///
/// # Thread Safety
///
/// `Cell` implements `Send` and `Sync` because:
/// - `version` is an `AtomicU16` with proper atomic operations
/// - `marker` is an `AtomicPtr` with proper atomic operations
/// - `key`/`value` access is guarded by version validation and marker checks
///
/// The lock-free protocol ensures that concurrent readers and writers
/// can operate safely without data races:
/// 1. Writer sets marker to indicate operation in progress
/// 2. Writer modifies key/value
/// 3. Writer increments version and updates marker to `Empty`
/// 4. Readers validate version before and after reading to detect concurrent modifications
pub struct Cell<K: Clone, V: Clone> {
    pub version: AtomicU16,
    pub marker: Option<AtomicPtr<Marker<K, V>>>,
    pub key: UnsafeCell<Option<K>>,
    pub value: UnsafeCell<Option<V>>,
}

// SAFETY: Cell is Send because all its fields can be safely transferred between threads:
// - AtomicU16 and AtomicPtr are Send
// - UnsafeCell contents are protected by the version/marker protocol
unsafe impl<K: Clone, V: Clone> Send for Cell<K, V> {}

// SAFETY: Cell is Sync because concurrent access is safe through:
// - Atomic operations on version and marker
// - Version validation preventing observation of partially-written data
unsafe impl<K: Clone, V: Clone> Sync for Cell<K, V> {}

impl<K: Clone, V: Clone> Cell<K, V> {
    /// Creates a new cell with the given marker pointer.
    ///
    /// The cell starts with version 1 and empty key/value slots.
    pub fn new(marker_ptr: *mut Marker<K, V>) -> Cell<K, V> {
        Cell {
            version: AtomicU16::new(1),
            marker: Some(AtomicPtr::new(marker_ptr)),
            key: UnsafeCell::new(None),
            value: UnsafeCell::new(None),
        }
    }
}

impl<K, V> Default for Cell<K, V>
where
    K: Clone,
    V: Clone,
{
    /// Creates a default empty cell with version 1 and an `Empty(1)` marker.
    fn default() -> Self {
        let marker = Box::new(Marker::<K, V>::Empty(1));
        let ptr = Box::into_raw(marker);
        Cell::new(ptr)
    }
}

impl<K: Clone, V: Clone> Drop for Cell<K, V> {
    fn drop(&mut self) {
        let ptr = self.marker.take().unwrap();
        let marker = ptr.load(AtomicOrdering::Acquire);
        // SAFETY: The marker was created via Box::into_raw and we have exclusive
        // access during Drop (guaranteed by Rust's ownership rules).
        // No other references can exist because Drop requires &mut self.
        unsafe { drop(Box::from_raw(marker)) };
    }
}

impl<K: Debug + Clone, V: Debug + Clone> Debug for Cell<K, V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let version = self.version.load(AtomicOrdering::Acquire);
        // SAFETY: marker pointer is always valid (set in new/default, only freed in drop)
        let marker = unsafe { &*self.marker.as_ref().unwrap().load(AtomicOrdering::Acquire) };
        // SAFETY: Debug formatting only reads data; we accept potential inconsistency
        // as this is for debugging purposes only, not for correctness-critical reads
        let key = unsafe { &*self.key.get() };
        // SAFETY: Same as above - debug read only
        let value = unsafe { &*self.value.get() };

        let mut dbg_struct = formatter.debug_struct("Cell");

        dbg_struct
            .field("version", &version)
            .field("marker", marker)
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
    pub marker: Marker<K, V>,
}

/// A guard that provides safe access to a cell's data.
///
/// `CellGuard` captures a consistent snapshot of a cell's state at creation time,
/// including the version number and marker pointer. It provides lazy caching of
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
    cache_marker_ptr: *mut Marker<K, V>,
    _phantom: PhantomData<&'a Cell<K, V>>,
}

impl<K: Clone, V: Clone> CellGuard<'_, K, V> {
    /// Returns `true` if this cell contains no key-value data.
    pub fn is_empty(&self) -> bool {
        !self.is_filled
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
        let current_marker_raw = self
            .inner
            .marker
            .as_ref()
            .unwrap()
            .load(AtomicOrdering::SeqCst);

        // Check version consistency BEFORE cloning any data
        // SAFETY: current_marker_raw is guaranteed to be valid because:
        // - It was loaded from an AtomicPtr that is always initialized with a valid Box::into_raw
        // - The marker is only deallocated in Cell::drop, which requires exclusive access
        let marker_version = unsafe { *(*current_marker_raw).version() };
        if version != marker_version {
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
            // SAFETY: current_marker_raw validity established above
            let marker = unsafe { (*current_marker_raw).clone() };
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

    /// Atomically updates the cell's marker using compare-and-swap.
    ///
    /// This operation will succeed only if the marker hasn't been modified
    /// since the guard was created. On success, returns the old marker pointer
    /// which the caller is responsible for managing (typically by overwriting
    /// its contents for reuse).
    ///
    /// # Errors
    ///
    /// Returns [`CellWriteError`] if the CAS operation fails due to concurrent
    /// modification. The caller should typically retry the entire operation.
    ///
    /// # Memory Safety
    ///
    /// On failure, the newly allocated marker is immediately deallocated.
    /// On success, the caller receives ownership of the old marker pointer.
    #[must_use = "this returns a Result that should be checked for errors"]
    pub fn update(&mut self, marker: Marker<K, V>) -> Result<*mut Marker<K, V>, Box<dyn Error>> {
        let boxed_marker = Box::new(marker);
        let new_marker_raw = Box::into_raw(boxed_marker);
        let result = self.inner.marker.as_ref().unwrap().compare_exchange(
            self.cache_marker_ptr,
            new_marker_raw,
            AtomicOrdering::SeqCst,
            AtomicOrdering::SeqCst,
        );

        if result.is_err() {
            // Deallocate memory, try again next time
            // SAFETY: new_marker_raw was just created via Box::into_raw above,
            // and the CAS failed so we still own it exclusively
            unsafe { drop(Box::from_raw(new_marker_raw)) };
            // Marker has been updated by another process, start loop over
            return Err(Box::new(CellWriteError {}));
        } else {
            let old_marker_box = self.cache_marker_ptr;
            self.cache_marker_ptr = new_marker_raw;
            Ok(old_marker_box)
        }
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
    /// This function validates that the cell's version matches the marker's
    /// embedded version before returning a valid guard.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to a valid, properly aligned `Cell<K, V>`
    /// - The `Cell` will remain valid for the lifetime `'a`
    /// - The `Cell`'s marker pointer is valid and non-null
    #[must_use = "this returns a Result that should be checked for errors"]
    pub unsafe fn from_raw(ptr: *const Cell<K, V>) -> Result<CellGuard<'a, K, V>, Box<dyn Error>> {
        // SAFETY: Caller guarantees ptr is valid and properly aligned
        let cell = &*ptr;
        let version = cell.version.load(AtomicOrdering::SeqCst);
        // SAFETY: Caller guarantees cell is valid, so its key UnsafeCell is valid
        let key = (*cell.key.get()).clone();
        let current_marker_raw = cell.marker.as_ref().unwrap().load(AtomicOrdering::SeqCst);

        // Check version in marker to make sure the cell was not modified in between
        // SAFETY: current_marker_raw loaded from valid AtomicPtr, always points to valid Marker
        let marker_version = *(*current_marker_raw).version();
        if version != marker_version {
            return Err(Box::new(CellReadError {}));
        }

        Ok(CellGuard {
            inner: cell,
            is_filled: key.is_some(),
            cache_version: version,
            cache_marker_ptr: current_marker_raw,
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
        // Create a cell with matching version (1) in both cell and marker
        let cell: Cell<u32, String> = Cell::default();

        // Default cell has version 1 and marker with version 1
        let guard_result = unsafe { CellGuard::from_raw(&cell as *const Cell<u32, String>) };

        assert!(
            guard_result.is_ok(),
            "CellGuard::from_raw should succeed when versions match"
        );
        let guard = guard_result.unwrap();
        assert_eq!(guard.cache_version, 1);
        assert!(guard.is_empty());
    }

    #[test]
    fn test_cell_guard_from_raw_with_mismatched_versions() {
        // Create a cell with version 1 in marker
        let cell: Cell<u32, String> = Cell::default();

        // Bump the cell version without updating the marker version
        cell.version.store(2, AtomicOrdering::SeqCst);

        // Now cell.version is 2 but marker.version() is still 1
        let guard_result = unsafe { CellGuard::from_raw(&cell as *const Cell<u32, String>) };

        assert!(
            guard_result.is_err(),
            "CellGuard::from_raw should fail when versions mismatch"
        );
    }

    #[test]
    fn test_cell_guard_from_raw_version_validation_detects_concurrent_modification() {
        // This test simulates the scenario where a cell is modified between
        // reading its version and creating the guard

        // Create a cell with initial version
        let marker = Box::new(Marker::<u32, String>::Empty(5));
        let ptr = Box::into_raw(marker);
        let cell = Cell::<u32, String>::new(ptr);
        cell.version.store(5, AtomicOrdering::SeqCst);

        // Versions match (both 5), so this should succeed
        let guard_result = unsafe { CellGuard::from_raw(&cell as *const Cell<u32, String>) };
        assert!(guard_result.is_ok(), "Should succeed when versions match");

        // Now simulate a concurrent modification by changing cell version
        cell.version.store(6, AtomicOrdering::SeqCst);

        // Marker still has version 5, cell has version 6 - should fail
        let guard_result = unsafe { CellGuard::from_raw(&cell as *const Cell<u32, String>) };
        assert!(
            guard_result.is_err(),
            "Should fail when cell version differs from marker version"
        );
    }

    #[test]
    fn test_marker_version_accessor() {
        let marker_empty = Marker::<u32, String>::Empty(42);
        assert_eq!(*marker_empty.version(), 42);

        let marker_move = Marker::<u32, String>::Move(17, 5);
        assert_eq!(*marker_move.version(), 17);

        let marker_insert = Marker::<u32, String>::InsertCell(99, 123, String::from("test"));
        assert_eq!(*marker_insert.version(), 99);

        let marker_delete = Marker::<u32, String>::DeleteCell(3, 456);
        assert_eq!(*marker_delete.version(), 3);
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
