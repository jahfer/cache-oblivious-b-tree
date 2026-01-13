use std::cell::OnceCell;
use std::cell::UnsafeCell;
use std::cmp::{Ord, Ordering};
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering as AtomicOrdering};

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

#[derive(Debug, Copy, Clone)]
pub enum Marker<K: Clone, V: Clone> {
    Empty(u32),
    Move(u32, isize),
    InsertCell(u32, K, V),
    DeleteCell(u32, K),
}

impl<K: Clone, V: Clone> Marker<K, V> {
    pub fn version(&self) -> &u32 {
        match self {
            Marker::Empty(v)
            | Marker::Move(v, _)
            | Marker::InsertCell(v, _, _)
            | Marker::DeleteCell(v, _) => v,
        }
    }
}

pub struct Cell<K: Clone, V: Clone> {
    pub version: AtomicU32, // SeqLock version (even = stable)
    pub marker: Option<AtomicPtr<Marker<K, V>>>,
    pub key: UnsafeCell<Option<K>>,
    pub value: UnsafeCell<Option<V>>,
}

unsafe impl<K: Clone, V: Clone> Send for Cell<K, V> {}
unsafe impl<K: Clone, V: Clone> Sync for Cell<K, V> {}

impl<K: Clone, V: Clone> Cell<K, V> {
    pub fn new(marker_ptr: *mut Marker<K, V>) -> Cell<K, V> {
        Cell {
            version: AtomicU32::new(0), // even = stable
            marker: Some(AtomicPtr::new(marker_ptr)),
            key: UnsafeCell::new(None),
            value: UnsafeCell::new(None),
        }
    }

    /// Acquire exclusive write access using SeqLock protocol.
    /// Spins until version is even (stable), then atomically sets it to odd (write in progress).
    /// Returns a guard that bumps version back to even on drop.
    pub fn begin_write(&self) -> SeqLockWriteGuard<'_, K, V> {
        loop {
            let v = self.version.load(AtomicOrdering::Acquire);
            // If version is odd, a write is in progress - spin
            if v % 2 == 1 {
                std::hint::spin_loop();
                continue;
            }
            // Try to CAS from even to odd (acquire write lock)
            if self
                .version
                .compare_exchange_weak(
                    v,
                    v.wrapping_add(1),
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                )
                .is_ok()
            {
                return SeqLockWriteGuard { cell: self };
            }
        }
    }

    /// Read cell data consistently using SeqLock protocol.
    /// Loops until a stable read is achieved (version unchanged and even throughout).
    pub fn read_consistent<F, R>(&self, f: F) -> R
    where
        F: Fn(&Option<K>, &Option<V>) -> R,
        R: Clone,
    {
        loop {
            let v1 = self.version.load(AtomicOrdering::Acquire);
            // If version is odd, a write is in progress - spin
            if v1 % 2 == 1 {
                std::hint::spin_loop();
                continue;
            }

            // Read the data
            let result = f(unsafe { &*self.key.get() }, unsafe { &*self.value.get() });

            // Ensure all reads complete before checking version
            std::sync::atomic::fence(AtomicOrdering::Acquire);

            // If version unchanged, we got a consistent read
            if v1 == self.version.load(AtomicOrdering::Relaxed) {
                return result;
            }
            // Version changed - retry
        }
    }

    /// Read cell data and return both the result and the version observed.
    /// Useful for caching the version to detect staleness later.
    pub fn read_with_version<F, R>(&self, f: F) -> (R, u32)
    where
        F: Fn(&Option<K>, &Option<V>) -> R,
        R: Clone,
    {
        loop {
            let v1 = self.version.load(AtomicOrdering::Acquire);
            // If version is odd, a write is in progress - spin
            if v1 % 2 == 1 {
                std::hint::spin_loop();
                continue;
            }

            // Read the data
            let result = f(unsafe { &*self.key.get() }, unsafe { &*self.value.get() });

            // Ensure all reads complete before checking version
            std::sync::atomic::fence(AtomicOrdering::Acquire);

            // If version unchanged, we got a consistent read
            if v1 == self.version.load(AtomicOrdering::Relaxed) {
                return (result, v1);
            }
            // Version changed - retry
        }
    }
}

/// RAII guard for SeqLock write access.
/// While held, the cell's version is odd (write in progress).
/// On drop, increments version to even (stable).
pub struct SeqLockWriteGuard<'a, K: Clone, V: Clone> {
    cell: &'a Cell<K, V>,
}

impl<K: Clone, V: Clone> SeqLockWriteGuard<'_, K, V> {
    /// Write new key and value to the cell.
    /// Safety: This is safe because we hold exclusive write access.
    pub fn write(&self, key: Option<K>, value: Option<V>) {
        unsafe {
            *self.cell.key.get() = key;
            *self.cell.value.get() = value;
        }
    }

    /// Get mutable reference to the key.
    /// Safety: This is safe because we hold exclusive write access.
    pub fn key_mut(&self) -> &mut Option<K> {
        unsafe { &mut *self.cell.key.get() }
    }

    /// Get mutable reference to the value.
    /// Safety: This is safe because we hold exclusive write access.
    pub fn value_mut(&self) -> &mut Option<V> {
        unsafe { &mut *self.cell.value.get() }
    }
}

impl<K: Clone, V: Clone> Drop for SeqLockWriteGuard<'_, K, V> {
    fn drop(&mut self) {
        // Bump version from odd to even (release write lock)
        self.cell.version.fetch_add(1, AtomicOrdering::Release);
    }
}

impl<K, V> Default for Cell<K, V>
where
    K: Clone,
    V: Clone,
{
    fn default() -> Self {
        let marker = Box::new(Marker::<K, V>::Empty(0));
        let ptr = Box::into_raw(marker);
        Cell::new(ptr)
    }
}

impl<K: Clone, V: Clone> Drop for Cell<K, V> {
    fn drop(&mut self) {
        let ptr = self.marker.take().unwrap();
        let marker = ptr.load(AtomicOrdering::Acquire);
        unsafe { drop(Box::from_raw(marker)) };
    }
}

impl<K: Debug + Clone, V: Debug + Clone> Debug for Cell<K, V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let version = self.version.load(AtomicOrdering::Acquire);
        let marker = unsafe { &*self.marker.as_ref().unwrap().load(AtomicOrdering::Acquire) };
        let key = unsafe { &*self.key.get() };
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

#[derive(Debug, Copy, Clone)]
pub struct CellData<K: Clone, V: Clone> {
    pub key: K,
    pub value: V,
    pub marker: Marker<K, V>,
}

pub struct CellGuard<'a, K: 'a + Clone, V: 'a + Clone> {
    pub inner: &'a Cell<K, V>,
    pub cache_version: u32,
    pub is_filled: bool,
    cache_data: OnceCell<Option<CellData<K, V>>>,
    cache_marker_ptr: *mut Marker<K, V>,
    _phantom: PhantomData<&'a Cell<K, V>>,
}

impl<K: Clone, V: Clone> CellGuard<'_, K, V> {
    pub fn is_empty(&self) -> bool {
        !self.is_filled
    }

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
        let marker_version = unsafe { *(*current_marker_raw).version() };
        if version != marker_version {
            return Err(CellReadError {});
        }

        // Only clone after validation passes
        let key = unsafe { (*self.inner.key.get()).clone() };
        let result = if let Some(k) = key {
            let value = unsafe { (*self.inner.value.get()).clone() };
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
    pub unsafe fn from_raw(ptr: *const Cell<K, V>) -> Result<CellGuard<'a, K, V>, Box<dyn Error>> {
        let cell = &*ptr;
        let version = cell.version.load(AtomicOrdering::SeqCst);
        let key = (*cell.key.get()).clone();
        let current_marker_raw = cell.marker.as_ref().unwrap().load(AtomicOrdering::SeqCst);

        // TODO: Check version in marker to make sure the cell was not modified in between

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

pub struct CellIterator<'a, K: Ord + Clone, V: Clone> {
    count: usize,
    address: *const Cell<K, V>,
    end_address: *const Cell<K, V>,
    _phantom: PhantomData<&'a Cell<K, V>>,
}

impl<'a, K: Clone + Ord, V: Clone> CellIterator<'a, K, V> {
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
            self.address = unsafe { self.address.add(1) };
            if self.address > self.end_address {
                return None;
            }
        }

        self.count += 1;

        let guard = unsafe { CellGuard::from_raw(self.address) }.unwrap();
        Some(guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_default_version_is_zero_even() {
        let cell: Cell<u32, u32> = Cell::default();
        let version = cell.version.load(AtomicOrdering::Acquire);
        assert_eq!(
            version, 0,
            "Cell should initialize with version 0 (even = stable)"
        );
        assert_eq!(version % 2, 0, "Version should be even (stable state)");
    }

    #[test]
    fn test_cell_new_version_is_zero_even() {
        let marker = Box::new(Marker::<u32, u32>::Empty(0));
        let ptr = Box::into_raw(marker);
        let cell: Cell<u32, u32> = Cell::new(ptr);
        let version = cell.version.load(AtomicOrdering::Acquire);
        assert_eq!(
            version, 0,
            "Cell::new should initialize with version 0 (even = stable)"
        );
        assert_eq!(version % 2, 0, "Version should be even (stable state)");
    }

    #[test]
    fn test_version_can_hold_large_values() {
        let cell: Cell<u32, u32> = Cell::default();
        // Store a value larger than u16::MAX to verify u32 works
        cell.version.store(70000, AtomicOrdering::Release);
        let version = cell.version.load(AtomicOrdering::Acquire);
        assert_eq!(version, 70000, "Version should support values > u16::MAX");
    }

    // ==================== SeqLock Primitive Tests ====================

    #[test]
    fn test_begin_write_acquires_lock() {
        let cell: Cell<u32, u32> = Cell::default();
        assert_eq!(
            cell.version.load(AtomicOrdering::Acquire) % 2,
            0,
            "Version should start even"
        );

        let _guard = cell.begin_write();
        assert_eq!(
            cell.version.load(AtomicOrdering::Acquire) % 2,
            1,
            "Version should be odd while write guard is held"
        );
    }

    #[test]
    fn test_write_guard_releases_on_drop() {
        let cell: Cell<u32, u32> = Cell::default();
        let initial_version = cell.version.load(AtomicOrdering::Acquire);
        assert_eq!(initial_version, 0);

        {
            let _guard = cell.begin_write();
            assert_eq!(cell.version.load(AtomicOrdering::Acquire), 1);
        }

        // After drop, version should be incremented to 2 (even again)
        assert_eq!(cell.version.load(AtomicOrdering::Acquire), 2);
        assert_eq!(
            cell.version.load(AtomicOrdering::Acquire) % 2,
            0,
            "Version should be even after write guard dropped"
        );
    }

    #[test]
    fn test_write_guard_write_method() {
        let cell: Cell<u32, String> = Cell::default();

        {
            let guard = cell.begin_write();
            guard.write(Some(42), Some("hello".to_string()));
        }

        // Verify data was written
        let (key, value) = cell.read_consistent(|k, v| (k.clone(), v.clone()));
        assert_eq!(key, Some(42));
        assert_eq!(value, Some("hello".to_string()));
    }

    #[test]
    fn test_write_guard_key_mut_and_value_mut() {
        let cell: Cell<u32, String> = Cell::default();

        {
            let guard = cell.begin_write();
            *guard.key_mut() = Some(100);
            *guard.value_mut() = Some("world".to_string());
        }

        let (key, value) = cell.read_consistent(|k, v| (k.clone(), v.clone()));
        assert_eq!(key, Some(100));
        assert_eq!(value, Some("world".to_string()));
    }

    #[test]
    fn test_read_consistent_returns_stable_data() {
        let cell: Cell<u32, u32> = Cell::default();

        // Write some data
        {
            let guard = cell.begin_write();
            guard.write(Some(123), Some(456));
        }

        // Read consistently
        let (key, value) = cell.read_consistent(|k, v| (k.clone(), v.clone()));
        assert_eq!(key, Some(123));
        assert_eq!(value, Some(456));
    }

    #[test]
    fn test_read_consistent_empty_cell() {
        let cell: Cell<u32, u32> = Cell::default();

        let (key, value) = cell.read_consistent(|k, v| (k.clone(), v.clone()));
        assert_eq!(key, None);
        assert_eq!(value, None);
    }

    #[test]
    fn test_read_with_version_returns_version() {
        let cell: Cell<u32, u32> = Cell::default();

        // Initial read
        let ((key, value), version) = cell.read_with_version(|k, v| (k.clone(), v.clone()));
        assert_eq!(key, None);
        assert_eq!(value, None);
        assert_eq!(version, 0);

        // Write and read again
        {
            let guard = cell.begin_write();
            guard.write(Some(42), Some(84));
        }

        let ((key, value), version) = cell.read_with_version(|k, v| (k.clone(), v.clone()));
        assert_eq!(key, Some(42));
        assert_eq!(value, Some(84));
        assert_eq!(version, 2, "Version should be 2 after one write cycle");
    }

    #[test]
    fn test_multiple_writes_increment_version() {
        let cell: Cell<u32, u32> = Cell::default();

        // First write
        {
            let guard = cell.begin_write();
            guard.write(Some(1), Some(1));
        }
        assert_eq!(cell.version.load(AtomicOrdering::Acquire), 2);

        // Second write
        {
            let guard = cell.begin_write();
            guard.write(Some(2), Some(2));
        }
        assert_eq!(cell.version.load(AtomicOrdering::Acquire), 4);

        // Third write
        {
            let guard = cell.begin_write();
            guard.write(Some(3), Some(3));
        }
        assert_eq!(cell.version.load(AtomicOrdering::Acquire), 6);
    }

    #[test]
    fn test_version_wrapping() {
        let cell: Cell<u32, u32> = Cell::default();
        // Set version close to u32::MAX (but even)
        cell.version.store(u32::MAX - 1, AtomicOrdering::Release);

        {
            let _guard = cell.begin_write();
            // Version should now be u32::MAX (odd)
            assert_eq!(cell.version.load(AtomicOrdering::Acquire), u32::MAX);
        }

        // After drop, should wrap to 0
        assert_eq!(cell.version.load(AtomicOrdering::Acquire), 0);
    }

    #[test]
    fn test_seqlock_write_guard_is_send() {
        fn assert_send<T: Send>() {}
        // SeqLockWriteGuard should be Send if K and V are Send
        assert_send::<SeqLockWriteGuard<'_, u32, u32>>();
    }
}
