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

#[derive(Debug, Copy, Clone)]
pub enum Marker<K: Clone, V: Clone> {
    Empty(u16),
    Move(u16, isize),
    InsertCell(u16, K, V),
    DeleteCell(u16, K),
}

impl<K: Clone, V: Clone> Marker<K, V> {
    pub fn version(&self) -> &u16 {
        match self {
            Marker::Empty(v)
            | Marker::Move(v, _)
            | Marker::InsertCell(v, _, _)
            | Marker::DeleteCell(v, _) => v,
        }
    }
}

pub struct Cell<K: Clone, V: Clone> {
    pub version: AtomicU16,
    pub marker: Option<AtomicPtr<Marker<K, V>>>,
    pub key: UnsafeCell<Option<K>>,
    pub value: UnsafeCell<Option<V>>,
}

unsafe impl<K: Clone, V: Clone> Send for Cell<K, V> {}
unsafe impl<K: Clone, V: Clone> Sync for Cell<K, V> {}

impl<K: Clone, V: Clone> Cell<K, V> {
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
    pub cache_version: u16,
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

        // Check version in marker to make sure the cell was not modified in between
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
}
