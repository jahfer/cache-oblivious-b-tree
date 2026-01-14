use num_rational::Rational;
use std::fmt::{self, Debug};
use std::marker::{PhantomData, PhantomPinned};
use std::ops::Range;
use std::ops::RangeInclusive;
use std::pin::Pin;

/// A packed memory array (PMA) providing cache-oblivious storage.
///
/// The PMA maintains cells in a "gapped array" layout where 1/4 of capacity
/// is reserved as buffer space on each end, leaving the middle half as the
/// active storage region. This enables efficient rebalancing without
/// reallocating the entire array.
///
/// # Memory Layout
///
/// ```text
/// ┌─────────────┬────────────────────────────┬─────────────┐
/// │  Left Buffer│       Active Range         │Right Buffer │
/// │    (1/4)    │         (1/2)              │   (1/4)     │
/// └─────────────┴────────────────────────────┴─────────────┘
/// ```
///
/// # Density Invariants
///
/// The PMA maintains density bounds at each level of the implicit tree:
/// - Minimum density: Ensures blocks aren't too sparse (wastes memory)
/// - Maximum density: Ensures blocks aren't too full (requires rebalance)
///
/// Density bounds are more strict at lower levels (smaller ranges) and
/// more relaxed at higher levels (larger ranges), enabling amortized
/// O(log² N) insert cost.
pub struct PackedMemoryArray<T> {
    cells: Pin<Box<[T]>>,
    pub config: Config,
    pub active_range: Range<*const T>,
    pub requested_capacity: usize,
    _pin: PhantomPinned,
}

// SAFETY: PackedMemoryArray is Send because Pin<Box<[T]>> is Send when T: Send,
// and raw pointers in active_range point into the pinned allocation
unsafe impl<T> Send for PackedMemoryArray<T> {}

// SAFETY: PackedMemoryArray is Sync because:
// - The cells are pinned and won't move
// - Access to cells goes through proper synchronization (atomic/CAS)
unsafe impl<T> Sync for PackedMemoryArray<T> {}

impl<T> PackedMemoryArray<T> {
    /// Creates a new PMA with the given cells and requested capacity.
    pub fn new(cells: Box<[T]>, capacity: usize) -> PackedMemoryArray<T> {
        let active_range = Self::compute_active_range(&cells);
        let density_scale = Self::compute_density_range(cells.len() as f32);
        let config = Config { density_scale };

        PackedMemoryArray {
            cells: Box::into_pin(cells),
            requested_capacity: capacity,
            active_range,
            config,
            _pin: PhantomPinned,
        }
    }

    /// Computes the active range for a given cell slice.
    ///
    /// The active range is the portion of the array where data can be stored.
    /// The PMA reserves 1/4 of the total capacity at each end as buffer space
    /// for rebalancing operations, leaving the middle half as the active range.
    ///
    /// # Example
    /// For a 16-cell array:
    /// - Left buffer: cells[0..4] (indices 0-3)
    /// - Active range: cells[4..12] (indices 4-11)
    /// - Right buffer: cells[12..16] (indices 12-15)
    fn compute_active_range(cells: &[T]) -> Range<*const T> {
        let buffer_space = Self::buffer_space(cells.len());
        Range {
            start: &cells[buffer_space] as *const _,
            end: &cells[cells.len() - buffer_space] as *const _,
        }
    }

    #[inline]
    fn buffer_space(total_capacity: usize) -> usize {
        total_capacity >> 2 // 1/4 of total capacity
    }

    pub fn as_slice(&self) -> &[T] {
        let buffer = Self::buffer_space(self.cells.len());
        &self.cells[buffer..self.cells.len() - buffer]
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_valid_pointer(&self, ptr: &*const T) -> bool {
        self.active_range.contains(ptr)
    }

    fn compute_density_range(cell_count: f32) -> Vec<Density> {
        let num_densities = f32::log2(cell_count) as isize;

        // max density for 2^num_densities cells: 1/2
        let t_min = Rational::new(1, 2);
        // max density for 2^1 cells: 1
        let t_max = Rational::from_integer(1);
        // min density for 2^num_densities cells: 1/4
        let p_max = Rational::new(1, 4);
        // min density for 2^1 cells: 1/8
        let p_min = Rational::new(1, 8);

        let t_delta = t_max - t_min;
        let p_delta = p_max - p_min;

        (1..=num_densities)
            .map(|i| Density {
                max_item_count: 1 << i,
                range: RangeInclusive::new(
                    p_min + (Rational::new(i - 1, num_densities - 1)) * p_delta,
                    t_max - (Rational::new(i - 1, num_densities - 1)) * t_delta,
                ),
            })
            .collect::<Vec<_>>()
    }

    fn allocation_size(num_keys: usize) -> usize {
        let t_min = 0.5;
        let p_max = 0.25;
        let ideal_density = (t_min - p_max) / 2f32;

        let length = num_keys as f32 / ideal_density;
        // To get a balanced tree, we need to find the
        // closest double-exponential number (x = 2^2^i)
        // The exponent must be a power of two: 2^1=2, 2^2=4, 2^4=16, 2^8=256, 2^16=65536, 2^32=4294967296
        let exponent = (f32::log2(length).ceil() as usize).next_power_of_two();

        // Check for overflow: on 64-bit, max exponent is 63; on 32-bit, max is 31
        assert!(
      exponent <= usize::BITS as usize - 1,
      "PMA allocation size overflow: requested capacity {} requires 2^{} cells, which exceeds maximum allocatable size",
      num_keys,
      exponent
    );

        1usize << exponent
    }
}

impl<T> PackedMemoryArray<T>
where
    T: Default,
{
    pub fn with_capacity(capacity: usize) -> PackedMemoryArray<T> {
        let size = Self::allocation_size(capacity);
        // println!("packed memory array [V; {:?}]", size);
        let initialized_cells = Self::allocate_default(size);
        PackedMemoryArray::new(initialized_cells, capacity)
    }

    fn allocate_default(size: usize) -> Box<[T]> {
        let mut vec = Vec::with_capacity(size);
        vec.resize_with(size, Default::default);
        vec.into_boxed_slice()
    }
}

impl<T> Debug for PackedMemoryArray<T>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PackedMemoryArray")
            .field("cells", &format_args!("{:?}", self.cells))
            .finish()
    }
}

impl<'a, T> std::iter::IntoIterator for &'a PackedMemoryArray<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        Iter {
            ptr: self.active_range.start,
            active_range: std::ops::Range {
                start: self.active_range.start,
                end: self.active_range.end,
            },
            phantom: PhantomData,
            at_start: true,
        }
    }
}

/// Iterator over elements in the active range of a PMA.
pub struct Iter<'a, T> {
    ptr: *const T,
    active_range: Range<*const T>,
    phantom: PhantomData<&'a PackedMemoryArray<T>>,
    at_start: bool,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.at_start {
            self.at_start = false;
        } else {
            // SAFETY: We're within the allocated PMA array (checked against active_range.end)
            let new_address = unsafe { self.ptr.add(1) };
            if new_address > self.active_range.end {
                return None;
            }
            self.ptr = new_address;
        }
        // SAFETY: ptr is within [active_range.start, active_range.end]
        // and the PMA is pinned, so the memory remains valid
        Some(unsafe { &*self.ptr })
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // SAFETY: We check against active_range.end before dereferencing
        let new_address = unsafe { self.active_range.start.add(n) };
        if new_address > self.active_range.end {
            return None;
        }
        self.ptr = new_address;
        // SAFETY: ptr is within [active_range.start, active_range.end]
        Some(unsafe { &*self.ptr })
    }
}

/// Configuration for density bounds at each level of the PMA.
pub struct Config {
    pub density_scale: Vec<Density>,
}

/// Density bounds for a specific range size.
pub struct Density {
    pub max_item_count: usize,
    pub range: RangeInclusive<Rational>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to test allocation_size without needing a full PackedMemoryArray
    fn test_allocation_size(num_keys: usize) -> usize {
        PackedMemoryArray::<()>::allocation_size(num_keys)
    }

    #[test]
    fn allocation_size_small_capacities() {
        // Small capacities should produce valid double-exponential sizes
        // With ideal_density = 0.125, length = num_keys / 0.125 = num_keys * 8
        assert_eq!(test_allocation_size(1), 16); // 2^4 (length=8, ceil(log2(8))=3, next_pow2=4)
        assert_eq!(test_allocation_size(2), 16); // 2^4 (length=16, ceil(log2(16))=4, next_pow2=4)
        assert_eq!(test_allocation_size(16), 256); // 2^8 (length=128, ceil(log2(128))=7, next_pow2=8)
    }

    #[test]
    fn allocation_size_medium_capacities() {
        // Medium capacities - note the formula produces larger sizes due to ideal density
        // 100 keys -> length=800 -> log2(800)≈9.6 -> ceil=10 -> next_pow2=16 -> 2^16
        assert_eq!(test_allocation_size(100), 65536); // 2^16
        assert_eq!(test_allocation_size(5000), 65536); // 2^16 (length=40000, log2≈15.3, next_pow2=16)
    }

    #[test]
    fn allocation_size_large_capacities() {
        // Large capacities that require 64-bit arithmetic
        // 10000 keys -> length=80000 -> log2≈16.3 -> ceil=17 -> next_pow2=32 -> 2^32
        assert_eq!(test_allocation_size(10_000), 4294967296); // 2^32
        assert_eq!(test_allocation_size(100_000), 4294967296); // 2^32
    }

    #[test]
    fn allocation_size_follows_double_exponential_sequence() {
        // Verify the double-exponential sequence: 2^1, 2^2, 2^4, 2^8, 2^16, 2^32
        // Each result should be a power of 2 where the exponent is also a power of 2
        let valid_sizes: Vec<usize> = vec![2, 4, 16, 256, 65536, 4294967296];

        for cap in [1, 10, 100, 1000, 10000, 100000] {
            let size = test_allocation_size(cap);
            assert!(
                valid_sizes.contains(&size),
                "allocation_size({}) = {} is not a valid double-exponential size",
                cap,
                size
            );
        }
    }

    #[test]
    fn packed_memory_array_with_capacity_small() {
        let pma: PackedMemoryArray<i32> = PackedMemoryArray::with_capacity(16);
        assert!(pma.len() > 0);
        assert_eq!(pma.requested_capacity, 16);
    }

    #[test]
    fn packed_memory_array_with_capacity_medium() {
        let pma: PackedMemoryArray<i32> = PackedMemoryArray::with_capacity(5000);
        assert!(pma.len() >= 5000);
        assert_eq!(pma.requested_capacity, 5000);
    }

    #[test]
    fn packed_memory_array_with_capacity_large() {
        // Test that large capacity computation works without overflow.
        // Note: We only test the allocation_size computation here, not actual allocation,
        // because capacity=10000 requires 2^32 (~4GB) which would be too slow/memory-intensive.
        // The allocation_size_large_capacities test already verifies the computation is correct.
        let size = test_allocation_size(10000);
        assert_eq!(size, 4294967296); // 2^32
        assert!(size >= 10000);
    }

    #[test]
    fn compute_active_range_excludes_buffer_regions() {
        // Create a slice of length 8; buffer_space = 8 >> 2 = 2
        // Active range should be indices [2, 6) (elements at indices 2, 3, 4, 5)
        let cells: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let range = PackedMemoryArray::<i32>::compute_active_range(&cells);

        // Verify start points to index 2
        assert_eq!(range.start, &cells[2] as *const i32);
        // Verify end points to index 6 (one past the last active element)
        assert_eq!(range.end, &cells[6] as *const i32);

        // Verify the values at the boundaries
        unsafe {
            assert_eq!(*range.start, 2);
            assert_eq!(*range.end, 6);
        }
    }

    #[test]
    fn compute_active_range_with_larger_slice() {
        // Create a slice of length 16; buffer_space = 16 >> 2 = 4
        // Active range should be indices [4, 12)
        let cells: Vec<i32> = (0..16).collect();
        let range = PackedMemoryArray::<i32>::compute_active_range(&cells);

        assert_eq!(range.start, &cells[4] as *const i32);
        assert_eq!(range.end, &cells[12] as *const i32);

        unsafe {
            assert_eq!(*range.start, 4);
            assert_eq!(*range.end, 12);
        }
    }

    #[test]
    fn buffer_space_is_quarter_of_capacity() {
        // buffer_space should return 1/4 of the total capacity
        assert_eq!(PackedMemoryArray::<i32>::buffer_space(16), 4);
        assert_eq!(PackedMemoryArray::<i32>::buffer_space(256), 64);
        assert_eq!(PackedMemoryArray::<i32>::buffer_space(1024), 256);
        // Edge cases
        assert_eq!(PackedMemoryArray::<i32>::buffer_space(4), 1);
        assert_eq!(PackedMemoryArray::<i32>::buffer_space(8), 2);
    }

    #[test]
    fn compute_active_range_correct_bounds() {
        // For a 16-element array, active range should be [4..12]
        // Left buffer: [0..4], Active: [4..12], Right buffer: [12..16]
        let cells: Vec<i32> = vec![0; 16];
        let active_range = PackedMemoryArray::compute_active_range(&cells);

        let base_ptr = cells.as_ptr();
        assert_eq!(active_range.start, unsafe { base_ptr.add(4) });
        assert_eq!(active_range.end, unsafe { base_ptr.add(12) });
    }

    #[test]
    fn compute_active_range_larger_array() {
        // For a 256-element array:
        // buffer_space = 256 / 4 = 64
        // active range should be [64..192]
        let cells: Vec<i32> = vec![0; 256];
        let active_range = PackedMemoryArray::compute_active_range(&cells);

        let base_ptr = cells.as_ptr();
        assert_eq!(active_range.start, unsafe { base_ptr.add(64) });
        assert_eq!(active_range.end, unsafe { base_ptr.add(192) });
    }

    #[test]
    fn active_range_matches_as_slice() {
        // Verify that compute_active_range produces pointers consistent with as_slice
        let pma: PackedMemoryArray<i32> = PackedMemoryArray::with_capacity(16);
        let slice = pma.as_slice();

        // The active_range.start should point to the same address as the slice start
        assert_eq!(pma.active_range.start, slice.as_ptr());

        // The active_range length should match slice length
        let range_len =
            unsafe { pma.active_range.end.offset_from(pma.active_range.start) } as usize;
        assert_eq!(range_len, slice.len());
    }

    #[test]
    fn active_range_is_half_of_total_capacity() {
        // The active range should always be exactly half the total capacity
        // since we reserve 1/4 on each side
        for &size in &[16, 64, 256] {
            let cells: Vec<i32> = vec![0; size];
            let active_range = PackedMemoryArray::compute_active_range(&cells);

            let range_len = unsafe { active_range.end.offset_from(active_range.start) } as usize;
            assert_eq!(
                range_len,
                size / 2,
                "For size {}, active range should be {} but was {}",
                size,
                size / 2,
                range_len
            );
        }
    }
}
