use num_rational::Rational;
use std::fmt::{self, Debug};
use std::marker::{PhantomData, PhantomPinned};
use std::ops::Range;
use std::ops::RangeInclusive;
use std::pin::Pin;

pub struct PackedMemoryArray<T> {
    cells: Pin<Box<[T]>>,
    pub config: Config,
    pub active_range: Range<*const T>,
    pub requested_capacity: usize,
    _pin: PhantomPinned,
}

unsafe impl<T> Send for PackedMemoryArray<T> {}
unsafe impl<T> Sync for PackedMemoryArray<T> {}

impl<T> PackedMemoryArray<T> {
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

    pub fn as_slice(&self) -> &[T] {
        let left_buffer_space = self.cells.len() >> 2;
        &self.cells[left_buffer_space..self.cells.len() - left_buffer_space]
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_valid_pointer(&self, ptr: &*const T) -> bool {
        self.active_range.contains(ptr)
    }
    /// Computes the active range of a slice, excluding buffer regions on both ends.
    /// The buffer space is 1/4 of the total length on each side.
    fn compute_active_range(cells: &[T]) -> Range<*const T> {
        let buffer_space = cells.len() >> 2;
        Range {
            start: &cells[buffer_space] as *const _,
            end: &cells[cells.len() - buffer_space] as *const _,
        }
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
            let new_address = unsafe { self.ptr.add(1) };
            if new_address > self.active_range.end {
                return None;
            }
            self.ptr = new_address;
        }
        Some(unsafe { &*self.ptr })
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_address = unsafe { self.active_range.start.add(n) };
        if new_address > self.active_range.end {
            return None;
        }
        self.ptr = new_address;
        Some(unsafe { &*self.ptr })
    }
}
pub struct Config {
    pub density_scale: Vec<Density>,
}

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
}
