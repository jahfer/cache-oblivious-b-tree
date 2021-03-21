use std::ops::Range;
use std::marker::{PhantomData, PhantomPinned};
use std::fmt::{self, Debug};
use std::pin::Pin;
use std::ops::RangeInclusive;
use num_rational::Rational;

pub struct PackedMemoryArray<T> {
  cells: Pin<Box<[T]>>,
  pub config: Config,
  pub active_range: Range<*const T>,
  pub requested_capacity: u32, // todo: Temporary...
  _pin: PhantomPinned
}

unsafe impl <T> Send for PackedMemoryArray<T> {}
unsafe impl <T> Sync for PackedMemoryArray<T> {}

impl <T> PackedMemoryArray<T> {
  pub fn new(cells: Box<[T]>, capacity: u32) -> PackedMemoryArray<T> {
    let left_buffer_space = cells.len() >> 2;

    // TODO: Generalize this
    let active_range = std::ops::Range {
      start: &cells[left_buffer_space] as *const _,
      end: &cells[cells.len() - left_buffer_space] as *const _,
    };

    let density_scale = Self::compute_density_range(cells.len() as f32);
    let config = Config { density_scale };

    PackedMemoryArray {
      cells: Box::into_pin(cells),
      requested_capacity: capacity,
      active_range,
      config,
      _pin: PhantomPinned
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

  fn allocation_size(num_keys: u32) -> u32 {
    let t_min = 0.5;
    let p_max = 0.25;
    let ideal_density = (t_min - p_max) / 2f32;

    let length = num_keys as f32 / ideal_density;
    // To get a balanced tree, we need to find the
    // closest double-exponential number (x = 2^2^i)
    let clean_length = 2 << ((f32::log2(length).ceil() as u32).next_power_of_two() - 1);
    clean_length
  }
}

impl <T> PackedMemoryArray<T> where T: Default {
  pub fn with_capacity(capacity: u32) -> PackedMemoryArray<T> {
    let size = Self::allocation_size(capacity);
    // println!("packed memory array [V; {:?}]", size);
    let initialized_cells = Self::allocate_default(size as usize);
    PackedMemoryArray::new(initialized_cells, capacity)
  }

  fn allocate_default(size: usize) -> Box<[T]> {
    let mut vec = Vec::with_capacity(size);
    vec.resize_with(size, Default::default);
    vec.into_boxed_slice()
  }
}

impl <T> Debug for PackedMemoryArray<T> where T: Debug {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("PackedMemoryArray")
      .field("cells", &format_args!("{:?}", self.cells))
      .finish()
  }
}

impl <'a, T> std::iter::IntoIterator for &'a PackedMemoryArray<T> {
  type Item = &'a T;
  type IntoIter = Iter<'a, T>;

  fn into_iter(self) -> Iter<'a, T> {
    Iter {
      ptr: self.active_range.start,
      active_range: std::ops::Range { start: self.active_range.start, end: self.active_range.end },
      phantom: PhantomData,
      at_start: true
    }
  }
}
pub struct Iter<'a, T> {
  ptr: *const T,
  active_range: Range<*const T>,
  phantom: PhantomData<&'a PackedMemoryArray<T>>,
  at_start: bool 
}

impl <'a, T> Iterator for Iter<'a, T> {
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
    if new_address> self.active_range.end {
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