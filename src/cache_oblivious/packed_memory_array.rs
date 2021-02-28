use std::{marker::PhantomData, ops::Range};

pub struct PackedMemoryArray<T> {
  cells: Box<[T]>,
  pub active_range: Range<*const T>,
}

impl <T> PackedMemoryArray<T> {
  pub fn new(cells: Box<[T]>) -> PackedMemoryArray<T> {
    let left_buffer_space = cells.len() >> 2;

    // TODO: Generalize this
    let active_range = std::ops::Range {
      start: &cells[left_buffer_space] as *const _,
      end: &cells[cells.len() - left_buffer_space] as *const _,
    };

    PackedMemoryArray { cells, active_range }
  }

  pub fn as_slice(&self) -> &[T] {
    let left_buffer_space = self.cells.len() >> 2;
    &self.cells[left_buffer_space..self.cells.len() - left_buffer_space]
  }

  pub fn len(&self) -> usize {
    self.cells.len()
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
    PackedMemoryArray::new(initialized_cells)
  }

  fn allocate_default(size: usize) -> Box<[T]> {
    let mut vec = Vec::with_capacity(size);
    vec.resize_with(size, Default::default);
    vec.into_boxed_slice()
  }
}


impl <'a, T> std::iter::IntoIterator for &'a PackedMemoryArray<T> {
  type Item = &'a T;
  type IntoIter = Iter<'a, T>;

  fn into_iter(self) -> Iter<'a, T> {
    Iter {
      ptr: self.active_range.start,
      max_ptr: self.active_range.end,
      phantom: PhantomData
    }
  }
}
pub struct Iter<'a, T> {
  ptr: *const T,
  max_ptr: *const T,
  phantom: PhantomData<&'a PackedMemoryArray<T>>
}

impl <'a, T> Iterator for Iter<'a, T> {
  type Item = &'a T;
  fn next(&mut self) -> Option<&'a T> {
    self.ptr = unsafe { self.ptr.add(1) };
    if self.ptr > self.max_ptr {
      return None;
    }
    Some(unsafe { &*self.ptr })
  }
}