use std::ops::Range;

pub struct PackedMemoryArray<T: Default> {
  cells: Box<[T]>,
  value_count: u32,
  pub active_range: Range<*const T>
}

impl <'a, T> std::iter::IntoIterator for &'a PackedMemoryArray<T> where T: Default {
  type Item = &'a T;
  type IntoIter = Iter<'a, T>;

  fn into_iter(self) -> Iter<'a, T> {
    Iter { pma: self, ptr: self.active_range.start, max_ptr: self.active_range.end }
  }
}

pub struct Iter<'a, T: Default> {
  pma: &'a PackedMemoryArray<T>,
  ptr: *const T,
  max_ptr: *const T,
}

impl <'a, T> Iterator for Iter<'a, T> where T: Default {
  type Item = &'a T;
  fn next(&mut self) -> Option<&'a T> {
    self.ptr = unsafe { self.ptr.add(1) };
    if self.ptr > self.max_ptr {
      return None;
    }
    Some(unsafe { &*self.ptr })
  }
}

impl <T> PackedMemoryArray<T> where T: Default {
  pub fn new(values_to_hold: u32, cells: Box<[T]>) -> PackedMemoryArray<T> {
    let left_buffer_space = cells.len() >> 2;

    // TODO: Generalize this
    let active_range = std::ops::Range {
      start: &cells[left_buffer_space] as *const _,
      end: &cells[cells.len() - left_buffer_space] as *const _,
    };

    PackedMemoryArray {
      cells,
      value_count: values_to_hold,
      active_range,
    }
  }

  pub fn with_capacity(capacity: u32) -> PackedMemoryArray<T> {
    let size = Self::values_mem_size(capacity);
    // println!("packed memory array [V; {:?}]", size);
    PackedMemoryArray::new(capacity, Self::init_cells(size as usize))
  }

  pub fn as_slice(&self) -> &[T] {
    let left_buffer_space = self.cells.len() >> 2;
    &self.cells[left_buffer_space..self.cells.len() - left_buffer_space]
  }

  pub fn chunks_for_leaves(&mut self) -> impl Iterator<Item = &mut [T]> {
    let size = self.value_count;
    let slot_size = f32::log2(size as f32) as usize; // https://github.com/rust-lang/rust/issues/70887
    let left_buffer_space = self.cells.len() >> 2;
    let left_buffer_slots = left_buffer_space / slot_size;
    self.cells.chunks_exact_mut(slot_size).skip(left_buffer_slots)
  }

  pub fn len(&self) -> usize {
    self.cells.len()
  }

  fn init_cells(size: usize) -> Box<[T]> {
    let mut vec = Vec::with_capacity(size);
    vec.resize_with(size, Default::default);
    vec.into_boxed_slice()
  }

  fn values_mem_size(num_keys: u32) -> u32 {
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