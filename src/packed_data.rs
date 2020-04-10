use std::ptr::{NonNull};
use std::alloc::{alloc, dealloc, Layout};

#[derive(Debug)]
pub struct PackedData<T> {
  ptr: NonNull<T>,
  length: usize,
}

impl <T> PackedData<T> {
  pub fn new(size: usize) -> Self {
    PackedData {
      length: size,
      ptr: unsafe {
        let layout = Layout::array::<T>(size);
        let ptr = alloc(layout.unwrap()) as *mut T;

        NonNull::new_unchecked(ptr)
      },
    }
  }

  pub fn set(&mut self, index: usize, value: T) {
    assert!(index < self.length);
    unsafe {
      let ptr = self.ptr.as_ptr();
      *(ptr.add(index)) = value;
    }
  }

  pub fn get(&self, index: usize) -> &T {
    unsafe {
      let ptr = self.ptr.as_ptr();
      &*(ptr.add(index))
    }
  }
}

impl <T> Drop for PackedData<T> {
  fn drop(&mut self) {
    unsafe {
      let layout = Layout::array::<T>(self.length);
      dealloc(self.ptr.as_ptr() as *mut u8, layout.unwrap())
    }
  }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut data = PackedData::new(4);
        data.set(2, "Hello");
        assert_eq!(data.get(2), &"Hello");
    }
}

