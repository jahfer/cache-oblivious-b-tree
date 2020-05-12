#![feature(new_uninit)]
#![feature(box_into_pin)]
#![feature(maybe_uninit_ref)]
#![feature(maybe_uninit_extra)]
#![feature(maybe_uninit_slice_assume_init)]

// #[macro_use]
// extern crate memoffset;
// extern crate alloc;

mod cache_oblivious;
pub use cache_oblivious::StaticSearchTree;
// pub use cache_oblivious::{CacheObliviousBTreeMap, PackedData, StaticSearchTree};

#[cfg(test)]
mod tests {
  use crate::StaticSearchTree;

  #[test]
  fn test() {
    let tree = StaticSearchTree::<u8, &str>::new(30);
    let leaf_block = tree.sample();
    println!("leaf: {:?}", leaf_block.get(1));
  }

  // #[test]
  // fn packed_data_blocks() {
  //   let mut data = PackedData::new(32);
  //   let block1 = data.set(0, 1, "Hello");
  //   let _block2 = data.set(1, 72, "World");
  //   block1.insert(2, "Goodbye");
  // }

  // #[test]
  // fn insert_and_get() {
  //   let mut map = CacheObliviousBTreeMap::new(32);
  //   map.insert(3, "Hello");
  //   map.insert(8, "World");
  //   map.insert(12, "!");

  //   assert_eq!(map.get(3), Some("Hello"));
  //   assert_eq!(map.get(4), None);
  //   assert_eq!(map.get(8), Some("World"));
  //   assert_eq!(map.get(12), Some("!"));
  // }

  // #[test]
  // fn move_cells() {
  //   let mut map = CacheObliviousBTreeMap::new(32);
  //   map.insert(5, "Hello");
  //   map.insert(3, "World");
  //   map.insert(2, "!");

  //   assert_eq!(map.get(5), Some("Hello"));
  //   assert_eq!(map.get(4), None);
  //   assert_eq!(map.get(3), Some("World"));
  //   assert_eq!(map.get(2), Some("!"));
  // }
}
