#![feature(new_uninit)]
#![feature(maybe_uninit_ref)]
#![feature(maybe_uninit_extra)]
#![feature(maybe_uninit_slice)]
#![feature(option_insert)]
#![feature(once_cell)]

mod cache_oblivious;
pub use cache_oblivious::StaticSearchTree;

#[cfg(test)]
mod tests {
  use crate::StaticSearchTree;

  #[test]
  fn find_missing() {
    let tree = StaticSearchTree::<u8, &str>::new(3);
    assert_eq!(tree.find(4), None);
  }

  #[test]
  fn add_existing() {
    let mut tree = StaticSearchTree::<u8, &str>::new(3);
    tree.add(5, "Test");
    tree.add(5, "Double");
  }

  #[test]
  fn add_ordered_values() {
    let mut tree = StaticSearchTree::<u8, &str>::new(3);

    tree.add(3, "Hello");
    tree.add(8, "World");
    tree.add(12, "!");

    assert_eq!(tree.find(3), Some("Hello"));
    assert_eq!(tree.find(8), Some("World"));
    assert_eq!(tree.find(12), Some("!"));
  }

  #[test]
  fn add_unordered_values() {
    let mut tree = StaticSearchTree::<u8, &str>::new(16);
    tree.add(5, "Hello");
    tree.add(3, "World");
    tree.add(2, "!");

    assert_eq!(tree.find(5), Some("Hello"));
    assert_eq!(tree.find(4), None);
    assert_eq!(tree.find(3), Some("World"));
    assert_eq!(tree.find(2), Some("!"));
  }
}
