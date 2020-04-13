#![feature(new_uninit)]
#![feature(alloc_layout_extra)]

extern crate alloc;

use std::cell::RefCell;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::fmt::Debug;
use std::marker::Copy;
use std::rc::Rc;

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::{copy, NonNull};

#[derive(Debug)]
pub struct PackedData<T> {
  data: NonNull<T>,
  capacity: usize,
  length: usize,
  block_length: usize,
}

impl<T> PackedData<T> {
  pub fn new(capacity: usize) -> Self {
    PackedData {
      capacity: capacity,
      length: 0,
      block_length: (capacity as f32).log2().ceil() as usize,
      data: unsafe {
        let layout = Layout::array::<T>(capacity);
        let ptr = alloc(layout.unwrap()) as *mut T;

        NonNull::new_unchecked(ptr)
      },
    }
  }

  // pub fn find_in_block(&self, index: usize) -> &T {
  //   assert!(index < self.capacity);
  // }

  pub fn insert_at(&mut self, index: usize, value: T) -> usize {
    assert!(index < self.capacity);
    unsafe {
      let ptr = self.data.as_ptr();
      *(ptr.add(index)) = value;
    }
    self.length += 1;
    index
  }

  pub fn get(&self, index: usize) -> &T {
    unsafe {
      let ptr = self.data.as_ptr();
      &*(ptr.add(index))
    }
  }
}

impl<T> Drop for PackedData<T> {
  fn drop(&mut self) {
    unsafe {
      let layout = Layout::array::<T>(self.capacity);
      dealloc(self.data.as_ptr() as *mut u8, layout.unwrap())
    }
  }
}

#[derive(Debug)]
enum Node<K: Ord + Eq> {
  Branch(BinaryTree<K>),
  Leaf { key: K, block_data: usize },
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
enum Element<T: Eq> {
  Infimum,
  Value(T),
  Supremum,
}

impl<T: Eq + Ord> Ord for Element<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    match (self, other) {
      (x, y) if x == y => Ordering::Equal,
      (Element::Infimum, _) | (_, Element::Supremum) => Ordering::Less,
      (Element::Supremum, _) | (_, Element::Infimum) => Ordering::Greater,
      (Element::Value(a), Element::Value(b)) => a.cmp(b),
    }
  }
}

#[derive(Debug)]
struct BinaryTree<K: Eq + Ord> {
  key: Element<K>,
  left: Rc<RefCell<Option<Node<K>>>>,
  right: Rc<RefCell<Option<Node<K>>>>,
}

impl<K: Eq + Ord> BinaryTree<K> {
  pub fn new() -> BinaryTree<K> {
    BinaryTree {
      key: Element::Infimum,
      left: Rc::new(RefCell::new(None)),
      right: Rc::new(RefCell::new(None)),
    }
  }
}

fn convert_leaf_to_branch<K: Copy + Ord>(leaf: &Node<K>, new_key: &K) -> Node<K> {
  match leaf {
    Node::Leaf { key, block_data } => {
      let new_cell = Rc::new(RefCell::new(None));
      let tree = if new_key > key {
        BinaryTree {
          key: Element::Value(*new_key),
          left: Rc::new(RefCell::new(Some(Node::Leaf {
            key: *key,
            block_data: *block_data,
          }))),
          right: Rc::clone(&new_cell),
        }
      } else {
        BinaryTree {
          key: Element::Value(*key),
          left: Rc::clone(&new_cell),
          right: Rc::new(RefCell::new(Some(Node::Leaf {
            key: *key,
            block_data: *block_data,
          }))),
        }
      };

      Node::Branch(tree)
    }
    Node::Branch(_) => unreachable!(),
  }
}

impl<K: Copy + Ord + Debug> BinaryTree<K> {
  pub fn search(&self, search_key: Element<K>) -> Option<usize> {
    if search_key < self.key {
      match &*self.left.borrow() {
        Some(Node::Leaf { key, block_data }) => {
          if Element::Value(*key) == search_key {
            Some(*block_data)
          } else {
            None
          }
        }
        Some(Node::Branch(tree)) => tree.search(search_key),
        None => None,
      }
    } else {
      match &*self.right.borrow() {
        Some(Node::Leaf { key, block_data }) => {
          if Element::Value(*key) == search_key {
            Some(*block_data)
          } else {
            None
          }
        }
        Some(Node::Branch(tree)) => tree.search(search_key),
        None => None,
      }
    }
  }

  fn search_closest(&self, search_key: Element<K>) -> Option<usize> {
    if search_key < self.key {
      match &*self.left.borrow() {
        Some(Node::Leaf { key, block_data }) => {
          if search_key < Element::Value(*key) {
            None
          } else {
            Some(*block_data)
          }
        }
        Some(Node::Branch(tree)) => tree.search_closest(search_key),
        None => None,
      }
    } else {
      match &*self.right.borrow() {
        Some(Node::Leaf { key: _, block_data }) => Some(*block_data),
        Some(Node::Branch(tree)) => tree.search_closest(search_key),
        None => None,
      }
    }
  }

  fn fetch_insertion_cell(&self, insertion_key: K) -> Rc<RefCell<Option<Node<K>>>> {
    if Element::Value(insertion_key) < self.key {
      let mut left_entry = self.left.borrow_mut();

      match &*left_entry {
        None => Rc::clone(&self.left),
        Some(entry) => match entry {
          Node::Leaf {
            key: _,
            block_data: _,
          } => {
            let branch = convert_leaf_to_branch(entry, &insertion_key);
            let cell = match &branch {
              Node::Branch(tree) => Rc::clone(&tree.left),
              Node::Leaf {
                key: _,
                block_data: _,
              } => unreachable!(),
            };
            left_entry.replace(branch);
            cell
          }
          Node::Branch(tree) => tree.fetch_insertion_cell(insertion_key),
        },
      }
    } else {
      let mut right_entry = self.right.borrow_mut();
      match &*right_entry {
        None => Rc::clone(&self.right),
        Some(entry) => match entry {
          Node::Leaf { key, block_data: _ } if *key == insertion_key => Rc::clone(&self.right),
          Node::Leaf {
            key: _,
            block_data: _,
          } => {
            let branch = convert_leaf_to_branch(entry, &insertion_key);
            let cell = match &branch {
              Node::Branch(tree) => {
                if Element::Value(insertion_key) == tree.key {
                  Rc::clone(&tree.right)
                } else {
                  Rc::clone(&tree.left)
                }
              }
              Node::Leaf {
                key: _,
                block_data: _,
              } => unreachable!(),
            };
            right_entry.replace(branch);
            cell
          }
          Node::Branch(tree) => tree.fetch_insertion_cell(insertion_key),
        },
      }
    }
  }
}

#[derive(Debug)]
pub struct CacheObliviousBTreeMap<K: Ord + Eq, V> {
  packed_dataset: PackedData<V>,
  tree: BinaryTree<K>,
  min_key: Element<K>,
  max_key: Element<K>,
}

impl<K: Ord + Eq, V> CacheObliviousBTreeMap<K, V> {
  pub fn new(size: usize) -> Self {
    CacheObliviousBTreeMap {
      packed_dataset: PackedData::new(size),
      tree: BinaryTree::new(),
      min_key: Element::Infimum,
      max_key: Element::Supremum,
    }
  }
}

impl<K: Copy + Debug + Ord, V: Copy + Debug> CacheObliviousBTreeMap<K, V> {
  pub fn get(&self, key: K) -> Option<&V> {
    self
      .tree
      .search(Element::Value(key))
      .map(|idx| self.packed_dataset.get(idx))
  }

  pub fn insert(&mut self, key: K, value: V) {
    match self.tree.search_closest(Element::Value(key)) {
      Some(i) => {
        println!("Closest block {:?} found for key {:?}", i, key);
        let inserted_index = self.packed_dataset.insert_at(i + 1, value);
        let cell = self.tree.fetch_insertion_cell(key);
        let new_leaf = Node::Leaf {
          key,
          block_data: inserted_index,
        };
        cell.replace(Some(new_leaf));
      }
      None => {
        println!("No closest block found for key {:?}", key);
        let inserted_index = self.packed_dataset.insert_at(0, value);
        let cell = self.tree.fetch_insertion_cell(key);
        let new_leaf = Node::Leaf {
          key,
          block_data: inserted_index,
        };
        cell.replace(Some(new_leaf));
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn it_works() {
    let mut map = CacheObliviousBTreeMap::new(32);
    map.insert(5, "Hello");
    map.insert(3, "World");
    map.insert(2, "!");

    assert_eq!(map.get(5), Some(&"Hello"));
    assert_eq!(map.get(4), None);
    assert_eq!(map.get(3), Some(&"World"));
    assert_eq!(map.get(2), Some(&"!"));
  }

  #[test]
  fn it_still_works() {
    let mut map = CacheObliviousBTreeMap::new(32);
    map.insert(3, "Hello");
    map.insert(8, "World");
    map.insert(12, "!");

    assert_eq!(map.get(3), Some(&"Hello"));
    assert_eq!(map.get(4), None);
    assert_eq!(map.get(8), Some(&"World"));
    assert_eq!(map.get(12), Some(&"!"));
  }

  #[test]
  fn blocks() {
    let mut map: CacheObliviousBTreeMap<i32, &str> = CacheObliviousBTreeMap::new(16);
    map.insert(3, "Hello");
  }
}
