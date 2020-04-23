#![feature(alloc_layout_extra)]

#[macro_use]
extern crate memoffset;

extern crate alloc;

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::fmt::{self, Debug};
use std::marker::Copy;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU16, Ordering as AtomicOrdering};
use std::sync::Arc;

const COUNTER_INIT_VALUE: u16 = 1;

#[derive(Debug)]
enum Marker<K, V> {
  Empty(u16),
  Move(u16, K, V),
  InsertCell(u16, K, V),
  DeleteCell(u16, K),
}

struct CellInner<K, V> {
  version: u16,
  marker: Marker<K, V>,
  empty: bool,
  key: K,
  value: V,
}

pub struct Cell<K, V> {
  version: NonNull<AtomicU16>,
  marker: AtomicPtr<Marker<K, V>>,
  empty: NonNull<AtomicBool>,
  key: AtomicPtr<K>,
  value: AtomicPtr<V>,
}

impl<K: Copy, V: Copy> Cell<K, V> {
  pub fn key(&self) -> Option<K> {
    if self.is_empty() {
      None
    } else {
      Some(self.get_key())
    }
  }

  pub fn value(&self) -> Option<V> {
    if self.is_empty() {
      None
    } else {
      Some(self.get_value())
    }
  }

  fn is_empty(&self) -> bool {
    let empty_ref = unsafe { &*self.empty.as_ref() };
    empty_ref.load(AtomicOrdering::Acquire)
  }

  fn get_value(&self) -> V {
    unsafe { *self.value.load(AtomicOrdering::Acquire) }
  }

  fn get_key(&self) -> K {
    unsafe { *self.key.load(AtomicOrdering::Acquire) }
  }
}

impl<K: Debug, V: Debug> Debug for Cell<K, V> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let version_ref = unsafe { self.version.as_ref() };
    let empty_ref = unsafe { self.empty.as_ref() };

    let version = version_ref.load(AtomicOrdering::Acquire);
    let marker = unsafe { &*self.marker.load(AtomicOrdering::Acquire) };
    let empty = empty_ref.load(AtomicOrdering::Acquire);
    let key = unsafe { &*self.key.load(AtomicOrdering::Acquire) };
    let value = unsafe { &*self.value.load(AtomicOrdering::Acquire) };

    let mut dbg_struct = f.debug_struct("Cell");

    dbg_struct
      .field("version", &version)
      .field("marker", marker)
      .field("empty", &empty);

    if empty {
      let none: Option<K> = Option::None;
      dbg_struct.field("key", &none).field("value", &none);
    } else {
      dbg_struct.field("key", key).field("value", value);
    }

    dbg_struct.finish()
  }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy)]
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
pub struct Block<K: Eq, V> {
  min_key: Element<K>, // TODO: use AtomicPtr for update?
  max_key: Element<K>,
  cells: Arc<[Cell<K, V>]>,
}

impl<K: Eq + Copy, V> Clone for Block<K, V> {
  fn clone(&self) -> Self {
    Block {
      min_key: self.min_key,
      max_key: self.max_key,
      cells: Arc::clone(&self.cells),
    }
  }
}

impl<K: Eq + Copy + Debug, V: Copy + Debug> Block<K, V> {
  pub fn get(&self, key: K) -> Option<V> {
    for cell in self.cells.iter() {
      match cell.key() {
        Some(k) if k == key => return cell.value(),
        _ => continue,
      }
    }
    None
  }
}

impl<K: Eq + PartialOrd + Copy + Debug, V: Copy + Debug> Block<K, V> {
  pub fn insert<'a>(&self, key: K, value: V) -> Block<K, V> {
    // default to using first slot
    let mut insertion_cell: &Cell<K, V> = &self.cells[0];

    let replace_min_key = false; // self.min_key == Element::Infimum || self.min_key > Element::Value(key);

    if !replace_min_key {
      for cell in self.cells.iter() {
        if cell.is_empty() {
          insertion_cell = cell;
          break;
        }
        let cell_key = cell.key().unwrap();
        if cell_key < key {
          insertion_cell = cell;
        } else {
          break;
        }
      }
    }

    loop {
      let version_ref = unsafe { insertion_cell.version.as_ref() };
      let empty_ref = unsafe { insertion_cell.empty.as_ref() };
      let marker_ptr = insertion_cell.marker.load(AtomicOrdering::Acquire);

      let marker = unsafe { &mut *marker_ptr };
      let current_version = version_ref.load(AtomicOrdering::Acquire);

      match *marker {
        Marker::Empty(version) => {
          if version < current_version {
            println!("Marker out of date. Another process has claimed this cell.");
            continue;
          } else if version > current_version {
            panic!("This should never happen...");
          }
        }
        _ => unimplemented!("TODO: Complete existing marker..."),
      };

      let empty = empty_ref.load(AtomicOrdering::Acquire);
      if !empty {
        // unimplemented!("Cell occupied!");
      }

      let key_slot = insertion_cell.key.load(AtomicOrdering::Acquire);
      let value_slot = insertion_cell.value.load(AtomicOrdering::Acquire);

      let new_version = current_version + 1;

      // attempt to bump version, "claiming" it
      match version_ref.compare_exchange(
        current_version,
        new_version,
        AtomicOrdering::AcqRel,
        AtomicOrdering::Acquire,
      ) {
        Ok(_) => {
          *marker = Marker::InsertCell(new_version, key, value);

          unsafe {
            key_slot.write(key);
            value_slot.write(value);
          };

          empty_ref.store(false, AtomicOrdering::Release);

          *marker = Marker::Empty(new_version);
          break;
        }
        Err(_) => {
          println!("Version has changed since process began. Restarting.");
          continue;
        }
      }
    }

    let min_key = if replace_min_key {
      Element::Value(key)
    } else {
      self.min_key
    };

    Block {
      min_key,
      max_key: self.max_key,
      cells: Arc::clone(&self.cells),
    }
  }
}

pub struct PackedData<K: Eq + Sized, V: Sized> {
  data: NonNull<CellInner<K, V>>,
  capacity: usize,
  block_length: usize,
}

impl<K: Eq + Debug + PartialOrd + Copy, V: Debug + Copy> PackedData<K, V> {
  pub fn new(capacity: usize) -> Self {
    let block_length = (capacity as f32).log2().ceil() as usize;
    PackedData {
      capacity,
      block_length,
      data: unsafe {
        let layout = Layout::array::<CellInner<K, V>>(capacity);
        let ptr = alloc(layout.unwrap()) as *mut CellInner<K, V>;
        NonNull::new_unchecked(ptr)
      },
    }
  }

  fn initialize_block(&self, block_index: usize) -> Block<K, V> {
    assert!(block_index < self.block_length);
    unsafe {
      let block_ptr = self.data.as_ptr().add(self.block_length * block_index);
      let mut vec = Vec::with_capacity(self.block_length);

      for i in 0..self.block_length {
        let cell_inner = block_ptr.add(i);

        let field_version = raw_field!(cell_inner, CellInner<K,V>, version) as *mut AtomicU16;
        let field_marker = raw_field!(cell_inner, CellInner<K,V>, marker) as *mut Marker<K, V>;
        let field_empty = raw_field!(cell_inner, CellInner<K,V>, empty) as *mut AtomicBool;
        let field_key = raw_field!(cell_inner, CellInner<K,V>, key) as *mut K;
        let field_value = raw_field!(cell_inner, CellInner<K,V>, value) as *mut V;

        field_version.write(AtomicU16::new(COUNTER_INIT_VALUE));

        let marker = Marker::Empty(COUNTER_INIT_VALUE);
        field_marker.write(marker);

        field_empty.write(AtomicBool::new(true));

        vec.push(Cell {
          version: NonNull::new_unchecked(field_version),
          marker: AtomicPtr::new(field_marker),
          empty: NonNull::new_unchecked(field_empty),
          key: AtomicPtr::new(field_key),
          value: AtomicPtr::new(field_value),
        });
      }

      Block {
        min_key: Element::Infimum,
        max_key: Element::Supremum,
        // prev_block:,
        // next_block:,
        cells: Arc::from(vec),
      }
    }
  }

  pub fn set(&mut self, index: usize, key: K, value: V) -> Block<K, V> {
    let block = self.initialize_block(index);
    block.insert(key, value);
    block
  }
}

impl<K: Eq, V> Drop for PackedData<K, V> {
  fn drop(&mut self) {
    unsafe {
      let layout = Layout::array::<CellInner<K, V>>(self.capacity);
      dealloc(self.data.as_ptr() as *mut u8, layout.unwrap())
    }
  }
}

#[derive(Debug)]
enum Node<K: Ord + Eq, V> {
  Branch(BinaryTree<K, V>),
  Leaf { key: K, block_data: Block<K, V> },
}

#[derive(Debug)]
struct BinaryTree<K: Eq + Ord, V> {
  key: Element<K>,
  left: Rc<RefCell<Option<Node<K, V>>>>,
  right: Rc<RefCell<Option<Node<K, V>>>>,
}

impl<K: Eq + Ord, V> BinaryTree<K, V> {
  pub fn new() -> BinaryTree<K, V> {
    BinaryTree {
      key: Element::Infimum,
      left: Rc::new(RefCell::new(None)),
      right: Rc::new(RefCell::new(None)),
    }
  }
}

fn convert_leaf_to_branch<K: Copy + Ord, V>(leaf: &Node<K, V>, new_key: &K) -> Node<K, V> {
  match leaf {
    Node::Leaf { key, block_data } => {
      let new_cell = Rc::new(RefCell::new(None));
      let tree = if new_key > key {
        BinaryTree {
          key: Element::Value(*new_key),
          left: Rc::new(RefCell::new(Some(Node::Leaf {
            key: *key,
            block_data: block_data.clone(),
          }))),
          right: Rc::clone(&new_cell),
        }
      } else {
        BinaryTree {
          key: Element::Value(*key),
          left: Rc::clone(&new_cell),
          right: Rc::new(RefCell::new(Some(Node::Leaf {
            key: *key,
            block_data: block_data.clone(),
          }))),
        }
      };

      Node::Branch(tree)
    }
    Node::Branch(_) => unreachable!(),
  }
}

impl<K: Copy + Ord + Debug, V> BinaryTree<K, V> {
  pub fn search(&self, search_key: Element<K>) -> Option<Block<K, V>> {
    if search_key < self.key {
      match &*self.left.borrow() {
        Some(Node::Leaf { key, block_data }) => {
          if Element::Value(*key) == search_key {
            Some(block_data.clone())
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
            Some(block_data.clone())
          } else {
            None
          }
        }
        Some(Node::Branch(tree)) => tree.search(search_key),
        None => None,
      }
    }
  }

  fn search_closest(&self, search_key: Element<K>) -> Option<Block<K, V>> {
    if search_key < self.key {
      match &*self.left.borrow() {
        Some(Node::Leaf { key, block_data }) => {
          if search_key < Element::Value(*key) {
            None
          } else {
            Some(block_data.clone())
          }
        }
        Some(Node::Branch(tree)) => tree.search_closest(search_key),
        None => None,
      }
    } else {
      match &*self.right.borrow() {
        Some(Node::Leaf { key: _, block_data }) => Some(block_data.clone()),
        Some(Node::Branch(tree)) => tree.search_closest(search_key),
        None => None,
      }
    }
  }

  fn fetch_insertion_cell(&self, insertion_key: K) -> Rc<RefCell<Option<Node<K, V>>>> {
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

pub struct CacheObliviousBTreeMap<K: Ord + Eq, V> {
  packed_dataset: PackedData<K, V>,
  tree: BinaryTree<K, V>,
}

impl<K: Ord + Eq + Copy + Debug, V: Copy + Debug> CacheObliviousBTreeMap<K, V> {
  pub fn new(size: usize) -> Self {
    CacheObliviousBTreeMap {
      packed_dataset: PackedData::new(size),
      tree: BinaryTree::new(),
    }
  }
}

impl<K: Copy + Debug + Ord, V: Copy + Debug> CacheObliviousBTreeMap<K, V> {
  pub fn get(&self, key: K) -> Option<V> {
    self
      .tree
      .search(Element::Value(key))
      .and_then(|block| block.get(key))
  }

  pub fn insert(&mut self, key: K, value: V) {
    match self.tree.search_closest(Element::Value(key)) {
      Some(block) => {
        let new_block = block.insert(key, value);
        // TODO swap block with new_block in tree
        let cell = self.tree.fetch_insertion_cell(key);
        let new_leaf = Node::Leaf {
          key,
          block_data: block, // todo this doesn't seem right at all...
        };
        cell.replace(Some(new_leaf));
      }
      None => {
        let inserted_index = self.packed_dataset.set(0, key, value);
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
  fn packed_data_blocks() {
    let mut data = PackedData::new(32);
    let block1 = data.set(0, 1, "Hello");
    let _block2 = data.set(1, 72, "World");
    block1.insert(2, "Goodbye");
  }

  #[test]
  fn insert_and_get() {
    let mut map = CacheObliviousBTreeMap::new(32);
    map.insert(3, "Hello");
    map.insert(8, "World");
    map.insert(12, "!");

    assert_eq!(map.get(3), Some("Hello"));
    assert_eq!(map.get(4), None);
    assert_eq!(map.get(8), Some("World"));
    assert_eq!(map.get(12), Some("!"));
  }

  #[test]
  fn move_cells() {
    let mut map = CacheObliviousBTreeMap::new(32);
    map.insert(5, "Hello");
    map.insert(3, "World");
    map.insert(2, "!");

    assert_eq!(map.get(5), Some("Hello"));
    assert_eq!(map.get(4), None);
    assert_eq!(map.get(3), Some("World"));
    assert_eq!(map.get(2), Some("!"));
  }
}
