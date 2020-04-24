use super::packed_data::{Block, Element, PackedData};

use std::cell::RefCell;
use std::cmp::Ord;
use std::fmt::Debug;
use std::marker::Copy;
use std::rc::Rc;

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
