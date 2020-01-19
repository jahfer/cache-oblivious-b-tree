#![feature(new_uninit)]

use std::fmt::Debug;
use std::marker::Copy;
use std::mem::MaybeUninit;
use std::cmp::Ord;
use std::rc::Rc;
use std::cell::RefCell;
use std::ptr;

#[derive(Debug)]
enum BinaryTreeEntry<K> {
    Node(BinaryTree<K>),
    Leaf(K, usize),
}

#[derive(Debug)]
struct BinaryTree<K> {
    key: K,
    left: Rc<RefCell<Option<BinaryTreeEntry<K>>>>,
    right: Rc<RefCell<Option<BinaryTreeEntry<K>>>>
}

impl <K: Copy + Ord + Debug> BinaryTree<K> {
    fn search(&self, key: K) -> Option<usize> {
        if key < self.key {
            match &*self.left.borrow() {
                Some(BinaryTreeEntry::Leaf(k, v)) => if *k == key { Some(*v) } else { None },
                Some(BinaryTreeEntry::Node(tree)) => tree.search(key),
                None => None
            }
        } else {
            match &*self.right.borrow() {
                Some(BinaryTreeEntry::Leaf(k, v)) => if *k == key { Some(*v) } else { None },
                Some(BinaryTreeEntry::Node(tree)) => tree.search(key),
                None => None
            }
        }
    }

    fn insertion_cell(&self, key: K) -> Rc<RefCell<Option<BinaryTreeEntry<K>>>> {
        if key < self.key {
            let mut left_entry = self.left.borrow_mut();
            match &*left_entry {
                None => Rc::clone(&self.left),
                Some(entry) => {
                    match entry {
                        BinaryTreeEntry::Leaf(k, idx) => {
                            let new_cell = Rc::new(RefCell::new(None));
                            let node = BinaryTreeEntry::Node(
                                BinaryTree {
                                    key: *k,
                                    left: Rc::clone(&new_cell),
                                    right: Rc::new(RefCell::new(Some(BinaryTreeEntry::Leaf(*k, *idx))))
                                }
                            );
                            left_entry.replace(node);
                            new_cell
                        },
                        BinaryTreeEntry::Node(tree) => tree.insertion_cell(key)
                    }
                }
            }
        } else {
            let mut right_entry = self.right.borrow_mut();
            match &*right_entry {
                None => Rc::clone(&self.right),
                Some(entry) => {
                    match entry {
                        BinaryTreeEntry::Leaf(k, idx) => {
                            let new_cell = Rc::new(RefCell::new(None));
                            let node = BinaryTreeEntry::Node(
                                BinaryTree {
                                    key: *k,
                                    left: Rc::clone(&new_cell),
                                    right: Rc::new(RefCell::new(Some(BinaryTreeEntry::Leaf(*k, *idx))))
                                }
                            );
                            right_entry.replace(node);
                            new_cell
                        },
                        BinaryTreeEntry::Node(tree) => tree.insertion_cell(key)
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct CacheObliviousBTreeMap<K, V> {
    packed_dataset: [MaybeUninit<V>; 32],
    tree: Option<BinaryTree<K>>
}

impl <K, V> CacheObliviousBTreeMap<K, V> {
    pub fn new() -> Self {
        CacheObliviousBTreeMap {
            packed_dataset: unsafe { MaybeUninit::uninit().assume_init() },
            tree: None
        }
    }
}

impl <K, V> Drop for CacheObliviousBTreeMap<K, V> {
    fn drop(&mut self) {
        for elem in &mut self.packed_dataset {
            unsafe { ptr::drop_in_place(elem.as_mut_ptr()) }
        }
    }
}

impl <K: Copy + Debug + Ord, V: Copy + Debug> CacheObliviousBTreeMap<K, V> {
    pub fn get(&self, key: K) -> Option<V> {
        self.tree
            .as_ref()
            .and_then(|t| t.search(key))
            .map(|idx| unsafe { *(self.packed_dataset[idx].as_ptr()) })
    }

    pub fn insert(&mut self, key: K, value: V, position: usize) {
        self.packed_dataset[position] = MaybeUninit::new(value);
        let new_leaf = BinaryTreeEntry::Leaf(key, position);

        match self.tree.as_ref() {
            None => {
                let node =  Rc::new(RefCell::new(Some(new_leaf)));
                self.tree = Some(BinaryTree { key: key, left: Rc::new(RefCell::new(None)), right: Rc::clone(&node) });
            },
            Some(tree) => {
                let cell = tree.insertion_cell(key);
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
        let mut map = CacheObliviousBTreeMap::new();
        map.insert(5, "Hello", 0);
        map.insert(3, "World", 1);
        map.insert(2, "!", 2);

        assert_eq!(map.get(5), Some("Hello"));
        assert_eq!(map.get(4), None);
        assert_eq!(map.get(3), Some("World"));
        assert_eq!(map.get(2), Some("!"));
    }
}
