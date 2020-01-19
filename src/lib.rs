#![feature(new_uninit)]

use std::default::Default;
use std::fmt::Debug;
use std::marker::Copy;
use std::mem::MaybeUninit;
use std::cmp::Ord;
use std::rc::Rc;
use std::cell::RefCell;
use std::ptr;

#[derive(Debug)]
enum BinaryTreeEntry<K> {
    Node(Rc<RefCell<BinaryTree<K>>>),
    Leaf(K, usize),
}

#[derive(Debug)]
struct BinaryTree<K> {
    key: Option<K>,
    left: RefCell<Option<BinaryTreeEntry<K>>>,
    right: RefCell<Option<BinaryTreeEntry<K>>>
}

fn unwrap_to_node<K: Copy>(entry: &RefCell<Option<BinaryTreeEntry<K>>>) -> Rc<RefCell<BinaryTree<K>>> {
    match &*entry.borrow() {
        Some(BinaryTreeEntry::Node(reference)) => return Rc::clone(reference),
        Some(BinaryTreeEntry::Leaf(key, idx)) => {
            let new_tree: BinaryTree<K> = BinaryTree {
                key: Some(*key),
                left: RefCell::new(None),
                right: RefCell::new(Some(BinaryTreeEntry::Leaf(*key, *idx)))
            };
            let subtree_reference = Rc::new(RefCell::new(new_tree));
            entry.replace(Some(BinaryTreeEntry::Node(Rc::clone(&subtree_reference))));
            return subtree_reference
        },
        None => ()
    }

    let subtree_reference = Rc::new(RefCell::new(BinaryTree::default()));
    entry.replace(Some(BinaryTreeEntry::Node(Rc::clone(&subtree_reference))));
    subtree_reference
}

fn find_node_for_insert<K: Copy + Ord>(tree: Rc<RefCell<BinaryTree<K>>>, key: K) -> Rc<RefCell<BinaryTree<K>>> {
    let tree_ref = tree.borrow();
    if let Some(node_index) = tree_ref.key {
        if key < node_index {
            match &*tree_ref.left.borrow() {
                Some(BinaryTreeEntry::Leaf(leaf_key, _value)) => {
                    let insertion_node = unwrap_to_node(&tree_ref.left);
                    if key < *leaf_key {
                        unwrap_to_node(&insertion_node.borrow().left)
                    } else {
                        insertion_node
                    }
                },
                Some(BinaryTreeEntry::Node(_)) => {
                    find_node_for_insert(unwrap_to_node(&tree_ref.left), key)
                },
                None => Rc::clone(&tree)
            }
        } else {
            match &*tree_ref.right.borrow() {
                Some(BinaryTreeEntry::Leaf(leaf_key, _value)) => {
                    let insertion_node = unwrap_to_node(&tree_ref.right);
                    if key < *leaf_key {
                        unwrap_to_node(&insertion_node.borrow().right)
                    } else {
                        insertion_node
                    }
                },
                Some(BinaryTreeEntry::Node(_)) => {
                    find_node_for_insert(unwrap_to_node(&tree_ref.right), key)
                },
                None => Rc::clone(&tree)
            }
        }
    } else {
        Rc::clone(&tree)
    }
}

impl <K> Default for BinaryTree<K> {
    fn default() -> Self {
        BinaryTree {
            key: None,
            left: RefCell::new(None),
            right: RefCell::new(None)
        }
    }
}

impl <K: Ord + Debug> BinaryTree<K> {
    fn search(&self, key: K) -> Option<usize> {
        if let Some(node_index) = &self.key {
            if key < *node_index {
                match &*self.left.borrow() {
                    Some(BinaryTreeEntry::Leaf(k, v)) => if *k == key { Some(*v) } else { None },
                    Some(BinaryTreeEntry::Node(boxed_tree)) => boxed_tree.borrow().search(key),
                    None => None
                }
            } else {
                match &*self.right.borrow() {
                    Some(BinaryTreeEntry::Leaf(k, v)) => if *k == key { Some(*v) } else { None },
                    Some(BinaryTreeEntry::Node(boxed_tree)) => boxed_tree.borrow().search(key),
                    None => None
                }
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct CacheObliviousBTreeMap<K: Clone, V> {
    packed_dataset: [MaybeUninit<V>; 32],
    tree: Rc<RefCell<BinaryTree<K>>>
}

impl <K: Clone, V> CacheObliviousBTreeMap<K, V> {
    pub fn new() -> Self {
        CacheObliviousBTreeMap {
            packed_dataset: unsafe { MaybeUninit::uninit().assume_init() },
            tree: Rc::new(RefCell::new(BinaryTree::default()))
        }
    }

    pub fn drop(mut self) {
        for elem in &mut self.packed_dataset {
            unsafe { ptr::drop_in_place(elem.as_mut_ptr()) }
        }
    }
}

impl <K: Copy + Debug + Ord, V: Copy + Debug> CacheObliviousBTreeMap<K, V> {
    pub fn insert(&mut self, key: K, value: V, position: usize) {
        self.packed_dataset[position] = MaybeUninit::new(value);

        let tree_reference = Rc::clone(&self.tree);
        let subtree = find_node_for_insert(tree_reference, key);

        let subtree_key = subtree.borrow().key;
        if let Some(node_key) = subtree_key {
            if key < node_key {
                subtree.borrow().left.replace(Some(BinaryTreeEntry::Leaf(key, position)));
            }
        } else {
            let mut st = subtree.borrow_mut();
            st.right.replace(Some(BinaryTreeEntry::Leaf(key, position)));
            st.key = Some(key);
        }
    }
}

impl <K: Debug + Ord + Clone, V: Debug + Copy> CacheObliviousBTreeMap<K, V> {
    pub fn get(&self, key: K) -> Option<V> {
        match self.tree.borrow().search(key) {
            Some(idx) => Some(unsafe { *(self.packed_dataset[idx].as_ptr()) }),
            None => None
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

        dbg!(&map.tree);

        assert_eq!(map.get(5), Some("Hello"));
        assert_eq!(map.get(4), None);
        assert_eq!(map.get(3), Some("World"));
        map.drop();
    }
}
