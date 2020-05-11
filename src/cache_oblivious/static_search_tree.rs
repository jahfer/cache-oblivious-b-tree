use std::cell::UnsafeCell;
use std::cmp::{Ord, Ordering};
use std::marker::PhantomPinned;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::AtomicPtr;

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy)]
pub enum Element<T> {
  Infimum,
  Value(T),
  Supremum,
}

impl<T: Ord> Ord for Element<T> {
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
enum Node<K, V> {
  Leaf(K, AtomicPtr<V>),
  Branch {
    min_rhs: Element<K>,
    left: MaybeUninit<UnsafeCell<NonNull<Node<K, V>>>>,
    right: MaybeUninit<UnsafeCell<NonNull<Node<K, V>>>>,
  },
  Dummy(usize),
}

pub struct StaticSearchTree<K, V> {
  _memory: Pin<Box<[Node<K, V>]>>,
  root: NonNull<Node<K, V>>,
  _pin: PhantomPinned,
}

impl<K, V> StaticSearchTree<K, V> {
  pub fn new(num_elements: u32) -> Self {
    let packed_data = Self::allocate_storage(num_elements);
    let memory = Self::allocate_tree(packed_data);
    let root = NonNull::from(&memory[0]);

    StaticSearchTree {
      root,
      _memory: memory,
      _pin: PhantomPinned,
    }
  }

  fn allocate_storage(num_elements: u32) -> Pin<Box<[MaybeUninit<V>]>> {
    let size = Self::values_mem_size(num_elements);
    println!("packed memory array [V; {:?}]", size);
    let memory = Box::<[V]>::new_uninit_slice(size as usize);
    Box::into_pin(memory)
  }

  fn init_nodes<'a>(
    memory: &'a mut [MaybeUninit<Node<K, V>>],
    parent_node: Option<&mut MaybeUninit<UnsafeCell<NonNull<Node<K, V>>>>>,
  ) -> Vec<UnsafeCell<NonNull<Node<K, V>>>> {
    let node_count = memory.len();

    if node_count <= 3 {
      for i in 0..node_count as usize {
        unsafe {
          let node_ptr = memory[i].as_mut_ptr();
          node_ptr.write(Node::Branch {
            min_rhs: Element::Supremum,
            left: MaybeUninit::uninit(),
            right: MaybeUninit::uninit(),
          });
        }
      }

      let raw_parent_node = unsafe { memory[0].get_ref() };

      let leaf = UnsafeCell::new(NonNull::from(raw_parent_node));
      match parent_node {
        Some(node) => {
          node.write(leaf);
        }
        None => {}
      };

      if node_count == 1 {
        let leaf = UnsafeCell::new(NonNull::from(raw_parent_node));
        return vec![leaf];
      }

      let leaf_a = UnsafeCell::new(NonNull::from(unsafe { memory[1].get_ref() }));
      let leaf_b = UnsafeCell::new(NonNull::from(unsafe { memory[2].get_ref() }));
      return vec![leaf_a, leaf_b];
    }

    let height = f32::log2(node_count as f32 + 1f32);
    let lower_height = ((height / 2f32).ceil() as u32).next_power_of_two();
    let upper_height = height as u32 - lower_height;

    let upper_subtree_length = 2 << (upper_height - 1);
    let (upper_subtree_mem, remaining_mem) = memory.split_at_mut(upper_subtree_length - 1);

    let mut leaves_of_upper = Self::init_nodes(upper_subtree_mem, parent_node);

    let lower_branch_count = upper_subtree_length;
    let cells_per_branch = remaining_mem.len() / lower_branch_count;

    let mut branches = remaining_mem.chunks_exact_mut(cells_per_branch);

    leaves_of_upper
      .iter_mut()
      .flat_map(|leaf| {
        let ptr = unsafe { (*leaf.get()).as_mut() };
        match ptr {
          Node::Dummy(_) => unreachable!(),
          Node::Leaf(_, _) => unreachable!(),
          Node::Branch {
            min_rhs: _,
            left,
            right,
          } => {
            let mut lhs = Self::init_nodes(branches.next().unwrap(), Some(left));
            let mut rhs = Self::init_nodes(branches.next().unwrap(), Some(right));
            lhs.append(&mut rhs);
            lhs
          }
        }
      })
      .collect::<Vec<_>>()
  }

  fn allocate_tree(packed_memory: Pin<Box<[MaybeUninit<V>]>>) -> Pin<Box<[Node<K, V>]>> {
    let size = packed_memory.len() as u32;
    // https://github.com/rust-lang/rust/issues/70887
    let slot_size = f32::log2(size as f32) as u32;
    let leaf_count = size / slot_size;
    let node_count = 2 * leaf_count - 1;
    println!("tree has {:?} leaves, {:?} nodes", leaf_count, node_count);

    let uninit_mem = Box::<[Node<K, V>]>::new_uninit_slice(node_count as usize);
    let mut pinned_mem = Box::into_pin(uninit_mem);
    let mut_ref = Pin::as_mut(&mut pinned_mem);
    let mut mem = unsafe { Pin::get_unchecked_mut(mut_ref) };
    let leaves = Self::init_nodes(&mut mem, None);
    println!("counted {:?} total leaves", leaves.len());
    println!("{:?}", leaves);

    let pinned_mem = {
      let raw_mem = unsafe { Pin::into_inner_unchecked(pinned_mem) };
      let nodes = unsafe { raw_mem.assume_init() };
      Box::into_pin(nodes)
    };

    pinned_mem
  }

  fn values_mem_size(num_elements: u32) -> u32 {
    let t_min = 0.5;
    let p_max = 0.25;
    let ideal_density = (t_min - p_max) / 2f32;

    let length = num_elements as f32 / ideal_density;
    println!("initial calc of storage length {:?}", length);
    // to get a balanced tree, we need to find the closest double-exponential number
    let clean_length = 2 << ((f32::log2(length).ceil() as u32).next_power_of_two() - 1);
    clean_length
  }
}

#[test]
fn test() {
  let _tree = StaticSearchTree::<u8, &str>::new(1);
}
