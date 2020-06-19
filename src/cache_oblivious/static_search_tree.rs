use std::cell::UnsafeCell;
use std::cmp::{Ord, Ordering};
use std::fmt::{self, Debug};
use std::marker::PhantomPinned;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicU16, Ordering as AtomicOrdering};

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
enum Marker<K, V> {
  Empty(u16),
  Move(u16, K, V),
  InsertCell(u16, K, V),
  DeleteCell(u16, K),
}

pub struct Cell<K, V> {
  version: AtomicU16,
  marker: AtomicPtr<Marker<K, V>>,
  empty: UnsafeCell<bool>,
  key: UnsafeCell<Option<K>>,
  value: UnsafeCell<Option<V>>,
}

impl<K: Debug, V: Debug> Debug for Cell<K, V> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let version = self.version.load(AtomicOrdering::Acquire);
    let marker = unsafe { &*self.marker.load(AtomicOrdering::Acquire) };
    let empty = unsafe { *self.empty.get() };
    let key = unsafe { &*self.key.get() };
    let value = unsafe { &*self.value.get() };

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

#[derive(Debug)]
pub struct Block<K: Eq, V> {
  min_key: Element<K>,
  max_key: Element<K>,
  cells: NonNull<[Cell<K, V>]>,
}

impl<K: Eq + Copy, V: Copy> Block<K, V> {
  pub fn get(&self, _key: K) -> (&AtomicU16, &Option<V>) {
    let cell = unsafe { &self.cells.as_ref()[0] };
    (&cell.version, unsafe { &*cell.value.get() })
  }
}

enum Node<K: Eq, V> {
  Leaf(Element<K>, Block<K, V>),
  Branch {
    min_rhs: Element<K>,
    left: MaybeUninit<NonNull<Node<K, V>>>,
    right: MaybeUninit<NonNull<Node<K, V>>>,
  },
}

impl<K: Eq + Debug, V: Debug> Debug for Node<K, V> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    if formatter.alternate() {
      match self {
        Self::Leaf(_, block) => formatter.write_fmt(format_args!(
          "{}",
          format!("|- Leaf => {:?} @ {:p}", block, self)
        )),
        Self::Branch {
          min_rhs,
          left,
          right,
        } => formatter.write_fmt(format_args!(
          "|- Branch {{ min_rhs: {:?} }} @ {:p}\n{:ident$}{}\n{:ident$}{}",
          min_rhs,
          self,
          "|",
          format!(
            "{:#ident$?}",
            unsafe { left.get_ref().as_ref() },
            ident = formatter.width().unwrap_or(0) + 3
          ),
          "|",
          format!(
            "{:#ident$?}",
            unsafe { right.get_ref().as_ref() },
            ident = formatter.width().unwrap_or(0) + 3
          ),
          ident = formatter.width().unwrap_or(0) + 3,
        )),
      }
    } else {
      match self {
        Self::Leaf(_, block) => {
          formatter.write_fmt(format_args!("{}", format!("|- Leaf => {:?}", block)))
        }
        Self::Branch {
          min_rhs,
          left,
          right,
        } => formatter.write_fmt(format_args!(
          "|- Branch {{ min_rhs: {:?} }}\n{:ident$}{}\n{:ident$}{}",
          min_rhs,
          "|",
          format!(
            "{:ident$?}",
            unsafe { left.get_ref().as_ref() },
            ident = formatter.width().unwrap_or(0) + 3
          ),
          "|",
          format!(
            "{:ident$?}",
            unsafe { right.get_ref().as_ref() },
            ident = formatter.width().unwrap_or(0) + 3
          ),
          ident = formatter.width().unwrap_or(0) + 3,
        )),
      }
    }
  }
}

pub struct StaticSearchTree<K: Eq, V> {
  _memory: Pin<Box<[Node<K, V>]>>,
  root: NonNull<Node<K, V>>,
  _pin: PhantomPinned,
}

impl<K: Eq + Debug, V: Debug> Debug for StaticSearchTree<K, V> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match f.alternate() {
      true => f.write_fmt(format_args!("{:#?}", unsafe { self.root.as_ref() })),
      false => f.write_fmt(format_args!("{:?}", unsafe { self.root.as_ref() })),
    }
  }
}

impl<K: Eq + Copy + Ord + std::fmt::Debug, V: std::fmt::Debug> StaticSearchTree<K, V> {
  fn root_node(&self) -> &Node<K, V> {
    unsafe { self.root.as_ref() }
  }

  pub fn new(num_elements: u32) -> Self {
    let mut packed_data = Self::allocate_storage(num_elements);
    let memory = Self::allocate_tree(&mut packed_data);
    let root = NonNull::from(&memory[0]);

    StaticSearchTree {
      root,
      _memory: memory,
      _pin: PhantomPinned,
    }
  }

  fn allocate_storage(num_elements: u32) -> Pin<Box<[MaybeUninit<Cell<K, V>>]>> {
    let size = Self::values_mem_size(num_elements);
    println!("packed memory array [V; {:?}]", size);
    let memory = Box::<[Cell<K, V>]>::new_uninit_slice(size as usize);
    Box::into_pin(memory)
  }

  fn allocate_tree(
    mut packed_memory: &mut Pin<Box<[MaybeUninit<Cell<K, V>>]>>,
  ) -> Pin<Box<[Node<K, V>]>> {
    let size = packed_memory.len() as usize;
    // https://github.com/rust-lang/rust/issues/70887
    let slot_size = f32::log2(size as f32) as usize;
    let leaf_count = size / slot_size;
    let node_count = 2 * leaf_count - 1;
    println!("tree has {:?} leaves, {:?} nodes", leaf_count, node_count);

    let mutable_mem = Pin::as_mut(&mut packed_memory);
    let mut slots = unsafe { Pin::get_unchecked_mut(mutable_mem).chunks_exact_mut(slot_size) };

    let boxed_uninit_nodes = Box::<[Node<K, V>]>::new_uninit_slice(node_count as usize);
    let mut boxed_pinned_uninit_nodes = Box::into_pin(boxed_uninit_nodes);

    let unint_nodes = Pin::as_mut(&mut boxed_pinned_uninit_nodes);
    let mut mut_uninit_nodes = unsafe { Pin::get_unchecked_mut(unint_nodes) };

    let _ = Self::init_nodes(&mut mut_uninit_nodes, None)
      .into_iter()
      .map(|leaf| Self::finalize_leaf_node(leaf, slots.next().unwrap()))
      .collect::<Vec<_>>();

    let pinned_mem = {
      let raw_mem = unsafe { Pin::into_inner_unchecked(boxed_pinned_uninit_nodes) };
      let nodes = unsafe { raw_mem.assume_init() };
      Box::into_pin(nodes)
    };

    pinned_mem
  }

  fn finalize_leaf_node(
    mut leaf: NonNull<Node<K, V>>,
    leaf_mem: &mut [MaybeUninit<Cell<K, V>>],
  ) -> NonNull<Node<K, V>> {
    let l = unsafe { leaf.as_mut() };
    match l {
      Node::Branch { min_rhs, .. } => {
        let initialized_mem = Self::init_cell_block(leaf_mem);
        let ptr = NonNull::from(initialized_mem);
        let block = Block {
          min_key: Element::Supremum,
          max_key: Element::Infimum,
          cells: ptr,
        };
        *l = Node::Leaf(*min_rhs, block);
      }
      Node::Leaf(_, _) => (),
    }
    leaf
  }

  fn init_cell_block(cell_memory: &mut [MaybeUninit<Cell<K, V>>]) -> &mut [Cell<K, V>] {
    for cell in cell_memory.iter_mut() {
      cell.write(Cell {
        version: AtomicU16::new(1),
        marker: AtomicPtr::new(&mut Marker::Empty(1)),
        empty: UnsafeCell::new(true),
        key: UnsafeCell::new(None),
        value: UnsafeCell::new(None),
      });
    }
    unsafe { MaybeUninit::slice_get_mut(cell_memory) }
  }

  fn init_nodes<'a>(
    memory: &'a mut [MaybeUninit<Node<K, V>>],
    parent_node: Option<&mut MaybeUninit<NonNull<Node<K, V>>>>,
  ) -> Vec<NonNull<Node<K, V>>> {
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

      let mut local_root = NonNull::from(raw_parent_node);

      // only possibility here is that this is the tree root
      if node_count == 1 {
        return vec![local_root];
      }

      let root_node = unsafe { local_root.as_mut() };
      match root_node {
        Node::Leaf(_, _) => unreachable!(),
        Node::Branch { left, right, .. } => {
          let leaf_a = NonNull::from(unsafe { memory[1].get_ref() });
          let leaf_b = NonNull::from(unsafe { memory[2].get_ref() });
          left.write(leaf_a);
          right.write(leaf_b);
        }
      }

      if let Some(node) = parent_node {
        node.write(local_root);
      }

      let leaf_a = NonNull::from(unsafe { memory[1].get_ref() });
      let leaf_b = NonNull::from(unsafe { memory[2].get_ref() });
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
      .flat_map(|subtree_leaf| {
        let ptr = unsafe { subtree_leaf.as_mut() };
        match ptr {
          Node::Leaf(_, _) => unreachable!(),
          Node::Branch { left, right, .. } => {
            let lhs_mem = branches.next().unwrap();
            let rhs_mem = branches.next().unwrap();
            let lhs = Self::init_nodes(lhs_mem, Some(left));
            let rhs = Self::init_nodes(rhs_mem, Some(right));
            lhs.into_iter().chain(rhs.into_iter())
          }
        }
      })
      .collect::<Vec<_>>()
  }

  fn values_mem_size(num_elements: u32) -> u32 {
    let t_min = 0.5;
    let p_max = 0.25;
    let ideal_density = (t_min - p_max) / 2f32;

    let length = num_elements as f32 / ideal_density;
    // To get a balanced tree, we need to find the
    // closest double-exponential number (x = 2^2^i)
    let clean_length = 2 << ((f32::log2(length).ceil() as u32).next_power_of_two() - 1);
    clean_length
  }
}
