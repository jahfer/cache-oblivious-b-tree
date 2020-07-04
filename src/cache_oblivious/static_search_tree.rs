use std::cell::UnsafeCell;
use std::cmp::{Ord, Ordering};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::slice;
use std::sync::atomic::{AtomicPtr, AtomicU16, Ordering as AtomicOrdering};

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy)]
pub enum Key<T: Ord> {
  Infimum,
  Value(T),
  Supremum,
}

impl<T: Ord> Ord for Key<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    match (self, other) {
      (x, y) if x == y => Ordering::Equal,
      (Key::Infimum, _) | (_, Key::Supremum) => Ordering::Less,
      (Key::Supremum, _) | (_, Key::Infimum) => Ordering::Greater,
      (Key::Value(a), Key::Value(b)) => a.cmp(b),
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

pub struct Cell<'a, K: 'a, V: 'a> {
  version: AtomicU16,
  marker: AtomicPtr<Marker<K, V>>,
  empty: UnsafeCell<bool>,
  key: UnsafeCell<Option<K>>,
  value: UnsafeCell<Option<V>>,
  _marker: PhantomData<&'a Marker<K, V>>,
}

impl<K: Debug, V: Debug> Debug for Cell<'_, K, V> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let version = self.version.load(AtomicOrdering::Acquire);
    let marker = unsafe { Box::from_raw(self.marker.load(AtomicOrdering::Acquire)) };
    let empty = unsafe { *self.empty.get() };
    let key = unsafe { &*self.key.get() };
    let value = unsafe { &*self.value.get() };

    let mut dbg_struct = f.debug_struct("Cell");

    dbg_struct
      .field("version", &version)
      .field("marker", &*marker)
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

pub struct Block<'a, K: Eq + Ord, V> {
  min_key: Key<K>,
  max_key: Key<K>,
  cells: *const Cell<'a, K, V>,
  length: usize,
  _marker: PhantomData<&'a [Cell<'a, K, V>]>,
}

impl<K: Eq + Ord, V> Block<'_, K, V> {
  pub fn get(&self, _key: K) -> (&AtomicU16, &Option<V>) {
    let cell = &self.cell_slice()[0];
    (&cell.version, unsafe { &*cell.value.get() })
  }

  fn cell_slice(&self) -> &[Cell<K, V>] {
    unsafe { slice::from_raw_parts(self.cells, self.length) }
  }
}

impl<K: Eq + Ord + Debug, V: Debug> Debug for Block<'_, K, V> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("Block")
      .field(
        "range",
        &format_args!("[{:?}..{:?}]", &self.min_key, &self.max_key),
      )
      .field("cells", &self.cell_slice())
      .finish()
  }
}

enum Node<'a, K: Eq + Ord, V> {
  Leaf(Key<K>, Block<'a, K, V>),
  Branch {
    min_rhs: Key<K>,
    left: MaybeUninit<*const Node<'a, K, V>>,
    right: MaybeUninit<*const Node<'a, K, V>>,
    _marker: PhantomData<&'a Node<'a, K, V>>,
  },
}

impl<K: Eq + Ord, V> Node<'_, K, V> {
  fn locate_block_for_insertion(&self, key: Key<K>) -> &Block<'_, K, V> {
    match self {
      Node::Leaf(_, block) => block,
      Node::Branch {
        min_rhs,
        left,
        right,
        ..
      } => {
        let node = if &key < min_rhs {
          unsafe { &**left.get_ref() }
        } else {
          unsafe { &**right.get_ref() }
        };
        node.locate_block_for_insertion(key)
      }
    }
  }
}

impl<K: Eq + Ord + Debug, V: Debug> Debug for Node<'_, K, V> {
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
          ..
        } => formatter.write_fmt(format_args!(
          "|- Branch {{ min_rhs: {:?} }} @ {:p}\n{:ident$}{}\n{:ident$}{}",
          min_rhs,
          self,
          "|",
          format!(
            "{:#ident$?}",
            unsafe { &**left.get_ref() },
            ident = formatter.width().unwrap_or(0) + 3
          ),
          "|",
          format!(
            "{:#ident$?}",
            unsafe { &**right.get_ref() },
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
          ..
        } => formatter.write_fmt(format_args!(
          "|- Branch {{ min_rhs: {:?} }}\n{:ident$}{}\n{:ident$}{}",
          min_rhs,
          "|",
          format!(
            "{:ident$?}",
            unsafe { &**left.get_ref() },
            ident = formatter.width().unwrap_or(0) + 3
          ),
          "|",
          format!(
            "{:ident$?}",
            unsafe { &**right.get_ref() },
            ident = formatter.width().unwrap_or(0) + 3
          ),
          ident = formatter.width().unwrap_or(0) + 3,
        )),
      }
    }
  }
}

pub struct StaticSearchTree<'a, K: Eq + Ord + 'a, V: 'a> {
  nodes: Box<[Node<'a, K, V>]>,
  _cells: Box<[Cell<'a, K, V>]>,
}

impl<'a, K: Eq + Ord, V> StaticSearchTree<'a, K, V> {
  fn root(&'a self) -> &Node<'a, K, V> {
    &self.nodes[0]
  }
}

impl<K: Eq + Ord + Debug, V: Debug> Debug for StaticSearchTree<'_, K, V> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match f.alternate() {
      true => f.write_fmt(format_args!("{:#?}", self.root())),
      false => f.write_fmt(format_args!("{:?}", self.root())),
    }
  }
}

impl<'a, K, V> StaticSearchTree<'a, K, V>
where
  K: Eq + Copy + Ord + std::fmt::Debug + 'a,
  V: std::fmt::Debug + 'a,
{
  pub fn add(&mut self, key: K, _value: V) -> bool {
    let _block = self.root().locate_block_for_insertion(Key::Value(key));
    false
  }

  pub fn new(num_keys: u32) -> Self {
    let mut cells = Self::allocate_leaf_cells(num_keys);
    let mut nodes = Self::allocate_nodes(cells.len());

    let size = num_keys;
    // https://github.com/rust-lang/rust/issues/70887
    let slot_size = f32::log2(size as f32) as usize;
    let mut slots = cells.chunks_exact_mut(slot_size);

    let mut leaves = Self::initialize_nodes(&mut *nodes, None);
    for leaf in leaves.iter_mut() {
      Self::finalize_leaf_node(leaf, slots.next().unwrap());
    }

    let initialized_nodes = unsafe { nodes.assume_init() };
    let initialized_cells = unsafe { cells.assume_init() };

    StaticSearchTree {
      _cells: initialized_cells,
      nodes: initialized_nodes,
    }
  }

  fn allocate_leaf_cells(num_keys: u32) -> Box<[MaybeUninit<Cell<'a, K, V>>]> {
    let size = Self::values_mem_size(num_keys);
    println!("packed memory array [V; {:?}]", size);
    Box::<[Cell<K, V>]>::new_uninit_slice(size as usize)
  }

  fn allocate_nodes(leaf_count: usize) -> Box<[MaybeUninit<Node<'a, K, V>>]> {
    let size = leaf_count;
    // https://github.com/rust-lang/rust/issues/70887
    let slot_size = f32::log2(size as f32) as usize;
    let leaf_count = size / slot_size;
    let node_count = 2 * leaf_count - 1;
    println!("tree has {:?} leaves, {:?} nodes", leaf_count, node_count);
    Box::<[Node<K, V>]>::new_uninit_slice(node_count as usize)
  }

  fn initialize_nodes<'b>(
    nodes: &'b mut [MaybeUninit<Node<'a, K, V>>],
    parent_node: Option<*mut *const Node<'a, K, V>>,
  ) -> Vec<&'b mut Node<'a, K, V>> {
    let num_nodes = nodes.len();

    if num_nodes == 1 {
      nodes[0].write(Node::Branch {
        min_rhs: Key::Supremum,
        left: MaybeUninit::uninit(),
        right: MaybeUninit::uninit(),
        _marker: PhantomData,
      });
      return vec![];
    }

    if num_nodes == 3 {
      nodes[1].write(Node::Branch {
        min_rhs: Key::Supremum,
        left: MaybeUninit::uninit(),
        right: MaybeUninit::uninit(),
        _marker: PhantomData,
      });

      nodes[2].write(Node::Branch {
        min_rhs: Key::Supremum,
        left: MaybeUninit::uninit(),
        right: MaybeUninit::uninit(),
        _marker: PhantomData,
      });

      let left_node = unsafe { nodes[1].get_ref() } as *const _;
      let right_node = unsafe { nodes[2].get_ref() } as *const _;

      nodes[0].write(Node::Branch {
        min_rhs: Key::Supremum,
        left: MaybeUninit::new(left_node),
        right: MaybeUninit::new(right_node),
        _marker: PhantomData,
      });

      let (lhs, rhs) = nodes.split_at_mut(2);

      if let Some(node) = parent_node {
        unsafe { node.write(lhs[0].get_ref() as *const _) };
      }

      let left_branch = unsafe { lhs[1].get_mut() };
      let right_branch = unsafe { rhs[0].get_mut() };

      return vec![left_branch, right_branch];
    }

    let (upper_mem, lower_mem) = Self::split_tree_memory(nodes);
    let upper_subtree_length = upper_mem.len() + 1;

    let leaves_of_upper = Self::initialize_nodes(upper_mem, parent_node);

    let lower_branch_count = upper_subtree_length;
    let cells_per_branch = lower_mem.len() / lower_branch_count;

    let mut branches = lower_mem.chunks_exact_mut(cells_per_branch);

    leaves_of_upper
      .into_iter()
      .flat_map(|subtree_leaf| match subtree_leaf {
        Node::Leaf(_, _) => unreachable!(),
        Node::Branch { left, right, .. } => {
          let lhs_mem = branches.next().unwrap();
          let rhs_mem = branches.next().unwrap();
          let lhs = Self::initialize_nodes(lhs_mem, Some(left.as_mut_ptr()));
          let rhs = Self::initialize_nodes(rhs_mem, Some(right.as_mut_ptr()));
          lhs.into_iter().chain(rhs.into_iter())
        }
      })
      .collect::<Vec<_>>()
  }

  fn split_tree_memory<'b>(
    nodes: &'b mut [MaybeUninit<Node<'a, K, V>>],
  ) -> (
    &'b mut [MaybeUninit<Node<'a, K, V>>],
    &'b mut [MaybeUninit<Node<'a, K, V>>],
  ) {
    let height = f32::log2(nodes.len() as f32 + 1f32);
    let lower_height = ((height / 2f32).ceil() as u32).next_power_of_two();
    let upper_height = height as u32 - lower_height;

    let upper_subtree_length = 2 << (upper_height - 1);
    nodes.split_at_mut(upper_subtree_length - 1)
  }

  fn finalize_leaf_node<'b>(
    leaf: &'b mut Node<'a, K, V>,
    leaf_mem: &'b mut [MaybeUninit<Cell<'a, K, V>>],
  ) -> () {
    match leaf {
      Node::Branch { min_rhs, .. } => {
        let length = leaf_mem.len();
        let initialized_mem = Self::init_cell_block(leaf_mem);
        let ptr = initialized_mem as *const [Cell<K, V>] as *const Cell<K, V>;
        let block = Block {
          min_key: Key::Supremum,
          max_key: Key::Infimum,
          cells: ptr,
          length,
          _marker: PhantomData,
        };
        *leaf = Node::Leaf(*min_rhs, block);
      }
      Node::Leaf(_, _) => (),
    };
  }

  fn init_cell_block<'b>(
    cell_memory: &'b mut [MaybeUninit<Cell<'a, K, V>>],
  ) -> &'b mut [Cell<'a, K, V>] {
    for cell in cell_memory.iter_mut() {
      let marker = Box::new(Marker::<K, V>::Empty(1));
      cell.write(Cell {
        version: AtomicU16::new(1),
        marker: AtomicPtr::new(Box::into_raw(marker)),
        empty: UnsafeCell::new(true),
        key: UnsafeCell::new(None),
        value: UnsafeCell::new(None),
        _marker: PhantomData,
      });
    }
    unsafe { MaybeUninit::slice_get_mut(cell_memory) }
  }

  fn values_mem_size(num_keys: u32) -> u32 {
    let t_min = 0.5;
    let p_max = 0.25;
    let ideal_density = (t_min - p_max) / 2f32;

    let length = num_keys as f32 / ideal_density;
    // To get a balanced tree, we need to find the
    // closest double-exponential number (x = 2^2^i)
    let clean_length = 2 << ((f32::log2(length).ceil() as u32).next_power_of_two() - 1);
    clean_length
  }
}
