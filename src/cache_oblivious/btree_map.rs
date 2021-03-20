use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::Ordering;
use std::collections::VecDeque;
use std::convert::TryInto;
use std::ptr::NonNull;

use num_rational::{Ratio, Rational};

use super::packed_memory_array::PackedMemoryArray;
use super::cell::{Cell, Key, CellIterator, CellGuard, Marker};

pub struct BTreeMap<'a, K: Copy + Ord, V: Clone> {
  data: PackedMemoryArray<Cell<K,V>>,
  index: BlockIndex<'a, K, V>
}

impl <'a, K, V> BTreeMap<'a, K, V> where K: Copy + Ord, V: Clone {
  pub fn new(capacity: u32) -> BTreeMap<'a, K, V> {
    let packed_cells = PackedMemoryArray::with_capacity(capacity);
    let cells_ptr = NonNull::from(&packed_cells);
    let index = Self::generate_index(cells_ptr);
    
    BTreeMap {
      data: packed_cells,
      index,
    }
  }

  pub fn get(&self, key: &K) -> Option<&V> {
    self.index.get(key)
  }

  pub fn insert(&mut self, key: K, value: V) -> () where K: Debug {
    let (block, min_key) = match self.index.get_block_for_insert(&key) {
      SearchResult::Block(block, min_key) => (block, min_key),
      _ => panic!("No block found for insert of key {:?}", key)
    };
    
    // Todo: Clean up (abstract out CellGuard)
    let vec = self.data
      .into_iter()
      .skip_while(|&x| x as *const _ != block.cell_slice_ptr)
      .map(|c| unsafe { CellGuard::from_raw(c).unwrap() })
      .collect::<Vec<_>>();

    let mut selected_cell: Option<CellGuard<K, V>> = None;

    for mut cell_guard in vec {
      if cell_guard.is_empty() {
        // node says there's a cell smaller than ours, keep looking
        if selected_cell.is_none() && min_key <= Key::Value(&key) {
          continue;
        // block is empty
        } else {
          selected_cell = None;
        }
      } else {
        let cache = cell_guard.cache().unwrap().clone().unwrap();
        if Key::Value(cache.key) < Key::Value(key) {
          selected_cell = Some(cell_guard);
          continue
        } else if Key::Value(cache.key) == Key::Value(key) {
          selected_cell = None;
        } else if selected_cell.is_none() {
          // we didn't find any cells that were <= our key, rebalance to make room
          // is_smallest_key = true;
          self.rebalance(cell_guard.inner as *const _, true);
        }
      }

      if let Some(cell_to_move) = &selected_cell {
        // move cell to make room for insert
        self.rebalance(cell_to_move.inner, true);
      }

      let marker_version = cell_guard.cache_version + 1;
      let cell = selected_cell.as_mut().unwrap_or(&mut cell_guard);
      let marker = Marker::InsertCell(marker_version, key, value.clone());

      let result = cell.update(marker);
      
      if result.is_err() {
        // Marker has been updated by another process, start loop over
        continue
      }

      let prev_marker = result.unwrap();

      // We now have exclusive access to the cell until we update `version`.
      // This works well for mutating through UnsafeCell<T>, but isn't really
      // "lock-free"...
      unsafe {
        cell.inner.key.get().write(Some(key));
        cell.inner.value.get().write(Some(value));
      };

      let next_version = marker_version + 1;

      // Reuse previous marker allocation
      unsafe { prev_marker.write(Marker::Empty(next_version)) };
      cell
        .inner
        .marker
        .as_ref()
        .unwrap()
        .swap(prev_marker, Ordering::SeqCst);
      cell.inner.version.swap(next_version, Ordering::SeqCst);

      break;
    }

    let cells_ptr = NonNull::from(&self.data);
    self.index = Self::generate_index(cells_ptr);
  }

  pub fn generate_index(data: NonNull<PackedMemoryArray<Cell<K,V>>>) -> BlockIndex<'a, K, V> {
    BlockIndex {
      map: data,
      index_tree: BlockSearchTree::new(data)
    }
  }

  fn rebalance(&self, cell_ptr_start: *const Cell<K, V>, for_insertion: bool) {
    let mut count = 1;
    let mut cells_to_move: VecDeque<*const Cell<K, V>> = VecDeque::new();
    let mut current_cell_ptr = cell_ptr_start;

    let vec = self.data
      .into_iter()
      .skip_while(|&x| x as *const _ != cell_ptr_start)
      .map(|c| unsafe { CellGuard::from_raw(c).unwrap() })
      .collect::<Vec<_>>();

    for cell_guard in vec {
      current_cell_ptr = cell_guard.inner;

      if !cell_guard.is_empty() {
        cells_to_move.push_front(cell_guard.inner);
      }

      let numer = if for_insertion {
        cells_to_move.len() + 1
      } else {
        cells_to_move.len()
      };

      let current_density = Rational::new(
        numer.try_into().unwrap(), 
        count.try_into().unwrap()
      );

      if self.within_density_threshold(count, current_density) {
        break;
      }

      count += 1;
    }

    // There are different strategies available for rebalancing
    // depending on the inserts expected in the system.
    //
    // To support many inserts in the same area, we want to shift
    // elements as far right as possible, while still maintaining
    // our density thresholds. Since we know we needed exactly
    // this # of cells to meet our density, one less than this
    // should let each region maximize their respective thresholds.

    // Starting at current_marker_raw, move the cells rightward
    // until their max density is reached

    // TODO: Can we use #compare_and_swap for neighbouring cells
    // to make sure we don't leave any cells unallocated?

    println!("Identified {} cells to relocate", cells_to_move.len());

    for cell_ptr in cells_to_move.iter() {
      let cell = unsafe { &*current_cell_ptr };
      let cell_key = unsafe { *cell.key.get() };
      if cell_key.is_some() {
        // TODO: I think we can overwrite these records since their contents have been moved...
      }

      let cell_to_move = unsafe { &**cell_ptr };
      let version = cell_to_move.version.load(Ordering::SeqCst);
      let current_marker_raw = cell_to_move
        .marker
        .as_ref()
        .unwrap()
        .load(Ordering::SeqCst);
      let marker = unsafe { &*current_marker_raw };
      let marker_version = *marker.version();

      if version != marker_version {
        todo!("Restart rebalance!");
      }

      // todo: self.data.into_iter() seems suspicious here
      let dest_index =
        unsafe { current_cell_ptr.offset_from(self.data.into_iter().next().unwrap() as *const Cell<K, V>) };
      let new_marker = Box::new(Marker::Move(marker_version, dest_index));

      let new_marker_raw = Box::into_raw(new_marker);
      let prev_marker = cell_to_move.marker.as_ref().unwrap().compare_exchange(
        current_marker_raw,
        new_marker_raw,
        Ordering::SeqCst,
        Ordering::SeqCst,
      );

      if prev_marker.is_err() {
        // Marker has been updated by another process.
        // Deallocate memory, start loop over.
        unsafe { Box::from_raw(new_marker_raw) };
        todo!("Restart rebalance!");
      }

      unsafe {
        // update new cell
        cell.key.get().write(*cell_to_move.key.get());
        cell.value.get().write((*cell_to_move.value.get()).clone());
        // todo update version and marker of new cell?

        // update old cell
        cell_to_move.key.get().write(None);
        cell_to_move.value.get().write(None);
        let new_version = version + 1;
        // reuse prev_marker box
        prev_marker.unwrap().write(Marker::Empty(new_version));
        // use compare_exchange_weak because we can safely fail here
        let _ = cell_to_move.marker.as_ref().unwrap().compare_exchange_weak(
          new_marker_raw,
          prev_marker.unwrap(),
          Ordering::SeqCst,
          Ordering::SeqCst,
        );
        let _ = cell_to_move.version.compare_exchange_weak(
          marker_version,
          new_version,
          Ordering::SeqCst,
          Ordering::SeqCst
        );
        // TODO: increment version, clear marker
      };

      current_cell_ptr = unsafe { current_cell_ptr.sub(1) };
      if self.data.is_valid_pointer(&current_cell_ptr) {
        continue;
      } else {
        unreachable!("We've reached the end of the initialized cell buffer!");
      }
    }
  }

  fn within_density_threshold(&self, num_items: usize, current_density: Ratio<isize>) -> bool {
    let density = self
      .data
      .config
      .density_scale
      .iter()
      .find(|d| d.max_item_count >= num_items)
      .unwrap();

    density.range.contains(&current_density)
  }
}

impl <'a, K, V> Debug for BTreeMap<'a, K, V> where K: Copy + Ord + Debug, V: Clone + Debug {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("BTreeMap")
      .field("data", &format_args!("{:?}", self.data))
      .finish()
  }
}

pub struct BlockIndex<'a, K: Copy + Ord, V: Clone> {
  map: NonNull<PackedMemoryArray<Cell<K, V>>>,
  index_tree: BlockSearchTree<'a, K, V>
}

impl <'a, K, V> Debug for BlockIndex<'a, K, V>
where
  K: Copy + Ord + Debug,
  V: Clone + Debug 
{
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("BlockIndex")
      .field("map", &format_args!("{:?}", self.map))
      .field("index_tree", &format_args!("{:?}", self.index_tree))
      .finish()
  }
}

impl <'a, K, V> BlockIndex<'a, K, V> where K: Copy + Ord, V: Clone {
  fn get_block_for_insert<Q>(&'a self, search_key: &Q) -> SearchResult<'a, K, V>
  where
    Q: Ord,
    K: Borrow<Q> {
      self.index_tree.find(search_key, true)
  }

  pub fn get<Q>(&self, search_key: &Q) -> Option<&'a V>
  where
    Q: Ord,
    K: Borrow<Q>
  {
    match self.index_tree.find(search_key, false) {
      SearchResult::NotFound => return None,
      SearchResult::Block(block, ..) => {
        let iter = CellIterator::new(
          block.cell_slice_ptr, 
          unsafe { self.map.as_ref() }.active_range.end
        );
    
        for cell_guard in iter {
          if !cell_guard.is_empty() {
            let cache = cell_guard.cache().unwrap().clone().unwrap();
            let cache_key = cache.key.borrow();
            if cache_key == search_key {
              // todo: ABA problem
              let value = unsafe { (&*cell_guard.inner.value.get()).as_ref().unwrap() };
              return Some(value)
            } else if cache_key > search_key {
              return None
            }
          }
        }

        None
      },
      _ => unreachable!()
    }
  }
}

struct BlockSearchTree<'a, K: Clone + Ord, V: Clone> {
  nodes: Box<[Node<'a, K, V>]>
}

impl <'a, K, V> BlockSearchTree<'a, K, V>
where
  K: Clone + Ord,
  V: Clone
{
  fn new(cells: NonNull<PackedMemoryArray<Cell<K,V>>>) -> BlockSearchTree<'a, K, V> {
    let cells = unsafe { cells.as_ref() };
    let mut nodes = Self::allocate(cells.len());

    let mut leaves = Self::initialize_nodes(&mut *nodes, None);
    let slot_size = f32::log2(cells.requested_capacity as f32) as usize; // https://github.com/rust-lang/rust/issues/70887
    let mut slots = cells.as_slice().chunks_exact(slot_size);

    for leaf in leaves.iter_mut() {
      Self::finalize_leaf_node(leaf, slots.next().unwrap());
    }

    let initialized_nodes = unsafe { nodes.assume_init() };

    BlockSearchTree {
      nodes: initialized_nodes
    }
  }

  fn allocate(leaf_count: usize) -> Box<[MaybeUninit<Node<'a, K, V>>]> {
    let size = leaf_count;
    // https://github.com/rust-lang/rust/issues/70887
    let slot_size = f32::log2(size as f32) as usize;
    let leaf_count = size / slot_size;
    let node_count = 2 * leaf_count - 1;
    // println!("tree has {:?} leaves, {:?} nodes", leaf_count, node_count);
    Box::<[Node<K, V>]>::new_uninit_slice(node_count as usize)
  }

  fn initialize_nodes<'b>(
    nodes: &'b mut [MaybeUninit<Node<'a, K, V>>],
    parent_node: Option<*mut *const Node<'a, K, V>>,
  ) -> Vec<&'b mut Node<'a, K, V>> {
    if nodes.len() <= 3 {
      return Self::assign_node_values(nodes, parent_node);
    }

    let (upper_mem, lower_mem) = Self::split_tree_memory(nodes);
    let num_lower_branches = upper_mem.len() + 1;

    let leaves_of_upper = Self::initialize_nodes(upper_mem, parent_node);

    let nodes_per_branch = lower_mem.len() / num_lower_branches;
    let mut branches = lower_mem.chunks_exact_mut(nodes_per_branch);

    leaves_of_upper
      .into_iter()
      .flat_map(|subtree_leaf| match subtree_leaf {
        Node::Leaf(_, _) => unreachable!(),
        Node::Internal { left, right, .. } => {
          let lhs_mem = branches.next().unwrap();
          let rhs_mem = branches.next().unwrap();
          let lhs = Self::initialize_nodes(lhs_mem, Some(left.as_mut_ptr()));
          let rhs = Self::initialize_nodes(rhs_mem, Some(right.as_mut_ptr()));
          lhs.into_iter().chain(rhs.into_iter())
        }
      })
      .collect::<Vec<_>>()
  }

  fn assign_node_values<'b>(
    nodes: &'b mut [MaybeUninit<Node<'a, K, V>>],
    parent_node: Option<*mut *const Node<'a, K, V>>,
  ) -> Vec<&'b mut Node<'a, K, V>> {
    let num_nodes = nodes.len();
    assert!(num_nodes <= 3);

    for i in 0..num_nodes as usize {
      nodes[i].write(Node::Internal {
        min_rhs: Key::Supremum,
        left: MaybeUninit::uninit(),
        right: MaybeUninit::uninit(),
        _marker: PhantomData,
      });
    }

    if num_nodes == 1 {
      let node = unsafe { nodes[0].assume_init_mut() };
      return vec![node];
    }

    let left_node = unsafe { nodes[1].assume_init_ref() } as *const _;
    let right_node = unsafe { nodes[2].assume_init_ref() } as *const _;

    nodes[0].write(Node::Internal {
      min_rhs: Key::Supremum,
      left: MaybeUninit::new(left_node),
      right: MaybeUninit::new(right_node),
      _marker: PhantomData,
    });

    let (lhs, rhs) = nodes.split_at_mut(2);

    if let Some(node) = parent_node {
      unsafe { node.write(lhs[0].assume_init_ref() as *const _) };
    }

    let left_branch = unsafe { lhs[1].assume_init_mut() };
    let right_branch = unsafe { rhs[0].assume_init_mut() };

    return vec![left_branch, right_branch];
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
    leaf_mem: &'b [Cell<K, V>],
  ) -> () {
    match leaf {
      Node::Internal { .. } => {
        let min_key = leaf_mem
          .first()
          .and_then(|c| unsafe { (*c.key.get()).as_ref() })
          .and_then(|k| Some(Key::Value(k.clone())))
          .unwrap_or(Key::Supremum);
          
        let length = leaf_mem.len();
        let ptr = leaf_mem as *const [Cell<K, V>] as *const Cell<K, V>;
        let block = Block {
          cell_slice_ptr: ptr,
          length,
          _marker: PhantomData,
        };
        *leaf = Node::Leaf(min_key, block);
      }
      Node::Leaf(_, _) => (),
    };
  }

  fn root(&'a self) -> &Node<'a, K, V> {
    &self.nodes[0]
  }

  fn find<Q>(&'a self, search_key: &Q, for_insertion: bool) -> SearchResult<'a, K, V>
    where
      K: Borrow<Q>,
      Q: Ord
  {
    self.root().search_to_block(Key::Value(search_key), for_insertion)
  }
}

impl <'a, K, V> Debug for BlockSearchTree<'a, K, V>
where
  K: Ord + Clone + Debug,
  V: Clone + Debug 
{
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("BlockSearchTree")
      .field("nodes", &format_args!("{:?}", self.nodes))
      .finish()
  }
}

enum Node<'a, K: Clone + Ord, V: Clone> {
  Leaf(Key<K>, Block<'a, K, V>),
  Internal {
    min_rhs: Key<K>,
    left: MaybeUninit<*const Node<'a, K, V>>,
    right: MaybeUninit<*const Node<'a, K, V>>,
    _marker: PhantomData<&'a Node<'a, K, V>>,
  },
}

impl <K, V> Node<'_, K, V> 
where
  K: Clone + Ord,
  V: Clone
{
  fn search<'a, Q>(&'a self, key: Key<&Q>, allow_empty: bool) -> SearchResult<'a, K, V>
  where
    Q: Ord,
    K: Borrow<Q>
  {
    match self {
      Node::Leaf(key_lock, block) => {
        if !allow_empty && *key_lock == Key::Supremum {
          SearchResult::NotFound
        } else {
          SearchResult::Block(block, Key::from(key_lock))
        }
      },
      Node::Internal {
        min_rhs,
        left,
        right,
        ..
      } => {
        let node = if min_rhs.is_supremum() || key < Key::Value(min_rhs.clone().unwrap().borrow()) {
          unsafe { &**left.assume_init_ref() }
        } else {
          unsafe { &**right.assume_init_ref() }
        };
        SearchResult::Internal(node)
      }
    }
  }
  
  fn search_to_block<'a, Q>(&'a self, key: Key<&Q>, allow_empty: bool) -> SearchResult<'a, K, V>
  where
    Q: Ord,
    K: Borrow<Q>
  {
    let mut result = None;
    let mut node = self;

    while let None = result {
      match node.search(key, allow_empty) {
        SearchResult::Internal(next_node) => node = next_node,
        x @ _ => result = Some(x),
      }
    }

    result.unwrap()
  }
}

impl <'a, K, V> Debug for Node<'a, K, V>
where
  K: Ord + Clone + Debug,
  V: Clone + Debug 
{
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Node::Leaf(key, ..) => {
        formatter
        .debug_struct("Node::Leaf")
        .field("key", &format_args!("{:?}", key))
        .finish()
      },
      Node::Internal { min_rhs, .. } => {
        formatter
        .debug_struct("Node::Internal")
        .field("min_rhs", &format_args!("{:?}", min_rhs))
        .finish()
      }
    }
  }
}

enum SearchResult<'a, K: Clone + Ord, V: Clone> {
  Block(&'a Block<'a, K, V>, Key<&'a K>),
  Internal(&'a Node<'a, K, V>),
  NotFound
} 

struct Block<'a, K: Clone, V: Clone> {
  cell_slice_ptr: *const Cell<K, V>,
  length: usize,
  _marker: PhantomData<&'a [Cell<K, V>]>,
}