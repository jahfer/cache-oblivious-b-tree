use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::convert::TryInto;
use std::fmt::{self, Debug};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, RwLock, Weak};
use std::thread;
use std::time;

use num_rational::{Ratio, Rational};

use super::cell::{Cell, CellGuard, CellIterator, Key, Marker};
use super::packed_memory_array::PackedMemoryArray;

const INDEX_UPDATE_DELAY: time::Duration = time::Duration::from_millis(50);

pub struct BTreeMap<K: Clone + Ord, V: Clone> {
    data: Arc<PackedMemoryArray<Cell<K, V>>>,
    index: Arc<RwLock<BlockIndex<K, V>>>,
    tx: Sender<Weak<PackedMemoryArray<Cell<K, V>>>>,
    index_updating: Arc<AtomicBool>,
}

impl<K, V> BTreeMap<K, V>
where
    K: 'static + Clone + Ord,
    V: 'static + Clone,
{
    pub fn new(capacity: u32) -> BTreeMap<K, V> {
        let packed_cells = PackedMemoryArray::with_capacity(capacity);
        let data = Arc::new(packed_cells);

        let raw_index = Self::generate_index(Arc::clone(&data));
        let index = Arc::new(RwLock::new(raw_index));

        let thread_index = Arc::clone(&index);
        let (tx, rx) = channel::<Weak<PackedMemoryArray<Cell<K, V>>>>();
        let index_updating = Self::start_indexing_thread(thread_index, rx);

        BTreeMap {
            index,
            data,
            tx,
            index_updating,
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        self.index.read().unwrap().get(key)
    }

    pub fn insert<'a>(&'a mut self, key: K, value: V)
    where
        K: Debug,
    {
        let index = self.index.read().unwrap();
        let (block, min_key) = match index.get_block_for_insert(&key) {
            SearchResult::Block(block, min_key) => (block, min_key),
            _ => panic!("No block found for insert of key {:?}", key),
        };

        // Todo: Clean up (abstract out CellGuard)
        let iter = self
            .data
            .into_iter()
            .skip_while(|&x| x as *const _ != block.cell_slice_ptr)
            .map(|c| unsafe { CellGuard::from_raw(c).unwrap() });

        let mut selected_cell: Option<CellGuard<K, V>> = None;
        for mut cell_guard in iter {
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
                if Key::Value(&cache.key) < Key::Value(&key) {
                    selected_cell = Some(cell_guard);
                    continue;
                } else if Key::Value(&cache.key) == Key::Value(&key) {
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

            let marker = Marker::InsertCell(marker_version, key.clone(), value.clone());

            let result = cell.update(marker);

            if result.is_err() {
                // Marker has been updated by another process, start loop over
                continue;
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
            cell.inner
                .marker
                .as_ref()
                .unwrap()
                .swap(prev_marker, Ordering::SeqCst);
            cell.inner.version.swap(next_version, Ordering::SeqCst);

            break;
        }

        self.request_reindex();
    }

    pub fn generate_index(data: Arc<PackedMemoryArray<Cell<K, V>>>) -> BlockIndex<K, V> {
        BlockIndex {
            map: Arc::clone(&data),
            index_tree: BlockSearchTree::new(data),
        }
    }

    fn request_reindex(&self) {
        // debounce
        if !self.index_updating.load(Ordering::Acquire) {
            let _ = self.tx.send(Arc::downgrade(&self.data));
        }
    }

    fn start_indexing_thread(
        index: Arc<RwLock<BlockIndex<K, V>>>,
        rx: Receiver<Weak<PackedMemoryArray<Cell<K, V>>>>,
    ) -> Arc<AtomicBool> {
        let is_updating = Arc::new(AtomicBool::new(false));
        let thread_is_updating = Arc::clone(&is_updating);
        thread::spawn(move || {
            loop {
                let result = rx
                    .recv()
                    .ok()
                    .map(|x| {
                        thread_is_updating.store(true, Ordering::Release);
                        x
                    })
                    .and_then(|cells_ptr| cells_ptr.upgrade())
                    .map(|cells| {
                        let new_index = Self::generate_index(cells);
                        let mut i = index.write().unwrap();
                        // todo is the memory barrier in the right place here, relative to the index generation?
                        thread_is_updating.store(false, Ordering::Release);
                        *i = new_index;
                    });

                if let None = result {
                    break;
                }

                thread::sleep(INDEX_UPDATE_DELAY);
            }
        });

        is_updating
    }

    fn rebalance(&self, cell_ptr_start: *const Cell<K, V>, for_insertion: bool) {
        let mut count = 1;
        let mut cells_to_move: VecDeque<*const Cell<K, V>> = VecDeque::new();
        let mut current_cell_ptr = cell_ptr_start;

        let vec = self
            .data
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

            let current_density =
                Rational::new(numer.try_into().unwrap(), count.try_into().unwrap());

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

        for cell_ptr in cells_to_move.iter() {
            let cell = unsafe { &*current_cell_ptr };
            let cell_key = unsafe { &*cell.key.get() };
            if cell_key.is_some() {
                // TODO: I think we can overwrite these records since their contents have been moved...
            }

            let cell_to_move = unsafe { &**cell_ptr };
            let version = cell_to_move.version.load(Ordering::SeqCst);
            let current_marker_raw = cell_to_move.marker.as_ref().unwrap().load(Ordering::SeqCst);
            let marker = unsafe { &*current_marker_raw };
            let marker_version = *marker.version();

            if version != marker_version {
                todo!("Restart rebalance!");
            }

            // todo: self.data.into_iter() seems suspicious here
            let dest_index = unsafe {
                current_cell_ptr
                    .offset_from(self.data.into_iter().next().unwrap() as *const Cell<K, V>)
            };
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
                unsafe { drop(Box::from_raw(new_marker_raw)) };
                todo!("Restart rebalance!");
            }

            unsafe {
                // update new cell
                cell.key.get().write((*cell_to_move.key.get()).clone());
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
                    Ordering::SeqCst,
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

impl<K, V> Debug for BTreeMap<K, V>
where
    K: Copy + Ord + Debug,
    V: Clone + Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BTreeMap")
            .field("data", &format_args!("{:?}", self.data))
            .finish()
    }
}

pub struct BlockIndex<K: Clone + Ord, V: Clone> {
    map: Arc<PackedMemoryArray<Cell<K, V>>>,
    index_tree: BlockSearchTree<K, V>,
}

unsafe impl<K: Clone + Ord, V: Clone> Send for BlockIndex<K, V> {}
unsafe impl<K: Clone + Ord, V: Clone> Sync for BlockIndex<K, V> {}

impl<K, V> Debug for BlockIndex<K, V>
where
    K: Clone + Ord + Debug,
    V: Clone + Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlockIndex")
            .field("map", &format_args!("{:?}", self.map))
            .field("index_tree", &format_args!("{:?}", self.index_tree))
            .finish()
    }
}

impl<K, V> BlockIndex<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn get_block_for_insert<'a, Q>(&'a self, search_key: &Q) -> SearchResult<'a, K, V>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        self.index_tree.find(search_key, true)
    }

    pub fn get<'a, Q>(&self, search_key: &Q) -> Option<&'a V>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        match self.index_tree.find(search_key, false) {
            SearchResult::NotFound => return None,
            SearchResult::Block(block, ..) => {
                let iter = CellIterator::new(block.cell_slice_ptr, self.map.active_range.end);

                for cell_guard in iter {
                    if !cell_guard.is_empty() {
                        let cache = cell_guard.cache().unwrap().clone().unwrap();
                        let cache_key = cache.key.borrow();
                        if cache_key == search_key {
                            // todo: ABA problem
                            let value =
                                unsafe { (&*cell_guard.inner.value.get()).as_ref().unwrap() };
                            return Some(value);
                        } else if cache_key > search_key {
                            return None;
                        }
                    }
                }

                None
            }
            _ => unreachable!(),
        }
    }
}

struct BlockSearchTree<K: Clone + Ord, V: Clone> {
    nodes: Box<[UnsafeCell<Node<K, V>>]>,
    /// Precomputed subtree sizes for vEB layout navigation.
    /// Each entry stores (upper_subtree_size, lower_subtree_size) at each vEB recursion level.
    /// This enables O(log log N) child position computation without explicit pointers.
    subtree_sizes: Box<[SubtreeSizes]>,
    /// Height of the tree (number of levels from root to leaves).
    height: u8,
}

/// Stores the sizes of upper and lower subtrees at a given vEB recursion level.
/// Used for implicit child navigation in the vEB layout.
#[derive(Debug, Clone, Copy)]
struct SubtreeSizes {
    /// Number of nodes in the upper subtree (including root of that level).
    upper_size: usize,
    /// Number of nodes in each lower subtree.
    lower_size: usize,
}

/// Tracks navigation state within the vEB-layout tree.
/// This enables implicit child position computation without explicit pointers.
#[derive(Debug, Clone, Copy)]
struct VebNavigator {
    /// Current position in the flattened node array.
    position: usize,
    /// Base offset of the current vEB subtree in the node array.
    subtree_base: usize,
    /// Current depth within the vEB recursion (index into subtree_sizes).
    veb_depth: usize,
    /// Position within the current upper subtree (0-indexed from subtree's root).
    local_position: usize,
    /// Size of the current subtree (used for bounds checking at deepest level).
    current_subtree_size: usize,
}

impl VebNavigator {
    /// Creates a navigator starting at the root of the tree.
    fn at_root(tree_size: usize) -> Self {
        VebNavigator {
            position: 0,
            subtree_base: 0,
            veb_depth: 0,
            local_position: 0,
            current_subtree_size: tree_size,
        }
    }

    /// Computes the position of the left child and returns a new navigator for it.
    ///
    /// # Arguments
    /// * `subtree_sizes` - The precomputed subtree sizes for vEB navigation
    ///
    /// # Returns
    /// A new `VebNavigator` pointing to the left child position.
    fn left_child(&self, subtree_sizes: &[SubtreeSizes]) -> VebNavigator {
        self.child(subtree_sizes, false)
    }

    /// Computes the position of the right child and returns a new navigator for it.
    ///
    /// # Arguments
    /// * `subtree_sizes` - The precomputed subtree sizes for vEB navigation
    ///
    /// # Returns
    /// A new `VebNavigator` pointing to the right child position.
    fn right_child(&self, subtree_sizes: &[SubtreeSizes]) -> VebNavigator {
        self.child(subtree_sizes, true)
    }

    /// Checks if the current position is valid (within the bounds of its subtree).
    /// This should be called after navigating to check if the resulting position
    /// actually exists in the tree.
    fn is_valid(&self) -> bool {
        self.local_position < self.current_subtree_size
    }

    /// Computes child position within the vEB layout.
    ///
    /// The vEB layout organizes nodes as:
    /// 1. Upper subtree nodes are stored first (contiguously)
    /// 2. Lower subtrees follow, in order of their parent leaves in the upper subtree
    ///
    /// Within the upper subtree, standard binary tree indexing applies:
    /// - Left child of node at local position `p` is at `2*p + 1`
    /// - Right child of node at local position `p` is at `2*p + 2`
    ///
    /// When transitioning from upper to lower subtree:
    /// - We jump past the upper subtree
    /// - Then index into the correct lower subtree based on which leaf we came from
    fn child(&self, subtree_sizes: &[SubtreeSizes], is_right: bool) -> VebNavigator {
        if self.veb_depth >= subtree_sizes.len() {
            // At the deepest vEB level, use simple binary tree arithmetic
            // within the current subtree bounds
            let child_local = if is_right {
                2 * self.local_position + 2
            } else {
                2 * self.local_position + 1
            };

            return VebNavigator {
                position: self.subtree_base + child_local,
                subtree_base: self.subtree_base,
                veb_depth: self.veb_depth,
                local_position: child_local,
                current_subtree_size: self.current_subtree_size,
            };
        }

        let sizes = &subtree_sizes[self.veb_depth];
        let upper_size = sizes.upper_size;
        let lower_size = sizes.lower_size;

        // Child position in standard binary tree indexing within upper subtree
        let child_local = if is_right {
            2 * self.local_position + 2
        } else {
            2 * self.local_position + 1
        };

        if child_local < upper_size {
            // Child is still within the upper subtree
            VebNavigator {
                position: self.subtree_base + child_local,
                subtree_base: self.subtree_base,
                veb_depth: self.veb_depth,
                local_position: child_local,
                current_subtree_size: self.current_subtree_size,
            }
        } else {
            // Child transitions to a lower subtree
            // Compute which lower subtree (based on leaf index in upper subtree)
            // In a complete binary tree, leaves are at positions [n/2, n-1] where n = size
            // The leaf index (0-indexed) is: child_local - upper_size
            let leaf_index = child_local - upper_size;

            // Lower subtrees are stored after the upper subtree, each of size lower_size
            let lower_subtree_base = self.subtree_base + upper_size + leaf_index * lower_size;

            VebNavigator {
                position: lower_subtree_base,
                subtree_base: lower_subtree_base,
                veb_depth: self.veb_depth + 1,
                local_position: 0,
                current_subtree_size: lower_size,
            }
        }
    }
}

/// Precomputes subtree sizes at each vEB recursion level for a tree of the given height.
///
/// The vEB layout recursively splits a tree of height h into:
/// - An "upper" subtree of height ceil(h/2) containing the top portion
/// - Multiple "lower" subtrees of height floor(h/2) hanging off the upper leaves
///
/// This function returns the sizes at each recursion level, enabling O(1) child
/// position computation during navigation without explicit pointers.
///
/// # Arguments
/// * `total_height` - The height of the complete binary tree (levels from root to leaves)
///
/// # Returns
/// A boxed slice where each entry contains (upper_size, lower_size) for that vEB level.
/// The slice has O(log log N) entries since each recursion roughly halves the height.
fn precompute_subtree_sizes(total_height: u8) -> Box<[SubtreeSizes]> {
    if total_height == 0 {
        return Box::new([]);
    }

    let mut sizes = Vec::new();
    let mut h = total_height as usize;

    while h > 1 {
        // Split: upper gets ceil(h/2), lower gets floor(h/2)
        // For vEB, we want the lower height to be a power of 2 when possible
        let lower_h = if h.is_power_of_two() {
            h / 2
        } else {
            // Round lower_h down to nearest power of 2 for cache-oblivious properties
            (h / 2).next_power_of_two() / 2
        }
        .max(1);
        let upper_h = h - lower_h;

        // Size of a complete binary tree of height h is (2^h - 1) nodes
        let upper_size = (1usize << upper_h) - 1;
        let lower_size = (1usize << lower_h) - 1;

        sizes.push(SubtreeSizes {
            upper_size,
            lower_size,
        });
        h = lower_h;
    }

    sizes.into_boxed_slice()
}

/// Computes the height of a complete binary tree with the given number of nodes.
/// Height is defined as the number of levels (a single node has height 1).
fn compute_tree_height(node_count: usize) -> u8 {
    if node_count == 0 {
        return 0;
    }
    // For a complete binary tree with n nodes: height = floor(log2(n + 1))
    // Since node_count = 2^h - 1 for a full tree, h = log2(node_count + 1)
    ((node_count + 1).ilog2()) as u8
}

impl<'a, K, V> BlockSearchTree<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn new(cells: Arc<PackedMemoryArray<Cell<K, V>>>) -> BlockSearchTree<K, V> {
        let mut nodes = Self::allocate(cells.len());

        let mut leaves = Self::initialize_nodes(&mut *nodes, None);
        let slot_size = f32::log2(cells.requested_capacity as f32) as usize; // https://github.com/rust-lang/rust/issues/70887
        let mut slots = cells.as_slice().chunks_exact(slot_size);

        for leaf in leaves.iter_mut() {
            Self::finalize_leaf_node(leaf.get_mut(), slots.next().unwrap());
        }

        let initialized_nodes = unsafe { nodes.assume_init() };
        let node_count = initialized_nodes.len();
        let height = compute_tree_height(node_count);
        let subtree_sizes = precompute_subtree_sizes(height);

        BlockSearchTree {
            nodes: initialized_nodes,
            subtree_sizes,
            height,
        }
    }

    fn allocate(leaf_count: usize) -> Box<[MaybeUninit<UnsafeCell<Node<K, V>>>]> {
        let size = leaf_count;
        // https://github.com/rust-lang/rust/issues/70887
        let slot_size = f32::log2(size as f32) as usize;
        let leaf_count = size / slot_size;
        let node_count = 2 * leaf_count - 1;
        // println!("tree has {:?} leaves, {:?} nodes", leaf_count, node_count);
        Box::<[UnsafeCell<Node<K, V>>]>::new_uninit_slice(node_count as usize)
    }

    fn initialize_nodes<'b>(
        nodes: &'b mut [MaybeUninit<UnsafeCell<Node<K, V>>>],
        parent_subtree: Option<*mut NonNull<UnsafeCell<Node<K, V>>>>,
    ) -> Vec<&'b mut UnsafeCell<Node<K, V>>> {
        if nodes.len() <= 3 {
            return Self::assign_node_values(nodes, parent_subtree);
        }

        let (upper_mem, lower_mem) = Self::split_tree_memory(nodes);
        let num_lower_branches = upper_mem.len() + 1;

        let leaves_of_upper = Self::initialize_nodes(upper_mem, parent_subtree);

        let nodes_per_branch = lower_mem.len() / num_lower_branches;
        let mut branches = lower_mem.chunks_exact_mut(nodes_per_branch);

        leaves_of_upper
            .into_iter()
            .flat_map(|subtree_leaf| match unsafe { &mut *subtree_leaf.get() } {
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
        nodes: &'b mut [MaybeUninit<UnsafeCell<Node<K, V>>>],
        parent_subtree: Option<*mut NonNull<UnsafeCell<Node<K, V>>>>,
    ) -> Vec<&'b mut UnsafeCell<Node<K, V>>> {
        let num_nodes = nodes.len();
        assert!(num_nodes <= 3);

        for i in 0..num_nodes as usize {
            nodes[i].write(UnsafeCell::new(Node::Internal {
                min_rhs: Key::Supremum,
                left: MaybeUninit::uninit(), // Todo: NonNull<MaybeUninit<T>>
                right: MaybeUninit::uninit(),
            }));
        }

        if num_nodes == 1 {
            let node = unsafe { nodes[0].assume_init_mut() };
            return vec![node];
        }

        let left_node =
            NonNull::new(unsafe { nodes[1].assume_init_ref() as *const _ } as *mut _).unwrap();
        let right_node =
            NonNull::new(unsafe { nodes[2].assume_init_ref() as *const _ } as *mut _).unwrap();

        nodes[0].write(UnsafeCell::new(Node::Internal {
            min_rhs: Key::Supremum,
            left: MaybeUninit::new(left_node),
            right: MaybeUninit::new(right_node),
        }));

        let (lhs, rhs) = nodes.split_at_mut(2);

        if let Some(node) = parent_subtree {
            unsafe {
                NonNull::new((lhs[0].assume_init_ref().get()) as *mut _)
                    .map(|p| node.write(p))
                    .or_else(|| panic!("Pointer is null!"));
            }
        }

        let left_branch = unsafe { lhs[1].assume_init_mut() };
        let right_branch = unsafe { rhs[0].assume_init_mut() };

        return vec![left_branch, right_branch];
    }

    fn split_tree_memory<'b>(
        nodes: &'b mut [MaybeUninit<UnsafeCell<Node<K, V>>>],
    ) -> (
        &'b mut [MaybeUninit<UnsafeCell<Node<K, V>>>],
        &'b mut [MaybeUninit<UnsafeCell<Node<K, V>>>],
    ) {
        let height = f32::log2(nodes.len() as f32 + 1f32);
        let lower_height = ((height / 2f32).ceil() as u32).next_power_of_two();
        let upper_height = height as u32 - lower_height;

        let upper_subtree_length = 2 << (upper_height - 1);
        nodes.split_at_mut(upper_subtree_length - 1)
    }

    fn finalize_leaf_node<'b>(leaf: &'b mut Node<K, V>, leaf_mem: &'b [Cell<K, V>]) -> () {
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
                };
                *leaf = Node::Leaf(min_key, block);
            }
            Node::Leaf(_, _) => (),
        };
    }

    fn root(&'a self) -> &'a Node<K, V> {
        unsafe { &*self.nodes[0].get() }
    }

    /// Creates a navigator starting at the root of the tree.
    fn root_navigator(&self) -> VebNavigator {
        VebNavigator::at_root(self.nodes.len())
    }

    /// Gets the node at the given navigator position.
    fn node_at(&'a self, nav: &VebNavigator) -> &'a Node<K, V> {
        unsafe { &*self.nodes[nav.position].get() }
    }

    /// Computes the left child position using implicit vEB navigation.
    fn left_child(&self, nav: &VebNavigator) -> VebNavigator {
        nav.left_child(&self.subtree_sizes)
    }

    /// Computes the right child position using implicit vEB navigation.
    fn right_child(&self, nav: &VebNavigator) -> VebNavigator {
        nav.right_child(&self.subtree_sizes)
    }

    fn find<Q>(&'a self, search_key: &Q, for_insertion: bool) -> SearchResult<'a, K, V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self.root()
            .search_to_block(Key::Value(search_key), for_insertion)
    }
}

impl<K, V> Debug for BlockSearchTree<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone + Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlockSearchTree")
            .field("nodes", &format_args!("{:?}", self.nodes))
            .finish()
    }
}

enum Node<K: Clone + Ord, V: Clone> {
    Leaf(Key<K>, Block<K, V>),
    Internal {
        min_rhs: Key<K>,
        left: MaybeUninit<NonNull<UnsafeCell<Node<K, V>>>>,
        right: MaybeUninit<NonNull<UnsafeCell<Node<K, V>>>>,
    },
}

impl<K, V> Node<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn search<'a, Q>(&'a self, key: Key<&Q>, allow_empty: bool) -> SearchResult<'a, K, V>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        match self {
            Node::Leaf(key_lock, block) => {
                if !allow_empty && *key_lock == Key::Supremum {
                    SearchResult::NotFound
                } else {
                    SearchResult::Block(block, Key::from(key_lock))
                }
            }
            Node::Internal {
                min_rhs,
                left,
                right,
                ..
            } => {
                let node = if min_rhs.is_supremum()
                    || key < Key::Value(min_rhs.clone().unwrap().borrow())
                {
                    unsafe { &*left.assume_init_ref() }
                } else {
                    unsafe { &*right.assume_init_ref() }
                };
                SearchResult::Internal(unsafe { &*node.as_ref().get() })
            }
        }
    }

    fn search_to_block<'a, Q>(&'a self, key: Key<&Q>, allow_empty: bool) -> SearchResult<'a, K, V>
    where
        Q: Ord,
        K: Borrow<Q>,
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

impl<K, V> Debug for Node<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone + Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Leaf(key, ..) => formatter
                .debug_struct("Node::Leaf")
                .field("key", &format_args!("{:?}", key))
                .finish(),
            Node::Internal { min_rhs, .. } => formatter
                .debug_struct("Node::Internal")
                .field("min_rhs", &format_args!("{:?}", min_rhs))
                .finish(),
        }
    }
}

enum SearchResult<'a, K: Clone + Ord, V: Clone> {
    Block(&'a Block<K, V>, Key<&'a K>),
    Internal(&'a Node<K, V>),
    NotFound,
}

struct Block<K: Clone, V: Clone> {
    cell_slice_ptr: *const Cell<K, V>,
    length: usize,
}

unsafe impl<K: Clone, V: Clone> Send for Block<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_tree_height() {
        // Height 0 for empty tree
        assert_eq!(compute_tree_height(0), 0);

        // Height 1 for single node
        assert_eq!(compute_tree_height(1), 1);

        // Height 2 for 3 nodes (complete binary tree: 2^2 - 1 = 3)
        assert_eq!(compute_tree_height(3), 2);

        // Height 3 for 7 nodes (complete binary tree: 2^3 - 1 = 7)
        assert_eq!(compute_tree_height(7), 3);

        // Height 4 for 15 nodes (complete binary tree: 2^4 - 1 = 15)
        assert_eq!(compute_tree_height(15), 4);

        // Height 5 for 31 nodes (complete binary tree: 2^5 - 1 = 31)
        assert_eq!(compute_tree_height(31), 5);
    }

    #[test]
    fn test_precompute_subtree_sizes_empty() {
        let sizes = precompute_subtree_sizes(0);
        assert_eq!(sizes.len(), 0);
    }

    #[test]
    fn test_precompute_subtree_sizes_height_1() {
        // A tree with height 1 has only the root, no recursion needed
        let sizes = precompute_subtree_sizes(1);
        assert_eq!(sizes.len(), 0);
    }

    #[test]
    fn test_precompute_subtree_sizes_height_2() {
        // Height 2: split into upper(1) + lower(1)
        // upper_size = 2^1 - 1 = 1, lower_size = 2^1 - 1 = 1
        let sizes = precompute_subtree_sizes(2);
        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0].upper_size, 1);
        assert_eq!(sizes[0].lower_size, 1);
    }

    #[test]
    fn test_precompute_subtree_sizes_height_3() {
        // Height 3: split into upper(2) + lower(1), then upper recursion ends at height 2
        let sizes = precompute_subtree_sizes(3);
        // First level: h=3, lower_h=1, upper_h=2
        // upper_size = 2^2 - 1 = 3, lower_size = 2^1 - 1 = 1
        assert!(sizes.len() >= 1);
        assert_eq!(sizes[0].upper_size, 3);
        assert_eq!(sizes[0].lower_size, 1);
    }

    #[test]
    fn test_precompute_subtree_sizes_height_4() {
        // Height 4 (power of 2): split into upper(2) + lower(2)
        let sizes = precompute_subtree_sizes(4);
        assert!(sizes.len() >= 1);
        // h=4, split evenly: upper_h=2, lower_h=2
        // upper_size = 2^2 - 1 = 3, lower_size = 2^2 - 1 = 3
        assert_eq!(sizes[0].upper_size, 3);
        assert_eq!(sizes[0].lower_size, 3);

        // Second level: h=2, upper_h=1, lower_h=1
        assert!(sizes.len() >= 2);
        assert_eq!(sizes[1].upper_size, 1);
        assert_eq!(sizes[1].lower_size, 1);
    }

    #[test]
    fn test_precompute_subtree_sizes_height_8() {
        // Height 8 (power of 2): demonstrates logarithmic recursion depth
        let sizes = precompute_subtree_sizes(8);

        // First level: h=8, split evenly: upper_h=4, lower_h=4
        // upper_size = 2^4 - 1 = 15, lower_size = 2^4 - 1 = 15
        assert!(sizes.len() >= 1);
        assert_eq!(sizes[0].upper_size, 15);
        assert_eq!(sizes[0].lower_size, 15);

        // Second level: h=4, upper_h=2, lower_h=2
        assert!(sizes.len() >= 2);
        assert_eq!(sizes[1].upper_size, 3);
        assert_eq!(sizes[1].lower_size, 3);

        // Third level: h=2, upper_h=1, lower_h=1
        assert!(sizes.len() >= 3);
        assert_eq!(sizes[2].upper_size, 1);
        assert_eq!(sizes[2].lower_size, 1);
    }

    #[test]
    fn test_subtree_sizes_count_matches_expected() {
        // For power-of-2 heights, we expect log2(h) recursion levels
        // Height 2: 1 level
        assert_eq!(precompute_subtree_sizes(2).len(), 1);
        // Height 4: 2 levels (4->2->done)
        assert_eq!(precompute_subtree_sizes(4).len(), 2);
        // Height 8: 3 levels (8->4->2->done)
        assert_eq!(precompute_subtree_sizes(8).len(), 3);
        // Height 16: 4 levels (16->8->4->2->done)
        assert_eq!(precompute_subtree_sizes(16).len(), 4);
    }

    #[test]
    fn test_subtree_sizes_total_matches_tree() {
        // Verify that upper + lower subtrees can reconstruct the tree structure
        // For a complete binary tree of height h:
        // - Upper subtree has (2^upper_h - 1) nodes
        // - There are 2^upper_h leaves in upper, each pointing to a lower subtree
        // - Each lower subtree has (2^lower_h - 1) nodes
        // - Total = upper_size + 2^upper_h * lower_size = 2^h - 1

        for h in 2..=10u8 {
            let sizes = precompute_subtree_sizes(h);
            if !sizes.is_empty() {
                let s = &sizes[0];
                let upper_h = ((s.upper_size + 1) as f64).log2() as usize;
                let num_lower_subtrees = 1 << upper_h;
                let total = s.upper_size + num_lower_subtrees * s.lower_size;
                let expected_total = (1 << h) - 1;
                assert_eq!(
                    total, expected_total,
                    "For height {}: upper_size={} + {} * lower_size={} = {}, expected {}",
                    h, s.upper_size, num_lower_subtrees, s.lower_size, total, expected_total
                );
            }
        }
    }

    // === VebNavigator tests for Step 2: Implicit child navigation ===

    #[test]
    fn test_veb_navigator_at_root() {
        let nav = VebNavigator::at_root(15);
        assert_eq!(nav.position, 0);
        assert_eq!(nav.subtree_base, 0);
        assert_eq!(nav.veb_depth, 0);
        assert_eq!(nav.local_position, 0);
        assert_eq!(nav.current_subtree_size, 15);
    }

    #[test]
    fn test_veb_navigator_height_2() {
        // Height 2 tree: 3 nodes
        // vEB layout for h=2: [root, left, right] (same as standard)
        // upper_size=1 (root), lower_size=1 (each child)
        let sizes = precompute_subtree_sizes(2);
        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0].upper_size, 1);
        assert_eq!(sizes[0].lower_size, 1);

        let node_count = 3;
        let root = VebNavigator::at_root(node_count);
        assert_eq!(root.position, 0);

        // Left child: transitions from upper (size 1) to first lower subtree
        // Layout: [0: root] [1: left] [2: right]
        let left = root.left_child(&sizes);
        assert_eq!(left.position, 1, "Left child should be at position 1");
        assert_eq!(
            left.veb_depth, 1,
            "Left child should be in deeper vEB level"
        );
        assert_eq!(left.current_subtree_size, 1, "Lower subtree has size 1");

        // Right child: transitions from upper (size 1) to second lower subtree
        let right = root.right_child(&sizes);
        assert_eq!(right.position, 2, "Right child should be at position 2");
        assert_eq!(
            right.veb_depth, 1,
            "Right child should be in deeper vEB level"
        );
    }

    #[test]
    fn test_veb_navigator_height_3() {
        // Height 3 tree: 7 nodes
        // vEB split: upper_h=2 (3 nodes), lower_h=1 (1 node each)
        // Layout: [upper: 0,1,2] [lower0: 3] [lower1: 4] [lower2: 5] [lower3: 6]
        let sizes = precompute_subtree_sizes(3);
        assert_eq!(sizes[0].upper_size, 3);
        assert_eq!(sizes[0].lower_size, 1);

        let node_count = 7;
        let root = VebNavigator::at_root(node_count);
        assert_eq!(root.position, 0);

        // Within upper subtree, standard binary tree indexing
        let left = root.left_child(&sizes);
        assert_eq!(left.position, 1, "Left child of root in upper subtree");
        assert_eq!(left.veb_depth, 0, "Still in first vEB level");

        let right = root.right_child(&sizes);
        assert_eq!(right.position, 2, "Right child of root in upper subtree");
        assert_eq!(right.veb_depth, 0, "Still in first vEB level");

        // From node 1 (left child of root), descend to lower subtrees
        // Node 1's left child has local_position 2*1+1=3, which equals upper_size=3
        // So it transitions to lower subtree at leaf_index=0
        let left_left = left.left_child(&sizes);
        assert_eq!(
            left_left.position, 3,
            "Left-left should be in first lower subtree"
        );
        assert_eq!(left_left.veb_depth, 1, "Transitioned to deeper vEB level");

        // Node 1's right child has local_position 2*1+2=4, > upper_size=3
        // leaf_index = 4 - 3 = 1, so second lower subtree
        let left_right = left.right_child(&sizes);
        assert_eq!(
            left_right.position, 4,
            "Left-right should be in second lower subtree"
        );

        // Node 2's children
        let right_left = right.left_child(&sizes);
        assert_eq!(
            right_left.position, 5,
            "Right-left should be in third lower subtree"
        );

        let right_right = right.right_child(&sizes);
        assert_eq!(
            right_right.position, 6,
            "Right-right should be in fourth lower subtree"
        );
    }

    #[test]
    fn test_veb_navigator_height_4() {
        // Height 4 tree: 15 nodes
        // vEB split: upper_h=2 (3 nodes), lower_h=2 (3 nodes each)
        // Layout: [upper: 0,1,2] [lower0: 3,4,5] [lower1: 6,7,8] [lower2: 9,10,11] [lower3: 12,13,14]
        let sizes = precompute_subtree_sizes(4);
        assert_eq!(sizes[0].upper_size, 3);
        assert_eq!(sizes[0].lower_size, 3);

        let node_count = 15;
        let root = VebNavigator::at_root(node_count);

        // Navigate within upper subtree
        let left = root.left_child(&sizes);
        assert_eq!(left.position, 1);
        let right = root.right_child(&sizes);
        assert_eq!(right.position, 2);

        // Transition to lower subtrees
        // From node 1 (local_pos=1), left child has local_pos=3, >= upper_size=3
        // leaf_index = 3 - 3 = 0, lower subtree at base = 3 + 0*3 = 3
        let left_left = left.left_child(&sizes);
        assert_eq!(
            left_left.position, 3,
            "Should be root of first lower subtree"
        );
        assert_eq!(left_left.subtree_base, 3);
        assert_eq!(left_left.veb_depth, 1);

        // Within the lower subtree, continue navigation
        // Lower subtree has sizes[1] = {upper: 1, lower: 1}
        let lll = left_left.left_child(&sizes);
        assert_eq!(lll.position, 4, "Left child within lower subtree");

        let llr = left_left.right_child(&sizes);
        assert_eq!(llr.position, 5, "Right child within lower subtree");

        // Check other lower subtrees
        let left_right = left.right_child(&sizes);
        assert_eq!(left_right.position, 6, "Root of second lower subtree");
        assert_eq!(left_right.subtree_base, 6);

        let right_left = right.left_child(&sizes);
        assert_eq!(right_left.position, 9, "Root of third lower subtree");

        let right_right = right.right_child(&sizes);
        assert_eq!(right_right.position, 12, "Root of fourth lower subtree");
    }

    #[test]
    fn test_veb_navigator_all_positions_reachable() {
        // For a complete binary tree, verify all positions are reachable via navigation
        for height in 2..=5u8 {
            let sizes = precompute_subtree_sizes(height);
            let node_count = (1usize << height) - 1;
            let mut visited = vec![false; node_count];

            // BFS through the tree using navigation
            let mut queue = vec![VebNavigator::at_root(node_count)];
            while let Some(nav) = queue.pop() {
                if !nav.is_valid() {
                    continue;
                }
                visited[nav.position] = true;

                // Add children if they're within bounds
                let left = nav.left_child(&sizes);
                let right = nav.right_child(&sizes);

                if left.is_valid() {
                    queue.push(left);
                }
                if right.is_valid() {
                    queue.push(right);
                }
            }

            // All positions should be visited
            for (i, &v) in visited.iter().enumerate() {
                assert!(v, "Height {}: Position {} not reachable", height, i);
            }
        }
    }

    #[test]
    fn test_veb_navigator_no_duplicate_positions() {
        // Verify that left and right children are always different
        for height in 2..=5u8 {
            let sizes = precompute_subtree_sizes(height);
            let node_count = (1usize << height) - 1;

            let mut queue = vec![VebNavigator::at_root(node_count)];
            while let Some(nav) = queue.pop() {
                let left = nav.left_child(&sizes);
                let right = nav.right_child(&sizes);

                if left.is_valid() && right.is_valid() {
                    assert_ne!(
                        left.position, right.position,
                        "Height {}: Node at {} has same position for both children: {}",
                        height, nav.position, left.position
                    );
                    queue.push(left);
                    queue.push(right);
                }
            }
        }
    }

    #[test]
    fn test_veb_navigator_leaf_count() {
        // The number of leaves (nodes with children out of bounds) should be 2^(h-1)
        for height in 2..=5u8 {
            let sizes = precompute_subtree_sizes(height);
            let node_count = (1usize << height) - 1;
            let expected_leaf_count = 1usize << (height - 1);

            let mut leaf_count = 0;
            let mut queue = vec![VebNavigator::at_root(node_count)];
            let mut visited = vec![false; node_count];

            while let Some(nav) = queue.pop() {
                if !nav.is_valid() || visited[nav.position] {
                    continue;
                }
                visited[nav.position] = true;

                let left = nav.left_child(&sizes);
                let right = nav.right_child(&sizes);

                // A leaf has both children invalid (out of subtree bounds)
                let left_invalid = !left.is_valid();
                let right_invalid = !right.is_valid();

                if left_invalid && right_invalid {
                    leaf_count += 1;
                } else {
                    // Only add children that are valid
                    if !left_invalid {
                        queue.push(left);
                    }
                    if !right_invalid {
                        queue.push(right);
                    }
                }
            }

            assert_eq!(
                leaf_count, expected_leaf_count,
                "Height {}: Expected {} leaves, found {}",
                height, expected_leaf_count, leaf_count
            );
        }
    }

    #[test]
    fn test_veb_navigator_is_valid() {
        // Test the is_valid method
        let sizes = precompute_subtree_sizes(2);
        let node_count = 3;
        let root = VebNavigator::at_root(node_count);

        // Root should be valid
        assert!(root.is_valid());

        // Children of root should be valid
        let left = root.left_child(&sizes);
        let right = root.right_child(&sizes);
        assert!(left.is_valid());
        assert!(right.is_valid());

        // Children of leaves (positions 1 and 2) should be invalid
        // since they're single-node subtrees with no children
        let left_left = left.left_child(&sizes);
        let left_right = left.right_child(&sizes);
        assert!(!left_left.is_valid(), "Child of leaf should be invalid");
        assert!(!left_right.is_valid(), "Child of leaf should be invalid");
    }
}
