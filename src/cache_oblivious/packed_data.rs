use std::alloc::{alloc, dealloc, Layout};
use std::cmp::{Ord, Ordering, PartialOrd};
use std::fmt::{self, Debug};
use std::marker::Copy;
use std::ptr::NonNull;
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
pub enum Element<T: Eq> {
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
