# Cache-oblivious B-Tree Implementation

Notes on this project begin at http://jahfer.com/posts/co-btree-0/

## Current Status:

```rust
#[cfg(test)]
mod tests {
  use crate::BTreeMap;

  #[test]
  fn test() {
    let mut tree = BTreeMap::<u8, String>::new(30);
    tree.insert(6, "World!");
    tree.insert(5, "Hello");
    assert_eq!(tree.get(&5), Some(&String::from("Hello")));
    assert_eq!(tree.get(&6), Some(&String::from("World!")));
  }
}
```
