# Cache-oblivious B-Tree Implementation

Notes on this project begin at http://jahfer.com/posts/co-btree-0/

## Current Status:

```rust
#[cfg(test)]
mod tests {
  use crate::StaticSearchTree;

  #[test]
  fn test() {
    let mut tree = StaticSearchTree::<u8, &str>::new(30);
    tree.add(6, "World!");
    tree.add(5, "Hello");
    assert_eq!(tree.find(5), Some("Hello"));
    assert_eq!(tree.find(6), Some("World!"));
  }
}
```
