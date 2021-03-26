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
    let value1 = "Hello";
    let value2 = "World";

    tree.insert(5, value1.to_string());
    tree.insert(6, value2.to_string());
    
    assert_eq!(tree.get(&5), Some(&value1));
    assert_eq!(tree.get(&6), Some(&value2));
  }
}
```
