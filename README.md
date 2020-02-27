# Cache-oblivious B-Tree Implementation

## Current Status:

```rust
#[test]
fn it_works() {
    let mut map = CacheObliviousBTreeMap::new();
    map.insert(5, "Hello");
    map.insert(3, "World");
    map.insert(2, "!");

    assert_eq!(map.get(5), Some("Hello"));
    assert_eq!(map.get(4), None);
    assert_eq!(map.get(3), Some("World"));
    assert_eq!(map.get(2), Some("!"));
}
```
