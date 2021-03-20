use cache_oblivious_b_tree::BTreeMap;

fn main() {
  let mut tree = BTreeMap::new(16);

  for _ in 1..10_000_000 {
    tree.insert(5, "Hello");
    tree.insert(3, "World");
    tree.insert(2, "!");

    tree.get(&5);
    tree.get(&3);
    tree.get(&2);
  }
}