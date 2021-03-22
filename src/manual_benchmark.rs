use cache_oblivious_b_tree::BTreeMap;

fn main() {
  let mut tree = BTreeMap::new(16);

  for _ in 1..10_000_000 {
    tree.insert(5, "Hello");
  }
}