use cache_oblivious_b_tree::StaticSearchTree;

fn main() {
  let mut tree = StaticSearchTree::<u8, &str>::new(16);

  for _ in 1..10_000_000 {
    tree.add(5, "Hello");
    tree.add(3, "World");
    tree.add(2, "!");

    tree.find(5);
    tree.find(3);
    tree.find(2);
  }
}