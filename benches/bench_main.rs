use criterion::criterion_main;

mod benchmarks;

criterion_main! {
  benchmarks::static_search_tree::benches,
}