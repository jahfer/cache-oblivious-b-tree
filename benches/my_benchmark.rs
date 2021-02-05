use cache_oblivious_b_tree::StaticSearchTree;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
  let mut map = StaticSearchTree::new(30);

  c.bench_function("overwrite", |b| {
    b.iter(|| map.add(black_box(5), black_box("Hello")))
  });

  map.add(5, "Hello");
  c.bench_function("read", |b| b.iter(|| map.find(black_box(5))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
