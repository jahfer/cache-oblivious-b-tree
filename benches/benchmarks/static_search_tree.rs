use criterion::{black_box, criterion_group, Criterion};
use std::collections::BTreeMap;
use cache_oblivious_b_tree::StaticSearchTree;

fn compare_reads(c: &mut Criterion) {
  let mut std_map = BTreeMap::new();
  std_map.insert(5, "Hello");

  let mut co_sst_map = StaticSearchTree::new(30);
  co_sst_map.add(5, "Hello");

  let mut group = c.benchmark_group("Read Key");

  group.bench_function("StaticSearchTree", |b| {
    b.iter(|| co_sst_map.find(5))
  });

  group.bench_function("BTreeMap", |b| {
    b.iter(|| std_map.get(&5))
  });

  group.finish()
}

fn compare_writes(c: &mut Criterion) {
  let mut std_map = BTreeMap::new();
  let mut co_sst_map = StaticSearchTree::new(30);

  let mut group = c.benchmark_group("Overwrite Key");

  group.bench_function("StaticSearchTree", |b| {
    b.iter(|| co_sst_map.add(5, black_box("Hello")))
  });

  group.bench_function("BTreeMap", |b| {
    b.iter(|| std_map.insert(5, black_box("Hello")))
  });

  group.finish()
}

criterion_group!(benches, compare_reads, compare_writes);
