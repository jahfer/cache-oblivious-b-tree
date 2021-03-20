use criterion::{black_box, criterion_group, Criterion};
// use rand::{Rng, prelude::Distribution};
// use rand::distributions::Uniform;
use std::collections::BTreeMap;
use cache_oblivious_b_tree::BTreeMap as COBTreeMap;

fn compare_reads(c: &mut Criterion) {
  let mut std_map = BTreeMap::new();
  std_map.insert(5, "Hello");

  let mut co_btree_map = COBTreeMap::new(30);
  co_btree_map.insert(5, "Hello");

  let mut group = c.benchmark_group("Read");

  group.bench_function("StaticSearchTree", |b| {
    b.iter(|| co_btree_map.get(&5))
  });

  group.bench_function("BTreeMap", |b| {
    b.iter(|| std_map.get(&5))
  });

  group.finish()
}

fn compare_writes(c: &mut Criterion) {
  let mut std_map = BTreeMap::new();
  let mut co_btree_map = COBTreeMap::new(30);

  let mut group = c.benchmark_group("Overwrite");

  group.bench_function("StaticSearchTree", |b| {
    b.iter(|| co_btree_map.insert(5, black_box("Hello")))
  });

  group.bench_function("BTreeMap", |b| {
    b.iter(|| std_map.insert(5, black_box("Hello")))
  });

  group.finish()
}

fn compare_seqential_writes(c: &mut Criterion) {
  let mut std_map = BTreeMap::new();
  let mut co_btree_map = COBTreeMap::new(100);

  let mut group = c.benchmark_group("Sequential Write");

  group.bench_function("StaticSearchTree", |b| {
    let mut counter = 0u8;
    b.iter(|| {
      counter = black_box((counter + 1) % 100);
      co_btree_map.insert(counter, black_box("A"))
    })
  });

  group.bench_function("BTreeMap", |b| {
    let mut counter = 0u8;
    b.iter(|| {
      counter = black_box((counter + 1) % 100);
      std_map.insert(counter, black_box("A"))
    })
  });

  group.finish()
}

// fn compare_uniform_random_writes(c: &mut Criterion) {
//   let mut std_map = BTreeMap::new();
//   let mut co_sst_map = StaticSearchTree::new(100);

//   let mut group = c.benchmark_group("Uniform Random Write");

//   group.bench_function("StaticSearchTree", |b| {
//     let mut rng = rand::thread_rng();
//     let uniform = Uniform::from(1..100u8);
//     b.iter(|| {
//       let rand_key = black_box(uniform.sample(&mut rng));
//       co_sst_map.add(rand_key, black_box("A"))
//     })
//   });

//   group.bench_function("BTreeMap", |b| {
//     let mut rng = rand::thread_rng();
//     let uniform = Uniform::from(1..100u8);
//     b.iter(|| {
//       let rand_key = black_box(uniform.sample(&mut rng));
//       std_map.insert(rand_key, black_box("A"))
//     })
//   });

//   group.finish()
// }

criterion_group!(
  benches,
  compare_reads,
  compare_writes,
  compare_seqential_writes,
  // compare_uniform_random_writes
);
