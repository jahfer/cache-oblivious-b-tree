use cache_oblivious_b_tree::BTreeMap as COBTreeMap;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::collections::BTreeMap;
use std::hint::black_box;

/// Dataset sizes to test - now supports larger sizes after allocation_size() overflow fix
const SIZES: &[usize] = &[100, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000];

/// Simple LCG for deterministic "random" access patterns without external deps
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_usize(&mut self, max: usize) -> usize {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as usize) % max
    }
}

/// Benchmark point lookups at various dataset sizes.
/// Cache-oblivious structures should show better scaling as data exceeds cache.
fn bench_point_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("Point Lookup");

    for &size in SIZES {
        // Pre-populate both maps
        let mut std_map = BTreeMap::new();
        let mut co_map = COBTreeMap::new(size);

        for i in 0..size {
            std_map.insert(i, i);
            co_map.insert(i, i);
        }

        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("COBTree", size), &size, |b, &size| {
            let mut rng = SimpleRng::new(42);
            b.iter(|| {
                let key = rng.next_usize(size);
                black_box(co_map.get(&key))
            })
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", size), &size, |b, &size| {
            let mut rng = SimpleRng::new(42);
            b.iter(|| {
                let key = rng.next_usize(size);
                black_box(std_map.get(&key))
            })
        });
    }

    group.finish();
}

/// Benchmark sequential insertions - tests PMA rebalancing efficiency.
fn bench_sequential_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sequential Insert");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("COBTree", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = COBTreeMap::new(size);
                for i in 0..size {
                    map.insert(i, black_box(i));
                }
                black_box(&map);
            })
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = BTreeMap::new();
                for i in 0..size {
                    map.insert(i, black_box(i));
                }
                black_box(&map);
            })
        });
    }

    group.finish();
}

/// Benchmark random insertions - stress tests incremental index updates.
fn bench_random_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Insert");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("COBTree", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = COBTreeMap::new(size);
                let mut rng = SimpleRng::new(12345);
                for _ in 0..size {
                    let key = rng.next_usize(size * 10); // sparse keys
                    map.insert(key, black_box(key));
                }
                black_box(&map);
            })
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = BTreeMap::new();
                let mut rng = SimpleRng::new(12345);
                for _ in 0..size {
                    let key = rng.next_usize(size * 10);
                    map.insert(key, black_box(key));
                }
                black_box(&map);
            })
        });
    }

    group.finish();
}

/// Benchmark strided access pattern - demonstrates cache-oblivious memory layout benefits.
/// Accessing every Nth element is particularly punishing for cache-unfriendly layouts.
fn bench_strided_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Strided Access");
    let size: usize = 2_000;
    let stride: usize = 127; // prime stride to avoid alignment patterns

    // Pre-populate
    let mut std_map = BTreeMap::new();
    let mut co_map = COBTreeMap::new(size);

    for i in 0..size {
        std_map.insert(i, i);
        co_map.insert(i, i);
    }

    let num_accesses = size / stride;
    group.throughput(Throughput::Elements(num_accesses as u64));

    group.bench_function("COBTree", |b| {
        b.iter(|| {
            let mut key = 0usize;
            for _ in 0..num_accesses {
                black_box(co_map.get(&key));
                key = (key + stride) % size;
            }
        })
    });

    group.bench_function("BTreeMap", |b| {
        b.iter(|| {
            let mut key = 0usize;
            for _ in 0..num_accesses {
                black_box(std_map.get(&key));
                key = (key + stride) % size;
            }
        })
    });

    group.finish();
}

/// Benchmark mixed read/write workload - realistic usage pattern.
/// 80% reads, 20% writes simulating a typical read-heavy workload.
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Workload (80R/20W)");

    for &size in &[500usize, 1_000, 2_000] {
        // Pre-populate with half the keys
        let mut std_map = BTreeMap::new();
        let mut co_map = COBTreeMap::new(size);

        for i in 0..(size / 2) {
            std_map.insert(i, i);
            co_map.insert(i, i);
        }

        let ops = 1000usize;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(BenchmarkId::new("COBTree", size), &size, |b, &size| {
            let mut rng = SimpleRng::new(999);
            b.iter(|| {
                for _ in 0..ops {
                    let key = rng.next_usize(size);
                    if rng.next_usize(100) < 80 {
                        // 80% read
                        black_box(co_map.get(&key));
                    } else {
                        // 20% write
                        co_map.insert(key, black_box(key));
                    }
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", size), &size, |b, &size| {
            let mut rng = SimpleRng::new(999);
            b.iter(|| {
                for _ in 0..ops {
                    let key = rng.next_usize(size);
                    if rng.next_usize(100) < 80 {
                        black_box(std_map.get(&key));
                    } else {
                        std_map.insert(key, black_box(key));
                    }
                }
            })
        });
    }

    group.finish();
}

/// Benchmark bulk loading followed by lookups - common initialization pattern.
fn bench_bulk_load_then_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bulk Load + Query");
    let size: usize = 50_000;
    let queries = 10_000usize;

    group.throughput(Throughput::Elements((size + queries) as u64));

    group.bench_function("COBTree", |b| {
        b.iter(|| {
            let mut map = COBTreeMap::new(size);
            // Bulk load
            for i in 0..size {
                map.insert(i, i);
            }
            // Query phase
            let mut rng = SimpleRng::new(777);
            for _ in 0..queries {
                let key = rng.next_usize(size);
                black_box(map.get(&key));
            }
        })
    });

    group.bench_function("BTreeMap", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in 0..size {
                map.insert(i, i);
            }
            let mut rng = SimpleRng::new(777);
            for _ in 0..queries {
                let key = rng.next_usize(size);
                black_box(map.get(&key));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_point_lookup,
    bench_sequential_insert,
    bench_random_insert,
    bench_strided_access,
    bench_mixed_workload,
    bench_bulk_load_then_query,
);
