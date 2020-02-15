use cache_oblivious_b_tree::CacheObliviousBTreeMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let mut map = CacheObliviousBTreeMap::new();

    c.bench_function("overwrite", |b| b.iter(||
      map.insert(black_box(5), black_box("Hello"))
    ));

    map.insert(5, "Hello");
    c.bench_function("read", |b| b.iter(||
      map.get(black_box(5))
    ));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
