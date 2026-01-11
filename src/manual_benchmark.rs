use cache_oblivious_b_tree::BTreeMap;
use std::time::Instant;

const NUM_ELEMENTS: usize = 1_000_000;

fn main() {
    println!("Cache-Oblivious B-Tree Benchmark");
    println!("================================\n");

    // Benchmark sequential insertions
    benchmark_sequential_insert();

    // Benchmark random insertions
    benchmark_random_insert();

    // Benchmark lookups
    benchmark_lookups();

    // Benchmark mixed operations
    benchmark_mixed_operations();

    // Benchmark iteration
    benchmark_iteration();
}

fn benchmark_sequential_insert() {
    print!("Sequential insert ({} elements)... ", NUM_ELEMENTS);
    let mut tree = BTreeMap::new(16);

    let start = Instant::now();
    for i in 0..NUM_ELEMENTS {
        tree.insert(i as u64, i);
    }
    let duration = start.elapsed();

    println!(
        "{:?} ({:.2} ops/sec)",
        duration,
        NUM_ELEMENTS as f64 / duration.as_secs_f64()
    );
}

fn benchmark_random_insert() {
    print!("Random insert ({} elements)... ", NUM_ELEMENTS);
    let mut tree = BTreeMap::new(16);

    // Generate pseudo-random keys using simple LCG
    let mut rng_state: u64 = 12345;
    let keys: Vec<u64> = (0..NUM_ELEMENTS)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state
        })
        .collect();

    let start = Instant::now();
    for key in &keys {
        tree.insert(*key, 1);
    }
    let duration = start.elapsed();

    println!(
        "{:?} ({:.2} ops/sec)",
        duration,
        NUM_ELEMENTS as f64 / duration.as_secs_f64()
    );
}

fn benchmark_lookups() {
    // First, build the tree
    let mut tree = BTreeMap::new(16);
    for i in 0..NUM_ELEMENTS {
        tree.insert(i as u64, i);
    }

    // Benchmark sequential lookups
    print!("Sequential lookup ({} elements)... ", NUM_ELEMENTS);
    let start = Instant::now();
    for i in 0..NUM_ELEMENTS {
        let _ = tree.get(&(i as u64));
    }
    let duration = start.elapsed();
    println!(
        "{:?} ({:.2} ops/sec)",
        duration,
        NUM_ELEMENTS as f64 / duration.as_secs_f64()
    );

    // Benchmark random lookups
    print!("Random lookup ({} elements)... ", NUM_ELEMENTS);
    let mut rng_state: u64 = 67890;
    let start = Instant::now();
    for _ in 0..NUM_ELEMENTS {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let key = rng_state % (NUM_ELEMENTS as u64);
        let _ = tree.get(&key);
    }
    let duration = start.elapsed();
    println!(
        "{:?} ({:.2} ops/sec)",
        duration,
        NUM_ELEMENTS as f64 / duration.as_secs_f64()
    );

    // Benchmark missing key lookups
    print!("Missing key lookup ({} elements)... ", NUM_ELEMENTS);
    let start = Instant::now();
    for i in NUM_ELEMENTS..(NUM_ELEMENTS * 2) {
        let _ = tree.get(&(i as u64));
    }
    let duration = start.elapsed();
    println!(
        "{:?} ({:.2} ops/sec)",
        duration,
        NUM_ELEMENTS as f64 / duration.as_secs_f64()
    );
}

fn benchmark_mixed_operations() {
    print!("Mixed insert/lookup ({} operations)... ", NUM_ELEMENTS);
    let mut tree = BTreeMap::new(16);

    let start = Instant::now();
    for i in 0..NUM_ELEMENTS {
        tree.insert(i as u64, i);
        // Lookup a previous key every 10 inserts
        if i > 0 && i % 10 == 0 {
            let _ = tree.get(&((i / 2) as u64));
        }
    }
    let duration = start.elapsed();

    println!("{:?}", duration);
}

fn benchmark_iteration() {
    let mut tree = BTreeMap::new(16);
    for i in 0..NUM_ELEMENTS {
        tree.insert(i as u64, i);
    }

    // Simulate iteration by doing sequential lookups (iter not implemented)
    print!("Sequential access pattern ({} elements)... ", NUM_ELEMENTS);
    let start = Instant::now();
    let mut count = 0;
    for i in 0..NUM_ELEMENTS {
        if tree.get(&(i as u64)).is_some() {
            count += 1;
        }
    }
    let duration = start.elapsed();
    println!("{:?} (accessed {} elements)", duration, count);
}
