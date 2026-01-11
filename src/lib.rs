#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod cache_oblivious;
pub use cache_oblivious::BTreeMap;

#[cfg(test)]
mod tests {
    use crate::BTreeMap;
    use std::thread;
    use std::time;

    #[test]
    fn find_missing() {
        let tree = BTreeMap::<u8, String>::new(3);
        assert_eq!(tree.get(&4), None);
    }

    #[test]
    fn add_existing() {
        let mut tree = BTreeMap::<u8, String>::new(3);
        tree.insert(5, String::from("Test"));
        // assert_eq!(tree.get(&5), Some(&String::from("Test")));
        tree.insert(5, String::from("Double"));
        // assert_eq!(tree.get(&5), Some(&String::from("Double")));
    }

    #[test]
    fn add_ordered_values() {
        let mut tree = BTreeMap::<u8, String>::new(3);
        tree.insert(3, String::from("Hello"));
        tree.insert(8, String::from("World"));
        tree.insert(12, String::from("!"));

        assert_eq!(tree.get(&3), Some(&String::from("Hello")));
        assert_eq!(tree.get(&8), Some(&String::from("World")));
        assert_eq!(tree.get(&12), Some(&String::from("!")));
    }

    #[test]
    fn add_unordered_values() {
        let mut tree = BTreeMap::<u8, String>::new(16);
        tree.insert(5, String::from("Hello"));
        tree.insert(3, String::from("World"));
        tree.insert(2, String::from("!"));

        assert_eq!(tree.get(&5), Some(&String::from("Hello")));
        assert_eq!(tree.get(&4), None);
        assert_eq!(tree.get(&3), Some(&String::from("World")));
        assert_eq!(tree.get(&2), Some(&String::from("!")));
    }

    #[test]
    fn add_100_values() {
        let mut tree = BTreeMap::<u8, u8>::new(100);
        for i in 1..100u8 {
            tree.insert(i, i + 1);
        }

        assert_eq!(tree.get(&99), Some(&100));
    }
}
