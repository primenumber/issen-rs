use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::str::FromStr;
use crate::bits::*;
use crate::board::*;

pub trait CacheElement: Clone + Default {
    fn has_key(&self, board: Board) -> bool;
    fn get_key(&self) -> Board;
    fn update(&mut self, that: &Self, gen: u16);
}

#[derive(Clone)]
pub struct EvalCache {
    pub board: Board,
    pub lower: i16,
    pub upper: i16,
    pub gen: u16,
    pub best: u8,
    pub depth: i8
}

impl Default for EvalCache {
    fn default() -> Self {
        Self {
            board: Board::from_str("---------------------------------------------------------------- X;").unwrap(),
            lower: 0,
            upper: 0,
            gen: 0,
            best: PASS as u8,
            depth: 0
        }
    }
}

impl CacheElement for EvalCache {
    fn has_key(&self, board: Board) -> bool {
        self.board == board
    }

    fn get_key(&self) -> Board {
        self.board
    }

    fn update(&mut self, that: &Self, gen: u16) {
        if that.board == self.board {
            if that.depth >= self.depth {
                if that.depth == self.depth {
                    self.lower = self.lower.max(that.lower);
                    self.upper = self.upper.min(that.upper);
                }
                *self = that.clone();
                self.gen = gen;
            }
        } else {
            let empty_self = popcnt(self.board.empty());
            let empty_that = popcnt(that.board.empty());
            if empty_that > empty_self || gen > self.gen {
                *self = that.clone();
                self.gen = gen;
            }
        }
    }
}

#[derive(Clone)]
pub struct ResCache {
    pub board: Board,
    pub lower: i8,
    pub upper: i8,
    pub gen: u16,
    pub best: u8
}

impl Default for ResCache {
    fn default() -> Self {
        Self {
            board: Board::from_str("---------------------------------------------------------------- X;").unwrap(),
            lower: 0,
            upper: 0,
            gen: 0,
            best: PASS as u8
        }
    }
}

impl CacheElement for ResCache {
    fn has_key(&self, board: Board) -> bool {
        self.board == board
    }

    fn get_key(&self) -> Board {
        self.board
    }

    fn update(&mut self, that: &Self, gen: u16) {
        if that.board == self.board {
            *self = that.clone();
            self.gen = gen;
        } else {
            let empty_self = popcnt(self.board.empty());
            let empty_that = popcnt(that.board.empty());
            if empty_that > empty_self || gen > self.gen {
                *self = that.clone();
                self.gen = gen;
            }
        }
    }
}

struct CacheArray<T: CacheElement> {
    ary: Vec<T>,
    cycle: u64
}


impl<T: CacheElement> CacheArray<T> {
    fn new(size: usize) -> CacheArray<T> {
        let dummy: T = Default::default();
        CacheArray {
            ary: vec![dummy; size],
            cycle: size as u64
        }
    }

    fn get(&self, board: Board, hash: u64) -> Option<T> {
        let index = (hash % self.cycle) as usize;
        let elem = &self.ary[index];
        if elem.has_key(board) {
            Some(elem.clone())
        } else {
            None
        }
    }

    fn update(&mut self, new_elem: &T, hash: u64, gen: u16) {
        let index = (hash % self.cycle) as usize;
        let mut elem = &mut self.ary[index];
        elem.update(new_elem, gen);
    }
}

#[derive(Clone)]
pub struct CacheTable<T: CacheElement> {
    arrays: Arc<Vec<Mutex<CacheArray<T>>>>,
    buckets: u64,
    pub gen: u16
}

impl<T: CacheElement> CacheTable<T> {
    pub fn new(buckets: usize, capacity_per_bucket: usize) -> CacheTable<T> {
        let mut vec = Vec::new();
        for _ in 0..buckets {
            vec.push(Mutex::new(CacheArray::<T>::new(capacity_per_bucket)));
        }
        CacheTable::<T> {
            arrays: Arc::new(vec),
            buckets: buckets as u64,
            gen: 1
        }
    }

    pub fn get(&mut self, board: Board) -> Option<T> {
        let mut hasher = DefaultHasher::new();
        board.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        self.arrays[bucket_id].lock().unwrap().get(board, bucket_hash)
    }

    pub fn update(&mut self, cache: T) {
        let mut hasher = DefaultHasher::new();
        cache.get_key().hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        self.arrays[bucket_id].lock().unwrap().update(&cache, bucket_hash, self.gen);
    }

    pub fn inc_gen(&mut self) {
        self.gen += 1;
    }
}

pub type EvalCacheTable = CacheTable<EvalCache>;
pub type ResCacheTable = CacheTable<ResCache>;
