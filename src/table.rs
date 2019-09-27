use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::str::FromStr;
use std::cell::Cell;
use crate::bits::*;
use crate::board::*;

#[derive(Clone)]
pub struct EvalCache {
    pub board: Board,
    pub lower: i16,
    pub upper: i16,
    pub depth: i8
}

struct EvalCacheArray {
    ary: Vec<EvalCache>,
    cycle: u64
}

impl EvalCacheArray {
    fn new(size: usize) -> EvalCacheArray {
        let dummy = EvalCache {
            board: Board::from_str("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX X;").unwrap(),
            lower: 0,
            upper: 0,
            depth: 0
        };
        EvalCacheArray{
            ary: vec![dummy; size],
            cycle: size as u64
        }
    }

    fn get(&self, board: Board, hash: u64) -> Option<EvalCache> {
        let index = (hash % self.cycle) as usize;
        let elem = &self.ary[index];
        if elem.board == board {
            Some(elem.clone())
        } else {
            None
        }
    }

    fn update(&mut self, mut cache: EvalCache, hash: u64) -> bool {
        let index = (hash % self.cycle) as usize;
        let old = &self.ary[index];
        if cache.board == old.board {
            if cache.depth >= old.depth {
                if cache.depth == old.depth {
                    cache.lower = cache.lower.max(old.lower);
                    cache.upper = cache.upper.min(old.upper);
                }
                self.ary[index] = cache;
            }
            true
        } else {
            let empty_cache = popcnt(cache.board.empty());
            let empty_old = popcnt(old.board.empty());
            if empty_cache > empty_old {
                self.ary[index] = cache;
            }
            false
        }
    }
}

pub struct EvalCacheTable {
    arrays: Arc<Vec<Mutex<EvalCacheArray>>>,
    buckets: u64,
    pub cnt_get: Cell<usize>,
    pub cnt_update: Cell<usize>,
    pub cnt_hit: Cell<usize>,
}

impl Clone for EvalCacheTable {
    fn clone(&self) -> Self {
        EvalCacheTable {
            arrays: self.arrays.clone(),
            buckets: self.buckets,
            cnt_get: Cell::new(0),
            cnt_update: Cell::new(0),
            cnt_hit: Cell::new(0)
        }
    }
}

impl EvalCacheTable {
    pub fn new(buckets: usize, capacity_per_bucket: usize) -> EvalCacheTable {
        let mut vec = Vec::new();
        for _ in 0..buckets {
            vec.push(Mutex::new(EvalCacheArray::new(capacity_per_bucket)));
        }
        EvalCacheTable {
            arrays: Arc::new(vec),
            buckets: buckets as u64,
            cnt_get: Cell::new(0),
            cnt_update: Cell::new(0),
            cnt_hit: Cell::new(0)
        }
    }

    pub fn get(&self, board: Board) -> Option<EvalCache> {
        self.cnt_get.set(self.cnt_get.get() + 1);
        let mut hasher = DefaultHasher::new();
        board.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        let res = self.arrays[bucket_id].lock().unwrap().get(board, bucket_hash);
        match res {
            Some(_) => self.cnt_hit.set(self.cnt_hit.get() + 1),
            None => ()
        }
        res
    }

    pub fn update(&self, cache: EvalCache) {
        self.cnt_update.set(self.cnt_update.get() + 1);
        let mut hasher = DefaultHasher::new();
        cache.board.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        if self.arrays[bucket_id].lock().unwrap().update(cache, bucket_hash) {
            self.cnt_hit.set(self.cnt_hit.get() + 1);
        }
    }

    pub fn add_cnt_get(&self, add: usize) {
        self.cnt_get.set(self.cnt_get.get() + add);
    }

    pub fn add_cnt_update(&self, add: usize) {
        self.cnt_update.set(self.cnt_update.get() + add);
    }

    pub fn add_cnt_hit(&self, add: usize) {
        self.cnt_hit.set(self.cnt_hit.get() + add);
    }
}
