use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::str::FromStr;
use crate::bits::*;
use crate::board::*;

#[derive(Clone)]
pub struct EvalCache {
    pub board: Board,
    pub lower: i16,
    pub upper: i16,
    pub gen: u16,
    pub best: u8,
    pub depth: i8
}

struct EvalCacheArray {
    ary: Vec<EvalCache>,
    cycle: u64
}

impl EvalCacheArray {
    fn new(size: usize) -> EvalCacheArray {
        let dummy = EvalCache {
            board: Board::from_str("---------------------------------------------------------------- X;").unwrap(),
            lower: 0,
            upper: 0,
            gen: 0,
            best: PASS as u8,
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
            if empty_cache > empty_old || cache.gen > old.gen {
                self.ary[index] = cache;
            }
            false
        }
    }
}

pub struct EvalCacheTable {
    arrays: Arc<Vec<Mutex<EvalCacheArray>>>,
    buckets: u64,
    pub gen: u16,
    pub cnt_get: usize,
    pub cnt_update: usize,
    pub cnt_hit: usize,
}

impl Clone for EvalCacheTable {
    fn clone(&self) -> Self {
        EvalCacheTable {
            arrays: self.arrays.clone(),
            buckets: self.buckets,
            gen: self.gen.clone(),
            cnt_get: 0,
            cnt_update: 0,
            cnt_hit: 0,
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
            gen: 1,
            cnt_get: 0,
            cnt_update: 0,
            cnt_hit: 0,
        }
    }

    pub fn get(&mut self, board: Board) -> Option<EvalCache> {
        self.cnt_get += 1;
        let mut hasher = DefaultHasher::new();
        board.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        let res = self.arrays[bucket_id].lock().unwrap().get(board, bucket_hash);
        match res {
            Some(_) => { self.cnt_hit += 1; }
            None => ()
        }
        res
    }

    pub fn update(&mut self, mut cache: EvalCache) {
        self.cnt_update += 1;
        let mut hasher = DefaultHasher::new();
        cache.board.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        cache.gen = self.gen;
        if self.arrays[bucket_id].lock().unwrap().update(cache, bucket_hash) {
            self.cnt_hit += 1;
        }
    }

    pub fn inc_gen(&mut self) {
        self.gen += 1;
    }

    pub fn add_cnt_get(&mut self, add: usize) {
        self.cnt_get += add;
    }

    pub fn add_cnt_update(&mut self, add: usize) {
        self.cnt_update += add;
    }

    pub fn add_cnt_hit(&mut self, add: usize) {
        self.cnt_hit += add;
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

struct ResCacheArray {
    ary: Vec<ResCache>,
    cycle: u64
}

impl ResCacheArray {
    fn new(size: usize) -> ResCacheArray {
        let dummy = ResCache {
            board: Board::from_str("---------------------------------------------------------------- X;").unwrap(),
            lower: 0,
            upper: 0,
            gen: 0,
            best: PASS as u8
        };
        ResCacheArray{
            ary: vec![dummy; size],
            cycle: size as u64
        }
    }

    fn get(&self, board: Board, hash: u64) -> Option<ResCache> {
        let index = (hash % self.cycle) as usize;
        let elem = &self.ary[index];
        if elem.board == board {
            Some(elem.clone())
        } else {
            None
        }
    }

    fn update(&mut self, mut cache: ResCache, hash: u64) -> bool {
        let index = (hash % self.cycle) as usize;
        let old = &self.ary[index];
        if cache.board == old.board {
            cache.lower = cache.lower.max(old.lower);
            cache.upper = cache.upper.min(old.upper);
            self.ary[index] = cache;
            true
        } else {
            let empty_cache = popcnt(cache.board.empty());
            let empty_old = popcnt(old.board.empty());
            if empty_cache > empty_old || cache.gen > old.gen {
                self.ary[index] = cache;
            }
            false
        }
    }
}

#[derive(Clone)]
pub struct ResCacheTable {
    arrays: Arc<Vec<Mutex<ResCacheArray>>>,
    buckets: u64,
    pub gen: u16
}

impl ResCacheTable {
    pub fn new(buckets: usize, capacity_per_bucket: usize) -> ResCacheTable {
        let mut vec = Vec::new();
        for _ in 0..buckets {
            vec.push(Mutex::new(ResCacheArray::new(capacity_per_bucket)));
        }
        ResCacheTable {
            arrays: Arc::new(vec),
            buckets: buckets as u64,
            gen: 1
        }
    }

    pub fn get(&mut self, board: Board) -> Option<ResCache> {
        let mut hasher = DefaultHasher::new();
        board.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        self.arrays[bucket_id].lock().unwrap().get(board, bucket_hash)
    }

    pub fn update(&mut self, mut cache: ResCache) {
        let mut hasher = DefaultHasher::new();
        cache.board.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        cache.gen = self.gen;
        self.arrays[bucket_id].lock().unwrap().update(cache, bucket_hash);
    }

    pub fn inc_gen(&mut self) {
        self.gen += 1;
    }
}
