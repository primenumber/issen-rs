use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::hand::*;
use crc64::Crc64;
use spin::Mutex;
use std::io::Write;
use std::sync::Arc;

pub trait CacheElement: Clone + Default {
    fn has_key(&self, board: Board) -> bool;
    fn get_key(&self) -> Board;
    fn update(&mut self, that: &Self);
}

#[derive(Clone)]
pub struct EvalCache {
    pub board: Board,
    pub lower: i16,
    pub upper: i16,
    pub gen: u32,
    pub best: Option<Hand>,
    pub depth: i32,
}

impl Default for EvalCache {
    fn default() -> Self {
        Self {
            board: Board {
                player: 0,
                opponent: 0,
            },
            lower: 0,
            upper: 0,
            gen: 0,
            best: None,
            depth: 0,
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

    fn update(&mut self, that: &Self) {
        if that.board == self.board {
            if that.depth >= self.depth {
                if that.depth == self.depth {
                    let lower = self.lower.max(that.lower);
                    let upper = self.upper.min(that.upper);
                    *self = that.clone();
                    self.lower = lower;
                    self.upper = upper;
                } else {
                    *self = that.clone();
                }
                self.gen = that.gen;
            }
        } else {
            let empty_self = popcnt(self.board.empty());
            let empty_that = popcnt(that.board.empty());
            if empty_that >= empty_self || that.gen > self.gen {
                *self = that.clone();
                self.gen = that.gen;
            }
        }
    }
}

#[derive(Clone)]
pub struct ResCache {
    pub board: Board,
    pub lower: i8,
    pub upper: i8,
    pub gen: u32,
    pub best: Option<Hand>,
}

impl Default for ResCache {
    fn default() -> Self {
        Self {
            board: Board {
                player: 0,
                opponent: 0,
            },
            lower: 0,
            upper: 0,
            gen: 0,
            best: None,
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

    fn update(&mut self, that: &Self) {
        if that.board == self.board {
            let lower = self.lower.max(that.lower);
            let upper = self.upper.min(that.upper);
            *self = that.clone();
            self.lower = lower;
            self.upper = upper;
            self.gen = that.gen;
        } else {
            let empty_self = popcnt(self.board.empty());
            let empty_that = popcnt(that.board.empty());
            if empty_that >= empty_self || that.gen > self.gen {
                *self = that.clone();
                self.gen = that.gen;
            }
        }
    }
}

#[derive(Clone)]
pub struct CacheArray<T: CacheElement> {
    ary: Vec<T>,
    cycle: u64,
}

impl<T: CacheElement> CacheArray<T> {
    pub fn new(size: usize) -> CacheArray<T> {
        let dummy: T = Default::default();
        CacheArray {
            ary: vec![dummy; size],
            cycle: size as u64,
        }
    }

    pub fn get(&self, board: Board, hash: u64) -> Option<T> {
        let index = (hash % self.cycle) as usize;
        let elem = &self.ary[index];
        if elem.has_key(board) {
            Some(elem.clone())
        } else {
            None
        }
    }

    pub fn update(&mut self, new_elem: &T, hash: u64) {
        let index = (hash % self.cycle) as usize;
        let elem = &mut self.ary[index];
        elem.update(new_elem);
    }
}

pub struct CacheTable<T: CacheElement> {
    arrays: Vec<Mutex<CacheArray<T>>>,
    buckets: u64,
}

impl<T: CacheElement> CacheTable<T> {
    pub fn new(buckets: usize, capacity_per_bucket: usize) -> CacheTable<T> {
        let mut vec = Vec::new();
        for _ in 0..buckets {
            vec.push(Mutex::new(CacheArray::<T>::new(capacity_per_bucket)));
        }
        CacheTable::<T> {
            arrays: vec,
            buckets: buckets as u64,
        }
    }

    pub fn get(&self, board: Board) -> Option<T> {
        let mut crc64 = Crc64::new();
        crc64.write(&board.player.to_le_bytes()).unwrap();
        crc64.write(&board.opponent.to_le_bytes()).unwrap();
        let hash = crc64.get();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        self.arrays[bucket_id].lock().get(board, bucket_hash)
    }

    pub fn update(&self, cache: T) {
        let mut crc64 = Crc64::new();
        let board = cache.get_key();
        crc64.write(&board.player.to_le_bytes()).unwrap();
        crc64.write(&board.opponent.to_le_bytes()).unwrap();
        let hash = crc64.get();
        let bucket_id = (hash % self.buckets) as usize;
        let bucket_hash = hash / self.buckets;
        self.arrays[bucket_id].lock().update(&cache, bucket_hash);
    }
}

pub type EvalCacheTable = CacheTable<EvalCache>;
pub type ResCacheTable = CacheTable<ResCache>;

pub fn make_record(
    gen: u32,
    board: Board,
    mut res: i8,
    best: Option<Hand>,
    (alpha, beta): (i8, i8),
    range: (i8, i8),
) -> ResCache {
    res = res.clamp(range.0, range.1);
    let updated_range = if res <= alpha {
        (range.0, res)
    } else if res >= beta {
        (res, range.1)
    } else {
        (res, res)
    };
    ResCache {
        board,
        lower: updated_range.0,
        upper: updated_range.1,
        gen,
        best,
    }
}

pub fn update_table(
    res_cache: Arc<ResCacheTable>,
    cache_gen: u32,
    board: Board,
    res: i8,
    best: Option<Hand>,
    (alpha, beta): (i8, i8),
    range: (i8, i8),
) {
    let record = make_record(cache_gen, board, res, best, (alpha, beta), range);
    res_cache.update(record);
}
