#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::last_cache::*;
use crate::engine::midgame::*;
use crate::engine::table::*;
use crate::engine::think::*;
use anyhow::Result;
use arrayvec::ArrayVec;
use crc64::Crc64;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::io::Write;
use std::mem::swap;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Debug, Serialize, Deserialize)]
pub struct SolveRequest {
    pub board: String,
    pub alpha: i8,
    pub beta: i8,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolveResponse {
    pub result: i8,
    pub node_count: usize,
    pub st_cut_count: usize,
}

#[derive(Clone)]
pub struct SearchParams {
    pub reduce: bool,
    pub parallel_depth_limit: i8,
    pub parallel_empties_limit: i8,
    pub eval_ordering_limit: i8,
    pub res_cache_limit: i8,
    pub local_res_cache_limit: i8,
    pub stability_cut_limit: i8,
    pub ffs_ordering_limit: i8,
    pub static_ordering_limit: i8,
}

pub struct SolveObj<Eval: Evaluator> {
    pub res_cache: Arc<ResCacheTable>,
    pub eval_cache: Arc<EvalCacheTable>,
    pub local_res_cache: CacheArray<ResCache>,
    pub evaluator: Arc<Eval>,
    pub last_cache: Arc<LastCache>,
    pub params: SearchParams,
    pub cache_gen: u32,
    pub local_cache_gen: u32,
}

impl<Eval: Evaluator> Clone for SolveObj<Eval> {
    fn clone(&self) -> Self {
        SolveObj::<Eval> {
            res_cache: self.res_cache.clone(),
            eval_cache: self.eval_cache.clone(),
            local_res_cache: self.local_res_cache.clone(),
            evaluator: self.evaluator.clone(),
            last_cache: self.last_cache.clone(),
            params: self.params.clone(),
            cache_gen: self.cache_gen.clone(),
            local_cache_gen: self.local_cache_gen.clone(),
        }
    }
}

impl<Eval: Evaluator> SolveObj<Eval> {
    pub fn new(
        res_cache: Arc<ResCacheTable>,
        eval_cache: Arc<EvalCacheTable>,
        evaluator: Arc<Eval>,
        params: SearchParams,
        cache_gen: u32,
    ) -> SolveObj<Eval> {
        SolveObj {
            res_cache,
            eval_cache,
            local_res_cache: CacheArray::<ResCache>::new(65536),
            evaluator,
            last_cache: Arc::new(LastCache::new()),
            params,
            cache_gen,
            local_cache_gen: 0,
        }
    }
}

pub enum CutType {
    NoCut,
    MoreThanBeta(i8),
    LessThanAlpha(i8),
}

#[derive(Clone, Copy)]
pub struct SolveStat {
    pub node_count: usize,
    pub st_cut_count: usize,
}

impl SolveStat {
    pub fn one() -> SolveStat {
        SolveStat {
            node_count: 1,
            st_cut_count: 0,
        }
    }
    pub fn one_stcut() -> SolveStat {
        SolveStat {
            node_count: 1,
            st_cut_count: 1,
        }
    }
    pub fn zero() -> SolveStat {
        SolveStat {
            node_count: 0,
            st_cut_count: 0,
        }
    }
    pub fn merge(&mut self, that: SolveStat) {
        self.node_count += that.node_count;
        self.st_cut_count += that.st_cut_count;
    }
}

#[derive(Clone)]
pub struct SubSolver {
    pub client: Arc<Client>,
    pub sem: Arc<Semaphore>,
    pub workers: Arc<Vec<String>>,
}

impl SubSolver {
    pub fn new(worker_urls: &[String]) -> SubSolver {
        SubSolver {
            client: Arc::new(Client::new()),
            sem: Arc::new(Semaphore::new(128)),
            workers: Arc::new(worker_urls.to_vec()),
        }
    }

    #[allow(dead_code)]
    pub async fn solve_remote(&self, board: Board, (alpha, beta): (i8, i8)) -> Result<(i8, SolveStat)> {
        let data = SolveRequest {
            board: board.to_base81(),
            alpha,
            beta,
        };
        let mut hasher = Crc64::new();
        hasher.write(&board.player.to_le_bytes())?;
        hasher.write(&board.opponent.to_le_bytes())?;
        let data_json = serde_json::json!(data);
        let worker_id = hasher.get() as usize % self.workers.len();
        let url = &self.workers[worker_id];
        let permit = self.sem.acquire().await?;
        let resp = self.client.post(url).json(&data_json).send().await?;
        let res = resp.json::<SolveResponse>().await?;
        drop(permit);
        let stat = SolveStat {
            node_count: res.node_count,
            st_cut_count: res.st_cut_count,
        };
        Ok((res.result, stat))
    }
}

#[derive(PartialEq, Debug)]
pub enum CacheLookupResult {
    Cut(i8),
    NoCut(i8, i8, Option<Hand>),
}

pub fn make_lookup_result(res_cache: Option<ResCache>, (alpha, beta): (&mut i8, &mut i8)) -> CacheLookupResult {
    let (lower, upper, old_best) = match res_cache {
        Some(cache) => (cache.lower, cache.upper, cache.best),
        None => (-(BOARD_SIZE as i8), BOARD_SIZE as i8, None),
    };
    let old_alpha = *alpha;
    *alpha = max(lower, *alpha);
    *beta = min(upper, *beta);
    if *alpha >= *beta {
        if old_alpha >= upper {
            CacheLookupResult::Cut(upper)
        } else {
            CacheLookupResult::Cut(lower)
        }
    } else {
        CacheLookupResult::NoCut(lower, upper, old_best)
    }
}

pub fn lookup_table<Eval: Evaluator>(
    solve_obj: &mut SolveObj<Eval>,
    board: Board,
    (alpha, beta): (&mut i8, &mut i8),
) -> CacheLookupResult {
    let res_cache = solve_obj.res_cache.get(board);
    make_lookup_result(res_cache, (alpha, beta))
}

pub fn stability_cut(board: Board, (alpha, beta): (i8, i8)) -> CutType {
    let (bits_me, bits_op) = board.stable_partial();
    let lower = 2 * popcnt(bits_me) - BOARD_SIZE as i8;
    let upper = (BOARD_SIZE as i8) - 2 * popcnt(bits_op);
    if upper <= alpha {
        CutType::LessThanAlpha(upper)
    } else if lower >= beta {
        CutType::MoreThanBeta(lower)
    } else {
        CutType::NoCut
    }
}

fn calc_max_depth(rem: i8) -> i8 {
    let mut th_depth = 9;
    if rem <= 27 {
        th_depth += (30 - rem) / 3;
    }
    let mut max_depth = 0;
    if rem >= th_depth {
        max_depth = (rem - 15) / 3;
        if rem >= 27 {
            max_depth += 1;
        }
        max_depth = max_depth.max(0).min(6);
    }
    max_depth
}

pub fn move_ordering_impl<Eval: Evaluator>(
    solve_obj: &mut SolveObj<Eval>,
    board: Board,
    _old_best: Option<Hand>,
) -> Vec<(Hand, Board)> {
    const MAX_NEXT_COUNT: usize = 32;
    let mut nexts = ArrayVec::<_, MAX_NEXT_COUNT>::new();
    for (next, pos) in board.next_iter() {
        nexts.push((0, pos, next));
    }

    let rem = popcnt(board.empty());
    let max_depth = calc_max_depth(rem);
    let min_depth = (max_depth - 3).max(0);
    let mut tmp = ArrayVec::<_, MAX_NEXT_COUNT>::new();
    for think_depth in min_depth..=max_depth {
        tmp.clear();
        for &(_score, pos, next) in nexts.iter() {
            let mobility_score = popcnt(next.mobility_bits()) as i16;
            let bonus = if rem < 18 {
                mobility_score * solve_obj.evaluator.score_scale() * 1
            } else if rem < 22 {
                mobility_score * solve_obj.evaluator.score_scale() / 2
            } else {
                mobility_score * solve_obj.evaluator.score_scale() / 4
            };
            let mut searcher = Searcher {
                evaluator: solve_obj.evaluator.clone(),
                cache: solve_obj.eval_cache.clone(),
                timer: None,
                node_count: 0,
                cache_gen: solve_obj.cache_gen,
            };
            let score = searcher
                .think(
                    next,
                    solve_obj.evaluator.score_min(),
                    solve_obj.evaluator.score_max(),
                    false,
                    think_depth as i32 * DEPTH_SCALE,
                )
                .unwrap()
                .0;
            tmp.push((score + bonus, pos, next));
        }
        tmp.sort_by(|a, b| a.0.cmp(&b.0));
        swap(&mut nexts, &mut tmp);
    }
    if !nexts.is_empty() && solve_obj.params.reduce {
        let score_min = nexts[0].0;
        nexts
            .into_iter()
            .filter(|e| e.0 < score_min + 16 * solve_obj.evaluator.score_scale())
            .map(|e| (e.1, e.2))
            .collect()
    } else {
        nexts.into_iter().map(|e| (e.1, e.2)).collect()
    }
}

pub fn solve<Eval: Evaluator>(
    solve_obj: &mut SolveObj<Eval>,
    _worker_urls: &[String],
    board: Board,
    (alpha, beta): (i8, i8),
    passed: bool,
    depth: i8,
    num_threads: Option<usize>,
) -> (i8, Option<Hand>, SolveStat) {
    simplified_abdada(solve_obj, board, (alpha, beta), passed, depth, num_threads)
}

// num_threads: number of searching threads, use number of cpus when None
pub fn solve_with_move<Eval: Evaluator>(
    board: Board,
    solve_obj: &mut SolveObj<Eval>,
    _sub_solver: &Arc<SubSolver>,
    num_threads: Option<usize>,
) -> Hand {
    if let Some(best) = simplified_abdada(
        solve_obj,
        board,
        (-(BOARD_SIZE as i8), BOARD_SIZE as i8),
        false,
        0,
        num_threads,
    )
    .1
    {
        return best;
    }
    let mut best_pos = None;
    let mut result = -65;
    for pos in board.mobility() {
        let next = board.play(pos).unwrap();
        let res = -simplified_abdada(
            solve_obj,
            next,
            (-(BOARD_SIZE as i8), -result),
            false,
            0,
            num_threads,
        )
        .0;
        if res > result {
            result = res;
            best_pos = Some(pos);
        }
    }
    if let Some(pos) = best_pos {
        Hand::Play(pos)
    } else {
        Hand::Pass
    }
}
