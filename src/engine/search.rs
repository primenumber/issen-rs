#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::remote::*;
use crate::engine::table::*;
use crate::engine::think::*;
use bitintr::Tzcnt;
use futures::channel::mpsc;
use futures::executor;
use futures::executor::ThreadPool;
use futures::future::{BoxFuture, FutureExt};
use futures::task::SpawnExt;
use futures::StreamExt;
use std::cmp::{max, min};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::{swap, MaybeUninit};
use std::sync::Arc;
use surf::Client;

#[derive(Clone)]
pub struct SearchParams {
    pub reduce: bool,
    pub ybwc_depth_limit: i8,
    pub ybwc_elder_add: i8,
    pub ybwc_younger_add: i8,
    pub ybwc_empties_limit: i8,
    pub eval_ordering_limit: i8,
    pub res_cache_limit: i8,
    pub stability_cut_limit: i8,
    pub ffs_ordering_limit: i8,
    pub static_ordering_limit: i8,
    pub use_worker: bool,
}

#[derive(Clone)]
pub struct SolveObj {
    pub res_cache: ResCacheTable,
    pub eval_cache: EvalCacheTable,
    pub evaluator: Arc<Evaluator>,
    params: SearchParams,
    pool: ThreadPool,
}

enum CutType {
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

impl SolveObj {
    pub fn new(
        res_cache: ResCacheTable,
        eval_cache: EvalCacheTable,
        evaluator: Arc<Evaluator>,
        params: SearchParams,
        pool: ThreadPool,
    ) -> SolveObj {
        SolveObj {
            res_cache,
            eval_cache,
            evaluator,
            params,
            pool,
        }
    }
}

fn near_leaf(board: Board) -> (i8, SolveStat) {
    let bit = board.empty();
    let pos = bit.tzcnt() as usize;
    match board.play(pos) {
        Ok(next) => (-next.score(), SolveStat::one()),
        Err(_) => (
            match board.pass().play(pos) {
                Ok(next) => next.score(),
                Err(_) => board.score(),
            },
            SolveStat {
                node_count: 2,
                st_cut_count: 0,
            },
        ),
    }
}

fn naive(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    beta: i8,
    passed: bool,
) -> (i8, SolveStat) {
    let mut pass = true;
    let mut res = -(BOARD_SIZE as i8);
    let mut stat = SolveStat::one();
    for (next, _pos) in board.next_iter() {
        pass = false;
        let (child_res, child_stat) = solve_inner(solve_obj, next, -beta, -alpha, false);
        res = max(res, -child_res);
        stat.merge(child_stat);
        alpha = max(alpha, res);
        if alpha >= beta {
            return (res, stat);
        }
    }
    if pass {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass(), -beta, -alpha, true);
            stat.merge(child_stat);
            return (-child_res, stat);
        }
    }
    (res, stat)
}

fn static_order(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    beta: i8,
    passed: bool,
) -> (i8, SolveStat) {
    let mut pass = true;
    let mut res = -(BOARD_SIZE as i8);
    let mut stat = SolveStat::one();
    const MASKS: [u64; 3] = [
        0x8100_0000_0000_0081, // Corner
        0x3C3C_FFFF_FFFF_3C3C, // Normal
        0x42C3_0000_0000_C342, // C + X
    ];
    for mask in MASKS.iter() {
        let mut empties = board.empty() & mask;
        while empties != 0 {
            let pos = empties.tzcnt() as usize;
            empties = empties & (empties - 1);
            if let Ok(next) = board.play(pos) {
                pass = false;
                let (child_res, child_stat) = solve_inner(solve_obj, next, -beta, -alpha, false);
                res = max(res, -child_res);
                stat.merge(child_stat);
                alpha = max(alpha, res);
                if alpha >= beta {
                    return (res, stat);
                }
            }
        }
    }
    if pass {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass(), -beta, -alpha, true);
            stat.merge(child_stat);
            return (-child_res, stat);
        }
    }
    (res, stat)
}

fn negascout_impl(
    solve_obj: &mut SolveObj,
    next: Board,
    alpha: i8,
    beta: i8,
    is_first: bool,
) -> (i8, SolveStat) {
    if is_first {
        solve_inner(solve_obj, next, -beta, -alpha, false)
    } else {
        let (res, mut stat) = solve_inner(solve_obj, next, -alpha - 1, -alpha, false);
        let mut neg_result = -res;
        if neg_result >= beta {
            return (res, stat);
        }
        if neg_result > alpha {
            let (res2, stat2) = solve_inner(solve_obj, next, -beta, -neg_result, false);
            stat.merge(stat2);
            neg_result = -res2;
        }
        (-neg_result, stat)
    }
}

fn fastest_first(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    beta: i8,
    passed: bool,
) -> (i8, SolveStat) {
    const MAX_FFS_NEXT: usize = 20;
    let nexts = MaybeUninit::<[(i8, Board); MAX_FFS_NEXT]>::uninit();
    let mut nexts = unsafe { nexts.assume_init() };
    let mut count = 0;
    for (next, _pos) in board.next_iter() {
        nexts[count] = (weighted_mobility(&next), next);
        count += 1;
    }
    assert!(count <= MAX_FFS_NEXT);

    nexts[0..count].sort_by(|a, b| a.0.cmp(&b.0));
    let mut res = -(BOARD_SIZE as i8);
    let mut stat = SolveStat::one();
    for (i, &(_, next)) in nexts[0..count].iter().enumerate() {
        let (child_res, child_stat) = negascout_impl(solve_obj, next, alpha, beta, i == 0);
        res = max(res, -child_res);
        stat.merge(child_stat);
        alpha = max(alpha, res);
        if alpha >= beta {
            return (res, stat);
        }
    }
    if count == 0 {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_result, child_stat) =
                solve_inner(solve_obj, board.pass(), -beta, -alpha, true);
            stat.merge(child_stat);
            return (-child_result, stat);
        }
    }
    (res, stat)
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

fn move_ordering_impl(
    solve_obj: &mut SolveObj,
    board: Board,
    _old_best: Option<Hand>,
) -> Vec<(Hand, Board)> {
    let mut nexts = Vec::with_capacity(32);
    for (next, pos) in board.next_iter() {
        nexts.push((0, pos, next));
    }

    let rem = popcnt(board.empty());
    let max_depth = calc_max_depth(rem);
    let min_depth = (max_depth - 3).max(0);
    let mut tmp = Vec::with_capacity(32);
    for think_depth in min_depth..=max_depth {
        tmp.clear();
        for &(_score, pos, next) in nexts.iter() {
            let mobility_score = popcnt(next.mobility_bits()) as i16;
            let bonus = if rem < 18 {
                mobility_score * SCALE * 1
            } else if rem < 22 {
                mobility_score * SCALE / 2
            } else {
                mobility_score * SCALE / 4
            };
            let mut searcher = Searcher {
                evaluator: solve_obj.evaluator.clone(),
                cache: solve_obj.eval_cache.clone(),
                timer: None,
                node_count: 0,
            };
            let score = searcher
                .think(
                    next,
                    EVAL_SCORE_MIN,
                    EVAL_SCORE_MAX,
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
            .filter(|e| e.0 < score_min + 16 * SCALE)
            .map(|e| (e.1, e.2))
            .collect()
    } else {
        nexts.into_iter().map(|e| (e.1, e.2)).collect()
    }
}

fn move_ordering_by_eval(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    beta: i8,
    passed: bool,
    old_best: Option<Hand>,
) -> (i8, Option<Hand>, SolveStat) {
    let v = move_ordering_impl(solve_obj, board, old_best);
    let mut res = -(BOARD_SIZE as i8);
    let mut best = None;
    let mut stat = SolveStat::one();
    for (i, &(pos, next)) in v.iter().enumerate() {
        let (child_res, child_stat) = negascout_impl(solve_obj, next, alpha, beta, i == 0);
        if -child_res > res {
            res = -child_res;
            best = Some(pos);
        }
        stat.merge(child_stat);
        alpha = max(alpha, res);
        if res >= beta {
            return (res, best, stat);
        }
    }
    if v.is_empty() {
        if passed {
            return (board.score(), Some(Hand::Pass), stat);
        } else {
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass(), -beta, -alpha, true);
            stat.merge(child_stat);
            return (-child_res, Some(Hand::Pass), stat);
        }
    }
    (res, best, stat)
}

async fn ybwc(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    beta: i8,
    passed: bool,
    old_best: Option<Hand>,
    depth: i8,
) -> (i8, Option<Hand>, SolveStat) {
    let v = move_ordering_impl(solve_obj, board, old_best);
    let mut stat = SolveStat::one();
    if v.is_empty() {
        if passed {
            return (board.score(), Some(Hand::Pass), stat);
        } else {
            let (child_res, _child_best, child_stat) =
                solve_outer(solve_obj, board.pass(), -beta, -alpha, true, depth).await;
            stat.merge(child_stat);
            return (-child_res, Some(Hand::Pass), stat);
        }
    }
    let mut res = -(BOARD_SIZE as i8);
    let mut best = None;
    let (tx, mut rx) = mpsc::unbounded();
    let mut handles = Vec::new();
    for (i, &(pos, next)) in v.iter().enumerate() {
        if i == 0 {
            let next_depth = depth + solve_obj.params.ybwc_elder_add;
            let (child_res, _child_best, child_stat) =
                solve_outer(solve_obj, next, -beta, -alpha, false, next_depth).await;
            stat.merge(child_stat);
            res = -child_res;
            best = Some(pos);
            alpha = max(alpha, res);
            if alpha >= beta {
                return (res, best, stat);
            }
        } else if depth < solve_obj.params.ybwc_depth_limit {
            let tx = tx.clone();
            let mut child_obj = solve_obj.clone();
            let mut stat = SolveStat::zero();
            handles.push(
                solve_obj
                    .pool
                    .spawn_with_handle(async move {
                        let next_depth = depth + child_obj.params.ybwc_younger_add;
                        let child_future = solve_outer(
                            &mut child_obj,
                            next,
                            -alpha - 1,
                            -alpha,
                            false,
                            next_depth,
                        );
                        let (child_res, _child_best, child_stat) = child_future.await;
                        stat.merge(child_stat);
                        let mut tmp = -child_res;
                        if alpha < tmp && tmp < beta {
                            let child_future =
                                solve_outer(&mut child_obj, next, -beta, -tmp, false, next_depth);
                            let (child_res, _child_best, child_stat) = child_future.await;
                            stat.merge(child_stat);
                            tmp = -child_res;
                        }
                        if tmp > res {
                            best = Some(pos);
                            res = tmp;
                        }
                        let res_tuple = (res, best);
                        let _ = tx.unbounded_send((res_tuple, stat));
                    })
                    .unwrap(),
            );
        } else {
            let (child_res, _child_best, child_stat) =
                solve_outer(solve_obj, next, -alpha - 1, -alpha, false, depth).await;
            stat.merge(child_stat);
            let mut tmp = -child_res;
            if alpha < tmp && tmp < beta {
                let (child_res, _child_best, child_stat) =
                    solve_outer(solve_obj, next, -beta, -tmp, false, depth).await;
                stat.merge(child_stat);
                tmp = -child_res;
            }
            if tmp >= beta {
                return (tmp, Some(pos), stat);
            }
            if tmp > res {
                best = Some(pos);
                res = tmp;
            }
        }
    }
    drop(tx);
    while let Some((res_tuple, child_stat)) = rx.next().await {
        stat.merge(child_stat);
        let (child_res, child_best) = res_tuple;
        if child_res > res {
            res = child_res;
            best = child_best;
            if res >= beta {
                rx.close();
                return (res, best, stat);
            }
        }
    }
    (res, best, stat)
}

#[derive(PartialEq, Debug)]
enum CacheLookupResult {
    Cut(i8),
    NoCut(i8, i8, Option<Hand>),
}

fn make_lookup_result(
    res_cache: Option<ResCache>,
    alpha: &mut i8,
    beta: &mut i8,
) -> CacheLookupResult {
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

fn lookup_table(
    solve_obj: &mut SolveObj,
    board: Board,
    alpha: &mut i8,
    beta: &mut i8,
) -> CacheLookupResult {
    let res_cache = solve_obj.res_cache.get(board);
    make_lookup_result(res_cache, alpha, beta)
}

fn stability_cut(board: Board, alpha: i8, beta: i8) -> CutType {
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

pub fn solve_inner(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    mut beta: i8,
    passed: bool,
) -> (i8, SolveStat) {
    let rem = popcnt(board.empty());
    if rem == 0 {
        (board.score(), SolveStat::zero())
    } else if rem == 1 {
        near_leaf(board)
    } else if rem < solve_obj.params.static_ordering_limit {
        naive(solve_obj, board, alpha, beta, passed)
    } else if rem < solve_obj.params.ffs_ordering_limit {
        static_order(solve_obj, board, alpha, beta, passed)
    } else {
        if rem >= solve_obj.params.stability_cut_limit {
            match stability_cut(board, alpha, beta) {
                CutType::NoCut => (),
                CutType::MoreThanBeta(v) => {
                    return (
                        v,
                        SolveStat {
                            node_count: 1,
                            st_cut_count: 1,
                        },
                    )
                }
                CutType::LessThanAlpha(v) => {
                    return (
                        v,
                        SolveStat {
                            node_count: 1,
                            st_cut_count: 1,
                        },
                    )
                }
            }
        }
        if rem < solve_obj.params.res_cache_limit {
            fastest_first(solve_obj, board, alpha, beta, passed)
        } else if rem < solve_obj.params.eval_ordering_limit {
            let (lower, upper) = match lookup_table(solve_obj, board, &mut alpha, &mut beta) {
                CacheLookupResult::Cut(v) => return (v, SolveStat::zero()),
                CacheLookupResult::NoCut(l, u, _) => (l, u),
            };
            let (res, stat) = fastest_first(solve_obj, board, alpha, beta, passed);
            update_table(
                &mut solve_obj.res_cache,
                board,
                res,
                None,
                alpha,
                beta,
                (lower, upper),
            );
            (res, stat)
        } else {
            let (lower, upper, old_best) =
                match lookup_table(solve_obj, board, &mut alpha, &mut beta) {
                    CacheLookupResult::Cut(v) => return (v, SolveStat::zero()),
                    CacheLookupResult::NoCut(l, u, b) => (l, u, b),
                };
            let (res, best, stat) =
                move_ordering_by_eval(solve_obj, board, alpha, beta, passed, old_best);
            if rem >= solve_obj.params.res_cache_limit {
                update_table(
                    &mut solve_obj.res_cache,
                    board,
                    res,
                    best,
                    alpha,
                    beta,
                    (lower, upper),
                );
            }
            (res, stat)
        }
    }
}

pub fn solve_outer(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    mut beta: i8,
    passed: bool,
    depth: i8,
) -> BoxFuture<'static, (i8, Option<Hand>, SolveStat)> {
    let mut solve_obj = solve_obj.clone();
    async move {
        let rem = popcnt(board.empty());
        if rem < solve_obj.params.ybwc_empties_limit {
            if solve_obj.params.use_worker {
                let data = SolveRequest {
                    board: board.to_base81(),
                    alpha,
                    beta,
                };
                let mut hasher = DefaultHasher::new();
                board.player.hash(&mut hasher);
                board.opponent.hash(&mut hasher);
                let suffix = 192 + hasher.finish() % 4;
                let data_str = serde_json::json!(data);
                let res: SolveResponse = Client::new()
                    .post(format!("http://192.168.10.{}:7733/", suffix))
                    .body(http_types::Body::from_json(&data_str).unwrap())
                    .recv_json()
                    .await
                    .unwrap();
                let stat = SolveStat {
                    node_count: res.node_count,
                    st_cut_count: res.st_cut_count,
                };
                (res.result, None, stat)
            } else {
                let (res, stat) = solve_inner(&mut solve_obj, board, alpha, beta, passed);
                (res, None, stat)
            }
        } else {
            match stability_cut(board, alpha, beta) {
                CutType::NoCut => (),
                CutType::MoreThanBeta(v) => {
                    return (
                        v,
                        None,
                        SolveStat {
                            node_count: 1,
                            st_cut_count: 1,
                        },
                    )
                }
                CutType::LessThanAlpha(v) => {
                    return (
                        v,
                        None,
                        SolveStat {
                            node_count: 1,
                            st_cut_count: 1,
                        },
                    )
                }
            }
            let (lower, upper, old_best) =
                match lookup_table(&mut solve_obj, board, &mut alpha, &mut beta) {
                    CacheLookupResult::Cut(v) => return (v, None, SolveStat::zero()),
                    CacheLookupResult::NoCut(l, u, b) => (l, u, b),
                };
            let (res, best, stat) =
                ybwc(&mut solve_obj, board, alpha, beta, passed, old_best, depth).await;
            if rem >= solve_obj.params.res_cache_limit {
                update_table(
                    &mut solve_obj.res_cache,
                    board,
                    res,
                    best,
                    alpha,
                    beta,
                    (lower, upper),
                );
            }
            (res, best, stat)
        }
    }
    .boxed()
}

pub fn solve(
    solve_obj: &mut SolveObj,
    board: Board,
    alpha: i8,
    beta: i8,
    passed: bool,
    depth: i8,
) -> (i8, Option<Hand>, SolveStat) {
    executor::block_on(solve_outer(solve_obj, board, alpha, beta, passed, depth))
}

pub async fn solve_with_move(board: Board, solve_obj: &mut SolveObj) -> Hand {
    match solve_outer(
        solve_obj,
        board,
        -(BOARD_SIZE as i8),
        BOARD_SIZE as i8,
        false,
        0,
    )
    .await
    .1
    {
        Some(best) => best,
        None => {
            let mut best_pos = None;
            let mut result = -65;
            for pos in board.mobility() {
                let next = board.play(pos).unwrap();
                let res = -solve_outer(solve_obj, next, -(BOARD_SIZE as i8), -result, false, 0)
                    .await
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
    }
}
