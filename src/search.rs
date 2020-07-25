use std::sync::Arc;
use std::cmp::{min, max};
use std::mem::MaybeUninit;
use futures::executor;
use futures::executor::ThreadPool;
use futures::channel::mpsc;
use futures::StreamExt;
use futures::future::{BoxFuture, FutureExt};
use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::table::*;

#[derive(Clone)]
pub struct SearchParams {
    pub reduce: bool,
    pub ybwc_depth_limit: i8,
    pub ybwc_elder_add: i8,
    pub ybwc_younger_add: i8,
    pub ybwc_empties_limit: i8,
    pub res_cache_limit: i8,
    pub eval_ordering_limit: i8,
    pub ffs_ordering_limit: i8,
}

#[derive(Clone)]
pub struct SolveObj {
    res_cache: ResCacheTable,
    pub eval_cache: EvalCacheTable,
    evaluator: Arc<Evaluator>,
    params: SearchParams,
    pool: ThreadPool
}

enum CutType {
    NoCut,
    MoreThanBeta(i8),
    LessThanAlpha(i8)
}

#[derive(Clone, Copy)]
pub struct SolveStat {
    pub node_count: usize,
    pub st_cut_count: usize
}

impl SolveStat {
    pub fn one() -> SolveStat {
        SolveStat { node_count: 1, st_cut_count: 0 }
    }
    pub fn zero() -> SolveStat {
        SolveStat { node_count: 0, st_cut_count: 0 }
    }
    pub fn merge(&mut self, that: SolveStat) {
        self.node_count += that.node_count;
        self.st_cut_count += that.st_cut_count;
    }
}

impl SolveObj {
    pub fn new(res_cache: ResCacheTable, eval_cache: EvalCacheTable,
               evaluator: Arc<Evaluator>, params: SearchParams,
               pool: ThreadPool) -> SolveObj {
        SolveObj {
            res_cache,
            eval_cache,
            evaluator,
            params,
            pool
        }
    }
}

fn near_leaf(board: Board) -> (i8, SolveStat) {
    let bit = board.empty();
    let pos = popcnt(bit - 1) as usize;
    match board.play(pos) {
        Ok(next) => (
            -next.score(),
            SolveStat::one()
        ),
        Err(_) => (
            match board.pass().play(pos) {
                Ok(next) => next.score(),
                Err(_) => board.score()
            },
            SolveStat { node_count: 2, st_cut_count: 0 }
        )
    }
}

fn naive(
    solve_obj: &mut SolveObj, board: Board, mut alpha: i8, beta: i8, passed: bool,
    depth: i8)-> (i8, SolveStat) {
    let mut pass = true;
    let mut empties = board.empty();
    let mut res = -64;
    let mut stat = SolveStat::one();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                pass = false;
                let (child_res, child_stat) = solve_inner(
                    solve_obj, next, -beta, -alpha, false, depth+1);
                res = max(res, -child_res);
                stat.merge(child_stat);
                alpha = max(alpha, res);
                if alpha >= beta {
                    return (res, stat);
                }
            },
            Err(_) => ()
        }
    }
    if pass {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass(), -beta, -alpha, true, depth);
            stat.merge(child_stat);
            return (-child_res, stat);
        }
    }
    (res, stat)
}

fn fastest_first(
    solve_obj: &mut SolveObj, board: Board, mut alpha: i8, beta: i8, passed: bool,
    depth: i8) -> (i8, SolveStat) {
    const MAX_FFS_NEXT: usize = 20;
    let mut nexts: [(i8, Board); MAX_FFS_NEXT] = unsafe {
        MaybeUninit::uninit().assume_init()
    };
    let mut count = 0;
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                nexts[count] = (weighted_mobility(& next), next);
                count += 1;
                assert!(count <= MAX_FFS_NEXT);
            },
            Err(_) => ()
        }
    }
    nexts[0..count].sort_by(|a, b| a.0.cmp(&b.0));
    let mut res = -64;
    let mut stat = SolveStat::one();
    for (i, &(_, ref next)) in nexts[0..count].iter().enumerate() {
        if i == 0 {
            let (child_res, child_stat) = solve_inner(
                solve_obj, next.clone(), -beta, -alpha, false, depth+1);
            stat.merge(child_stat);
            res = max(res, -child_res);
        } else {
            let (child_res, child_stat) = solve_inner(
                solve_obj, next.clone(), -alpha-1, -alpha, false, depth+1);
            stat.merge(child_stat);
            let mut result = -child_res;
            if result >= beta {
                return (result, stat);
            }
            if result > alpha {
                alpha = result;
                let (child_res, child_stat) = solve_inner(
                    solve_obj, next.clone(), -beta, -alpha, false, depth+1);
                stat.merge(child_stat);
                result = -child_res;
            }
            res = max(res, result);
        }
        alpha = max(alpha, res);
        if alpha >= beta {
            return (res, stat);
        }
    }
    if count == 0 {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_result, child_stat) = solve_inner(solve_obj, board.pass(), -beta, -alpha, true, depth);
            stat.merge(child_stat);
            return (-child_result, stat);
        }
    }
    (res, stat)
}

fn move_ordering_impl(solve_obj: &mut SolveObj, board: Board, old_best: u8, _depth: i8)
    -> Vec<(u8, Board)> {
    let mut nexts = vec![(0i16, 0u8, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                nexts.push((0, pos as u8, next));
            },
            Err(_) => ()
        }
    }

    let rem = popcnt(board.empty());
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
    let min_depth = (max_depth - 3).max(0);
    for think_depth in min_depth..=max_depth {
        let mut tmp = vec![(0i16, 0u8, board.clone()); 0];
        let mut res = 64 * SCALE;
        for (i, (score, pos, next)) in nexts.iter().enumerate() {
            let bonus = if *pos == old_best {
                -16 * SCALE
            } else {
                0
            };
            if i == 0 {
                let window = if think_depth == min_depth {
                    16
                } else {
                    8
                };
                let alpha = (score - window * SCALE).max(-64 * SCALE);
                let beta = (score + window * SCALE).min(64 * SCALE);
                let new_res = think(next.clone(), alpha, beta, false, solve_obj.evaluator.clone(), &mut solve_obj.eval_cache, think_depth);
                if new_res <= alpha {
                    let new_alpha = -64 * SCALE;
                    let new_beta = new_res;
                    res = think(next.clone(), new_alpha, new_beta, false, solve_obj.evaluator.clone(), &mut solve_obj.eval_cache, think_depth);
                    tmp.push((res + bonus, *pos, next.clone()));
                } else if new_res >= beta {
                    let new_alpha = new_res;
                    let new_beta = 64 * SCALE;
                    res = think(next.clone(), new_alpha, new_beta, false, solve_obj.evaluator.clone(), &mut solve_obj.eval_cache, think_depth);
                    tmp.push((res + bonus, *pos, next.clone()));
                } else {
                    res = new_res;
                    tmp.push((res + bonus, *pos, next.clone()));
                }
            } else {
                let new_res = think(next.clone(), res, res+1, false, solve_obj.evaluator.clone(), &mut solve_obj.eval_cache, think_depth);
                if new_res < res {
                    let fixed_res = think(next.clone(), -64 * SCALE, new_res, false, solve_obj.evaluator.clone(), &mut solve_obj.eval_cache, think_depth);
                    tmp.push((fixed_res + bonus, *pos, next.clone()));
                    res = fixed_res;
                } else {
                    let score = new_res;
                    tmp.push((score + bonus, *pos, next.clone()));
                }
            }
        }
        tmp.sort_by(|a, b| {
            a.0.cmp(&b.0)
        });
        nexts = tmp;
    }
    if nexts.len() > 0 && solve_obj.params.reduce {
        let score_min = nexts[0].0;
        nexts.into_iter()
            .filter(|e| e.0 < score_min + 16 * SCALE)
            .map(|e| (e.1, e.2))
            .collect()
    } else {
        nexts.into_iter().map(|e| (e.1, e.2)).collect()
    }
}

fn move_ordering_by_eval(
        solve_obj: &mut SolveObj, board: Board, mut alpha: i8, beta: i8, passed: bool,
        old_best: u8, depth: i8) -> (i8, u8, SolveStat) {
    let v = move_ordering_impl(solve_obj, board.clone(), old_best, depth);
    let mut res = -64;
    let mut best = PASS as u8;
    let mut stat = SolveStat::one();
    for (i, (pos, next)) in v.iter().enumerate() {
        if i == 0 {
            let (child_res, child_stat) = solve_inner(solve_obj, next.clone(), -beta, -alpha, false, depth+1);
            stat.merge(child_stat);
            res = -child_res;
            best = *pos;
            alpha = max(alpha, res);
            if alpha >= beta {
                return (res, best, stat);
            }
        } else {
            let (child_res, child_stat) = solve_inner(
                    solve_obj, next.clone(), -alpha-1, -alpha, false, depth+1);
            stat.merge(child_stat);
            let tmp = -child_res;
            if tmp >= res {
                res = tmp;
                best = *pos;
            }
            if res >= beta {
                return (res, best, stat);
            }
            if res > alpha {
                alpha = res;
                let (child_res, child_stat) = solve_inner(
                        solve_obj, next.clone(), -beta, -alpha, false, depth+1);
                stat.merge(child_stat);
                res = max(res, -child_res);
            }
        }
    }
    if v.is_empty() {
        if passed {
            return (board.score(), PASS as u8, stat);
        } else {
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass(), -beta, -alpha, true, depth);
            stat.merge(child_stat);
            return (-child_res, PASS as u8, stat);
        }
    }
    (res, best, stat)
}

async fn ybwc(
        solve_obj: &mut SolveObj, board: Board, mut alpha: i8, beta: i8, passed: bool,
        old_best: u8, depth: i8) -> (i8, u8, SolveStat) {
    let v = move_ordering_impl(solve_obj, board.clone(), old_best, depth);
    let mut stat = SolveStat::one();
    if v.is_empty() {
        if passed {
            return (board.score(), PASS as u8, stat);
        } else {
            let (child_res, child_stat) = solve_outer(solve_obj, board.pass(), -beta, -alpha, true, depth).await;
            stat.merge(child_stat);
            return (-child_res, PASS as u8, stat);
        }
    }
    let mut res = -64;
    let mut best = PASS as u8;
    let (tx, mut rx) = mpsc::unbounded();
    for (i, &(pos, next)) in v.iter().enumerate() {
        if i == 0 {
            let next_depth = depth + solve_obj.params.ybwc_elder_add;
            let (child_res, child_stat) = solve_outer(solve_obj, next.clone(), -beta, -alpha, false, next_depth).await;
            stat.merge(child_stat);
            res = -child_res;
            best = pos;
            alpha = max(alpha, res);
            if alpha >= beta {
                return (res, best, stat);
            }
        } else {
            let tx = tx.clone();
            let mut child_obj = solve_obj.clone();
            let mut stat = SolveStat::zero();
            solve_obj.pool.spawn_ok(async move {
                let next_depth = depth + child_obj.params.ybwc_younger_add;
                let child_future = solve_outer(
                    &mut child_obj, next, -alpha-1, -alpha, false, next_depth);
                let (child_res, child_stat) = child_future.await;
                stat.merge(child_stat);
                let tmp = -child_res;
                if tmp >= res {
                    res = tmp;
                    best = pos;
                }
                if res < beta {
                    if res > alpha {
                        alpha = res;
                        let child_future = solve_outer(
                            &mut child_obj, next, -beta, -alpha, false, next_depth);
                        let (child_res, child_stat) = child_future.await;
                        stat.merge(child_stat);
                        res = max(res, -child_res);
                    }
                }
                let res_tuple = (res, best);
                let _ = tx.unbounded_send((res_tuple, stat));
            });
        }
    }
    drop(tx);
    while let Some((res_tuple, child_stat)) = rx.next().await {
        stat.merge(child_stat);
        let (child_res, child_best) = res_tuple;
        if child_res > res {
            res = child_res;
            best = child_best;
        }
    }
    (res, best, stat)
}

enum CacheLookupResult {
    Cut(i8),
    NoCut(i8, i8, u8)
}

fn lookup_table(solve_obj: &mut SolveObj, board: Board, alpha: &mut i8, beta: &mut i8) -> CacheLookupResult {
    let (lower, upper, old_best) = match solve_obj.res_cache.get(board) {
        Some(cache) => (cache.lower, cache.upper, cache.best),
        None => (-64, 64, PASS as u8)
    };
    let old_alpha = *alpha;
    *alpha = max(lower, *alpha);
    *beta = min(upper, *beta);
    if *alpha >= *beta {
        if old_alpha > upper {
            CacheLookupResult::Cut(upper)
        } else {
            CacheLookupResult::Cut(lower)
        }
    } else {
        CacheLookupResult::NoCut(lower, upper, old_best)
    }
}

fn update_table(solve_obj: &mut SolveObj, board: Board, res: i8, best: u8, alpha: i8, beta: i8, lower: i8, upper: i8) {
    let range = if res <= alpha {
        (lower, min(upper, res))
    } else if res >= beta {
        (max(lower, res), upper)
    } else {
        (res, res)
    };
    solve_obj.res_cache.update(ResCache {
        board,
        lower: range.0,
        upper: range.1,
        gen: solve_obj.res_cache.gen,
        best });
}

fn stability_cut(board: Board, alpha: &mut i8, beta: &mut i8) -> CutType {
    let (bits_me, bits_op) = board.stable_partial();
    let lower = 2 * popcnt(bits_me) - 64;
    let upper = 64 - 2 * popcnt(bits_op);
    if upper <= *alpha {
        CutType::LessThanAlpha(upper)
    } else if lower >= *beta {
        CutType::MoreThanBeta(lower)
    } else {
        CutType::NoCut
    }
}

fn solve_inner(
        solve_obj: &mut SolveObj, board: Board, mut alpha: i8, mut beta: i8, passed: bool,
        depth: i8) -> (i8, SolveStat) {
    let rem = popcnt(board.empty());
    if rem == 0 {
        (board.score(), SolveStat::zero())
    } else if rem == 1 {
        near_leaf(board)
    } else if rem < solve_obj.params.ffs_ordering_limit {
        naive(solve_obj, board, alpha, beta, passed, depth)
    } else {
        match stability_cut(board.clone(), &mut alpha, &mut beta) {
            CutType::NoCut => (),
            CutType::MoreThanBeta(v) => return (v, SolveStat { node_count: 1, st_cut_count: 1 }),
            CutType::LessThanAlpha(v) => return (v, SolveStat { node_count: 1, st_cut_count: 1 })
        }
        if rem < solve_obj.params.res_cache_limit {
            fastest_first(solve_obj, board, alpha, beta, passed, depth)
        } else {
            let (lower, upper, old_best) = match lookup_table(solve_obj, board, &mut alpha, &mut beta) {
                CacheLookupResult::Cut(v) => return (v, SolveStat::zero()),
                CacheLookupResult::NoCut(l, u, b) => (l, u, b)
            };
            let (res, best, stat) = if rem < solve_obj.params.eval_ordering_limit {
                let (res, stat) = fastest_first(solve_obj, board, alpha, beta, passed, depth);
                (res, PASS as u8, stat)
            } else {
                move_ordering_by_eval(solve_obj, board.clone(), alpha, beta, passed, old_best, depth)
            };
            update_table(solve_obj, board, res, best, alpha, beta, lower, upper);
            (res, stat)
        }
    }
}

fn solve_outer(
    solve_obj: &mut SolveObj, board: Board, mut alpha: i8, mut beta: i8, passed: bool,
    depth: i8) -> BoxFuture<'static, (i8, SolveStat)> {
    let mut solve_obj = solve_obj.clone();
    async move {
        let rem = popcnt(board.empty());
        if depth >= solve_obj.params.ybwc_depth_limit || rem < solve_obj.params.ybwc_empties_limit {
            solve_inner(&mut solve_obj, board, alpha, beta, passed, depth)
        } else {
            match stability_cut(board.clone(), &mut alpha, &mut beta) {
                CutType::NoCut => (),
                CutType::MoreThanBeta(v) => return (v, SolveStat { node_count: 1, st_cut_count: 1 }),
                CutType::LessThanAlpha(v) => return (v, SolveStat { node_count: 1, st_cut_count: 1 })
            }
            let (lower, upper, old_best) = match lookup_table(&mut solve_obj, board, &mut alpha, &mut beta) {
                CacheLookupResult::Cut(v) => return (v, SolveStat::zero()),
                CacheLookupResult::NoCut(l, u, b) => (l, u, b)
            };
            let (res, best, stat) = ybwc(&mut solve_obj, board.clone(), alpha, beta, passed, old_best, depth).await;
            update_table(&mut solve_obj, board, res, best, alpha, beta, lower, upper);
            (res, stat)
        }
    }.boxed()
}

pub fn solve(
        solve_obj: &mut SolveObj, board: Board, alpha: i8, beta: i8, passed: bool,
        depth: i8) -> (i8, SolveStat) {
    executor::block_on(solve_outer(solve_obj, board, alpha, beta, passed, depth))
}

fn think_impl(board: Board, mut alpha: i16, beta: i16, passed: bool,
         evaluator: Arc<Evaluator>,
         cache: &mut EvalCacheTable,
         old_best: u8, depth: i8) -> (i16, usize) {
    let mut v = vec![(0i16, 0i16, 0i8, 0usize, board.clone()); 0];
    let mut w = vec![(0i8, 0usize, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                let bonus = if pos as u8 == old_best {
                    -16 * SCALE
                } else {
                    0
                };
                v.push((bonus + weighted_mobility(&next) as i16, 0, 0, pos, next));
            },
            Err(_) => ()
        }
    }
    v.sort_by(|a, b| {
        if a.0 == b.0 {
            if a.1 == b.1 {
                a.2.cmp(&b.2)
            } else {
                a.1.cmp(&b.1)
            }
        } else {
            a.0.cmp(&b.0)
        }
    });
    w.sort_by(|a, b| a.0.cmp(&b.0));
    let mut nexts = Vec::new();
    for (_, _, _, pos, next) in &v {
        nexts.push((*pos, next.clone()));
    }
    for (_, pos, next) in &w {
        nexts.push((*pos, next.clone()));
    }
    let mut res = -64 * SCALE;
    let mut best = PASS;
    for (i, (pos, next)) in nexts.iter().enumerate() {
        if i == 0 {
            res = -think(
                    next.clone(), -beta, -alpha, false, evaluator.clone(),
                    cache, depth-1);
            best = *pos;
        } else {
            let reduce = if -evaluator.eval(next.clone()) < alpha - 16 * SCALE {
                2
            } else {
                1
            };
            let tmp = -think(
                    next.clone(), -alpha-1, -alpha, false, evaluator.clone(),
                    cache, depth-reduce);
            if tmp > res {
                res = tmp;
                best = *pos;
            }
            if res >= beta {
                return (res, best);
            }
            if res > alpha {
                alpha = res;
                res = res.max(-think(
                        next.clone(), -beta, -alpha, false, evaluator.clone(),
                        cache, depth-1));
            }
        }
        alpha = alpha.max(res);
        if alpha >= beta {
            return (res, best);
        }
    }
    if nexts.is_empty() {
        if passed {
            return ((board.score() as i16) * SCALE, PASS);
        } else {
            return (-think(
                board.pass(), -beta, -alpha, true, evaluator,
                cache, depth), PASS);
        }
    }
    (res, best)
}

pub fn think(board: Board, alpha: i16, beta: i16, passed: bool,
         evaluator: Arc<Evaluator>,
         cache: &mut EvalCacheTable,
         depth: i8) -> i16 {
    if depth <= 0 {
        let res = evaluator.eval(board.clone());
        res
    } else {
        let (lower, upper, old_best) = match cache.get(board.clone()) {
            Some(entry) => {
                if entry.depth >= depth {
                    (entry.lower, entry.upper, entry.best)
                } else {
                    (-64 * SCALE, 64 * SCALE, entry.best)
                }
            },
            None => (-64 * SCALE, 64 * SCALE, PASS as u8)
        };
        let new_alpha = alpha.max(lower);
        let new_beta = beta.min(upper);
        if new_alpha >= new_beta {
            return if alpha > upper {
                upper
            } else {
                lower
            }
        }
        let (res, best) = think_impl(
            board.clone(), new_alpha, new_beta, passed, evaluator,
            cache, old_best, depth);
        let range = if res <= new_alpha {
            (-64 * SCALE, res)
        } else if res >= new_beta {
            (res, 64 * SCALE)
        } else {
            (res, res)
        };
        let entry = EvalCache {
            board: board.clone(),
            lower: range.0,
            upper: range.1,
            gen: cache.gen,
            best: best as u8,
            depth
        };
        cache.update(entry);
        res
    }
}
