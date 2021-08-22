use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::table::*;
use bitintr::Tzcnt;
use futures::channel::mpsc;
use futures::executor;
use futures::executor::ThreadPool;
use futures::future::{BoxFuture, FutureExt};
use futures::task::SpawnExt;
use futures::StreamExt;
use std::cmp::{max, min};
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::time::Instant;

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
}

#[derive(Clone)]
pub struct SolveObj {
    res_cache: ResCacheTable,
    pub eval_cache: EvalCacheTable,
    evaluator: Arc<Evaluator>,
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

#[derive(Clone)]
pub struct Timer {
    period: Instant,
    time_limit: u128,
}

impl Timer {
    fn is_ok(&self) -> bool {
        self.period.elapsed().as_millis() <= self.time_limit
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
    depth: i8,
) -> (i8, SolveStat) {
    let mut pass = true;
    let mut empties = board.empty();
    let mut res = -64;
    let mut stat = SolveStat::one();
    while empties != 0 {
        let pos = empties.tzcnt() as usize;
        empties = empties & (empties - 1);
        match board.play(pos) {
            Ok(next) => {
                pass = false;
                let (child_res, child_stat) =
                    solve_inner(solve_obj, next, -beta, -alpha, false, depth + 1);
                res = max(res, -child_res);
                stat.merge(child_stat);
                alpha = max(alpha, res);
                if alpha >= beta {
                    return (res, stat);
                }
            }
            Err(_) => (),
        }
    }
    if pass {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_res, child_stat) =
                solve_inner(solve_obj, board.pass(), -beta, -alpha, true, depth);
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
    depth: i8,
) -> (i8, SolveStat) {
    let mut pass = true;
    let mut res = -64;
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
            match board.play(pos) {
                Ok(next) => {
                    pass = false;
                    let (child_res, child_stat) =
                        solve_inner(solve_obj, next, -beta, -alpha, false, depth + 1);
                    res = max(res, -child_res);
                    stat.merge(child_stat);
                    alpha = max(alpha, res);
                    if alpha >= beta {
                        return (res, stat);
                    }
                }
                Err(_) => (),
            }
        }
    }
    if pass {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_res, child_stat) =
                solve_inner(solve_obj, board.pass(), -beta, -alpha, true, depth);
            stat.merge(child_stat);
            return (-child_res, stat);
        }
    }
    (res, stat)
}

fn fastest_first(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    beta: i8,
    passed: bool,
    depth: i8,
) -> (i8, SolveStat) {
    const MAX_FFS_NEXT: usize = 20;
    let mut nexts: [(i8, Board); MAX_FFS_NEXT] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut count = 0;
    let mut empties = board.empty();
    while empties != 0 {
        let pos = empties.tzcnt() as usize;
        empties = empties & (empties - 1);
        match board.play(pos) {
            Ok(next) => {
                nexts[count] = (weighted_mobility(&next), next);
                count += 1;
                assert!(count <= MAX_FFS_NEXT);
            }
            Err(_) => (),
        }
    }
    nexts[0..count].sort_by(|a, b| a.0.cmp(&b.0));
    let mut res = -64;
    let mut stat = SolveStat::one();
    for (i, &(_, ref next)) in nexts[0..count].iter().enumerate() {
        if i == 0 {
            let (child_res, child_stat) =
                solve_inner(solve_obj, next.clone(), -beta, -alpha, false, depth + 1);
            stat.merge(child_stat);
            res = max(res, -child_res);
        } else {
            let (child_res, child_stat) = solve_inner(
                solve_obj,
                next.clone(),
                -alpha - 1,
                -alpha,
                false,
                depth + 1,
            );
            stat.merge(child_stat);
            let mut result = -child_res;
            if result >= beta {
                return (result, stat);
            }
            if result > alpha {
                alpha = result;
                let (child_res, child_stat) =
                    solve_inner(solve_obj, next.clone(), -beta, -alpha, false, depth + 1);
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
            let (child_result, child_stat) =
                solve_inner(solve_obj, board.pass(), -beta, -alpha, true, depth);
            stat.merge(child_stat);
            return (-child_result, stat);
        }
    }
    (res, stat)
}

fn move_ordering_impl(
    solve_obj: &mut SolveObj,
    board: Board,
    old_best: u8,
    _depth: i8,
) -> Vec<(u8, Board)> {
    let mut nexts = vec![(0i16, 0u8, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let pos = empties.tzcnt() as usize;
        empties = empties & (empties - 1);
        match board.play(pos) {
            Ok(next) => {
                nexts.push((0, pos as u8, next));
            }
            Err(_) => (),
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
            let bonus = if *pos == old_best { -16 * SCALE } else { 0 };
            if i == 0 {
                let window = if think_depth == min_depth { 16 } else { 8 };
                let alpha = (score - window * SCALE).max(-64 * SCALE);
                let beta = (score + window * SCALE).min(64 * SCALE);
                let new_res = think(
                    next.clone(),
                    alpha,
                    beta,
                    false,
                    solve_obj.evaluator.clone(),
                    &mut solve_obj.eval_cache,
                    &None,
                    think_depth,
                )
                .unwrap()
                .0;
                if new_res <= alpha {
                    let new_alpha = -64 * SCALE;
                    let new_beta = new_res;
                    res = think(
                        next.clone(),
                        new_alpha,
                        new_beta,
                        false,
                        solve_obj.evaluator.clone(),
                        &mut solve_obj.eval_cache,
                        &None,
                        think_depth,
                    )
                    .unwrap()
                    .0;
                    tmp.push((res + bonus, *pos, next.clone()));
                } else if new_res >= beta {
                    let new_alpha = new_res;
                    let new_beta = 64 * SCALE;
                    res = think(
                        next.clone(),
                        new_alpha,
                        new_beta,
                        false,
                        solve_obj.evaluator.clone(),
                        &mut solve_obj.eval_cache,
                        &None,
                        think_depth,
                    )
                    .unwrap()
                    .0;
                    tmp.push((res + bonus, *pos, next.clone()));
                } else {
                    res = new_res;
                    tmp.push((res + bonus, *pos, next.clone()));
                }
            } else {
                let new_res = think(
                    next.clone(),
                    res,
                    res + 1,
                    false,
                    solve_obj.evaluator.clone(),
                    &mut solve_obj.eval_cache,
                    &None,
                    think_depth,
                )
                .unwrap()
                .0;
                if new_res < res {
                    let fixed_res = think(
                        next.clone(),
                        -64 * SCALE,
                        new_res,
                        false,
                        solve_obj.evaluator.clone(),
                        &mut solve_obj.eval_cache,
                        &None,
                        think_depth,
                    )
                    .unwrap()
                    .0;
                    tmp.push((fixed_res + bonus, *pos, next.clone()));
                    res = fixed_res;
                } else {
                    let score = new_res;
                    tmp.push((score + bonus, *pos, next.clone()));
                }
            }
        }
        tmp.sort_by(|a, b| a.0.cmp(&b.0));
        nexts = tmp;
    }
    if nexts.len() > 0 && solve_obj.params.reduce {
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

async fn move_ordering_by_eval(
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    beta: i8,
    passed: bool,
    old_best: u8,
    depth: i8,
) -> (i8, u8, SolveStat) {
    let v = move_ordering_impl(solve_obj, board.clone(), old_best, depth);
    let mut res = -64;
    let mut best = PASS as u8;
    let mut stat = SolveStat::one();
    for (i, &(pos, next)) in v.iter().enumerate() {
        if i == 0 {
            let next_depth = depth + solve_obj.params.ybwc_elder_add;
            let (child_res, _child_best, child_stat) = if popcnt(board.empty())
                == solve_obj.params.ybwc_empties_limit
            {
                let mut child_obj = solve_obj.clone();
                solve_obj
                    .pool
                    .spawn_with_handle(async move {
                        solve_outer(&mut child_obj, next, -beta, -alpha, false, next_depth).await
                    })
                    .unwrap()
                    .await
            } else {
                solve_outer(solve_obj, next, -beta, -alpha, false, next_depth).await
            };
            stat.merge(child_stat);
            res = -child_res;
            best = pos;
            if res >= beta {
                return (res, best, stat);
            }
            alpha = max(alpha, res);
        } else {
            let next_depth = depth + solve_obj.params.ybwc_younger_add;
            let (child_res, _child_best, child_stat) =
                solve_outer(solve_obj, next, -alpha - 1, -alpha, false, next_depth).await;
            stat.merge(child_stat);
            let mut tmp = -child_res;
            if alpha < tmp && tmp < beta {
                let (child_res, _child_best, child_stat) =
                    solve_outer(solve_obj, next, -beta, -tmp, false, next_depth).await;
                stat.merge(child_stat);
                tmp = -child_res;
            }
            if tmp >= beta {
                return (tmp, pos, stat);
            }
            if tmp > res {
                res = tmp;
                best = pos;
                alpha = max(alpha, res);
            }
        }
    }
    if v.is_empty() {
        if passed {
            return (board.score(), PASS as u8, stat);
        } else {
            let (child_res, _child_best, child_stat) =
                solve_outer(solve_obj, board.pass(), -beta, -alpha, true, depth).await;
            stat.merge(child_stat);
            return (-child_res, PASS as u8, stat);
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
    old_best: u8,
    depth: i8,
) -> (i8, u8, SolveStat) {
    let v = move_ordering_impl(solve_obj, board.clone(), old_best, depth);
    let mut stat = SolveStat::one();
    if v.is_empty() {
        if passed {
            return (board.score(), PASS as u8, stat);
        } else {
            let (child_res, _child_best, child_stat) =
                solve_outer(solve_obj, board.pass(), -beta, -alpha, true, depth).await;
            stat.merge(child_stat);
            return (-child_res, PASS as u8, stat);
        }
    }
    let mut res = -64;
    let mut best = PASS as u8;
    let (tx, mut rx) = mpsc::unbounded();
    let mut handles = Vec::new();
    for (i, &(pos, next)) in v.iter().enumerate() {
        if i == 0 {
            let next_depth = depth + solve_obj.params.ybwc_elder_add;
            let (child_res, _child_best, child_stat) =
                solve_outer(solve_obj, next.clone(), -beta, -alpha, false, next_depth).await;
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
                            best = pos;
                            res = tmp;
                        }
                        let res_tuple = (res, best);
                        let _ = tx.unbounded_send((res_tuple, stat));
                    })
                    .unwrap(),
            );
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
                return (res, best, stat);
            }
        }
    }
    (res, best, stat)
}

#[derive(PartialEq, Debug)]
enum CacheLookupResult {
    Cut(i8),
    NoCut(i8, i8, u8),
}

fn make_lookup_result(
    res_cache: Option<ResCache>,
    alpha: &mut i8,
    beta: &mut i8,
) -> CacheLookupResult {
    let (lower, upper, old_best) = match res_cache {
        Some(cache) => (cache.lower, cache.upper, cache.best),
        None => (-64, 64, PASS as u8),
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

#[test]
fn test_lookup_result() {
    const TEST_BASE81: &str = "!#jiR;rO[ORNM2MN";
    let board = Board::from_base81(TEST_BASE81).unwrap();
    let res_cache = ResCache {
        board,
        lower: -24,
        upper: 16,
        gen: 3,
        best: 0,
    };
    // [alpha, beta] is contained in [lower, upper]
    {
        let mut alpha = -12;
        let mut beta = 4;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, -12);
        assert_eq!(beta, 4);
    }
    // [alpha, beta] contains [lower, upper]
    {
        let mut alpha = -30;
        let mut beta = 20;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, res_cache.upper);
    }
    // alpha < lower < beta < upper
    {
        let mut alpha = -32;
        let mut beta = 8;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, 8);
    }
    // lower < alpha < upper < beta
    {
        let mut alpha = -6;
        let mut beta = 26;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, -6);
        assert_eq!(beta, res_cache.upper);
    }
    // lower < upper < alpha < beta
    {
        let mut alpha = 22;
        let mut beta = 46;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.upper));
        assert_eq!(alpha, 22);
        assert_eq!(beta, res_cache.upper);
    }
    // alpha < beta < lower < upper
    {
        let mut alpha = -42;
        let mut beta = -24;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.lower));
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, -24);
    }
    // res_cache is None
    {
        let mut alpha = -6;
        let mut beta = 26;
        let result = make_lookup_result(None, &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::NoCut(-64, 64, PASS as u8));
        assert_eq!(alpha, -6);
        assert_eq!(beta, 26);
    }
    // lower < alpha = upper < beta
    {
        let mut alpha = 16;
        let mut beta = 26;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.upper));
        assert_eq!(alpha, 16);
        assert_eq!(beta, 16);
    }
    // alpha < beta = lower < upper
    {
        let mut alpha = -38;
        let mut beta = -24;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.lower));
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, -24);
    }
    // alpha < lower = upper < beta
    {
        let res_cache = ResCache {
            board,
            lower: 16,
            upper: 16,
            gen: 3,
            best: 0,
        };
        let mut alpha = -38;
        let mut beta = 30;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.lower));
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, res_cache.lower);
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

fn make_record(
    gen: u16,
    board: Board,
    res: i8,
    best: u8,
    alpha: i8,
    beta: i8,
    lower: i8,
    upper: i8,
) -> ResCache {
    let range = if res <= alpha {
        (lower, min(upper, res))
    } else if res >= beta {
        (max(lower, res), upper)
    } else {
        (res, res)
    };
    ResCache {
        board,
        lower: range.0,
        upper: range.1,
        gen: gen,
        best,
    }
}

fn update_table(
    solve_obj: &mut SolveObj,
    board: Board,
    res: i8,
    best: u8,
    alpha: i8,
    beta: i8,
    lower: i8,
    upper: i8,
) {
    let record = make_record(
        solve_obj.res_cache.gen,
        board,
        res,
        best,
        alpha,
        beta,
        lower,
        upper,
    );
    solve_obj.res_cache.update(record);
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
    solve_obj: &mut SolveObj,
    board: Board,
    mut alpha: i8,
    mut beta: i8,
    passed: bool,
    depth: i8,
) -> (i8, SolveStat) {
    let rem = popcnt(board.empty());
    if rem == 0 {
        (board.score(), SolveStat::zero())
    } else if rem == 1 {
        near_leaf(board)
    } else if rem < solve_obj.params.static_ordering_limit {
        naive(solve_obj, board, alpha, beta, passed, depth)
    } else if rem < solve_obj.params.ffs_ordering_limit {
        static_order(solve_obj, board, alpha, beta, passed, depth)
    } else {
        if rem >= solve_obj.params.stability_cut_limit {
            match stability_cut(board.clone(), &mut alpha, &mut beta) {
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
            fastest_first(solve_obj, board, alpha, beta, passed, depth)
        } else {
            let (lower, upper) = match lookup_table(solve_obj, board, &mut alpha, &mut beta) {
                CacheLookupResult::Cut(v) => return (v, SolveStat::zero()),
                CacheLookupResult::NoCut(l, u, _) => (l, u),
            };
            let (res, stat) = fastest_first(solve_obj, board, alpha, beta, passed, depth);
            update_table(solve_obj, board, res, PASS as u8, alpha, beta, lower, upper);
            (res, stat)
        }
    }
}

#[test]
fn test_solve_inner() {
    let name = "problem/stress_test_54_1k.b81r";
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    let evaluator = Arc::new(Evaluator::new("table"));
    let mut res_cache = ResCacheTable::new(256, 256);
    let mut eval_cache = EvalCacheTable::new(256, 256);
    let pool = ThreadPool::new().unwrap();
    let search_params = SearchParams {
        reduce: false,
        ybwc_depth_limit: 0,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 64,
        eval_ordering_limit: 64,
        res_cache_limit: 9,
        stability_cut_limit: 7,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
    };
    for (idx, line) in reader.lines().enumerate() {
        let line_str = line.unwrap();
        let desired: i8 = line_str[17..].parse().unwrap();
        match Board::from_base81(&line_str[..16]) {
            Ok(board) => {
                let mut obj = SolveObj::new(
                    res_cache.clone(),
                    eval_cache.clone(),
                    evaluator.clone(),
                    search_params.clone(),
                    pool.clone(),
                );
                let (res, stat) = solve_inner(&mut obj, board, -64, 64, false, 0);
                if res != desired {
                    board.print();
                }
                assert_eq!(res, desired);
            }
            Err(_) => {
                assert!(false);
            }
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
) -> BoxFuture<'static, (i8, Option<u8>, SolveStat)> {
    let mut solve_obj = solve_obj.clone();
    async move {
        let rem = popcnt(board.empty());
        if rem < solve_obj.params.eval_ordering_limit {
            let (res, stat) = solve_inner(&mut solve_obj, board, alpha, beta, passed, depth);
            (res, None, stat)
        } else {
            match stability_cut(board.clone(), &mut alpha, &mut beta) {
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
            let (res, best, stat) = if depth >= solve_obj.params.ybwc_depth_limit
                || rem < solve_obj.params.ybwc_empties_limit
            {
                move_ordering_by_eval(
                    &mut solve_obj,
                    board.clone(),
                    alpha,
                    beta,
                    passed,
                    old_best,
                    depth,
                )
                .await
            } else {
                ybwc(
                    &mut solve_obj,
                    board.clone(),
                    alpha,
                    beta,
                    passed,
                    old_best,
                    depth,
                )
                .await
            };
            if rem >= solve_obj.params.res_cache_limit {
                update_table(&mut solve_obj, board, res, best, alpha, beta, lower, upper);
            }
            (res, Some(best), stat)
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
) -> (i8, Option<u8>, SolveStat) {
    executor::block_on(solve_outer(solve_obj, board, alpha, beta, passed, depth))
}

pub async fn solve_with_move(board: Board, solve_obj: &mut SolveObj) -> usize {
    match solve_outer(solve_obj, board, -64, 64, false, 0).await.1 {
        Some(best) => best as usize,
        None => {
            let mut best_pos = None;
            let mut result = -65;
            for pos in board.mobility() {
                let next = board.play(pos).unwrap();
                let res = -solve_outer(solve_obj, next, -64, -result, false, 0).await.0;
                if res > result {
                    result = res;
                    best_pos = Some(pos);
                }
            }
            match best_pos {
                Some(pos) => pos,
                None => PASS,
            }
        }
    }
}

fn think_impl(
    board: Board,
    mut alpha: i16,
    beta: i16,
    passed: bool,
    evaluator: Arc<Evaluator>,
    cache: &mut EvalCacheTable,
    old_best: u8,
    timer: &Option<Timer>,
    depth: i8,
) -> Option<(i16, usize)> {
    let mut v = vec![(0i16, 0i16, 0i8, 0usize, board.clone()); 0];
    let mut w = vec![(0i8, 0usize, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties & empties.wrapping_neg();
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
            }
            Err(_) => (),
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
                next.clone(),
                -beta,
                -alpha,
                false,
                evaluator.clone(),
                cache,
                timer,
                depth - 1,
            )?
            .0;
            best = *pos;
        } else {
            let reduce = if -evaluator.eval(next.clone()) < alpha - 16 * SCALE {
                2
            } else {
                1
            };
            let tmp = -think(
                next.clone(),
                -alpha - 1,
                -alpha,
                false,
                evaluator.clone(),
                cache,
                timer,
                depth - reduce,
            )?
            .0;
            if tmp > res {
                res = tmp;
                best = *pos;
            }
            if res >= beta {
                return Some((res, best));
            }
            if res > alpha {
                alpha = res;
                res = res.max(
                    -think(
                        next.clone(),
                        -beta,
                        -alpha,
                        false,
                        evaluator.clone(),
                        cache,
                        timer,
                        depth - 1,
                    )?
                    .0,
                );
            }
        }
        alpha = alpha.max(res);
        if alpha >= beta {
            return Some((res, best));
        }
    }
    if nexts.is_empty() {
        if passed {
            return Some(((board.score() as i16) * SCALE, PASS));
        } else {
            return Some((
                -think(
                    board.pass(),
                    -beta,
                    -alpha,
                    true,
                    evaluator,
                    cache,
                    timer,
                    depth,
                )?
                .0,
                PASS,
            ));
        }
    }
    Some((res, best))
}

fn think(
    board: Board,
    alpha: i16,
    beta: i16,
    passed: bool,
    evaluator: Arc<Evaluator>,
    cache: &mut EvalCacheTable,
    timer: &Option<Timer>,
    depth: i8,
) -> Option<(i16, Option<usize>)> {
    if depth <= 0 {
        let res = evaluator.eval(board.clone());
        Some((res, None))
    } else {
        if depth > 8 {
            match timer {
                Some(t) => {
                    if !t.is_ok() {
                        return None;
                    }
                }
                _ => (),
            }
        }
        let (lower, upper, old_best) = match cache.get(board.clone()) {
            Some(entry) => {
                if entry.depth >= depth {
                    (entry.lower, entry.upper, entry.best)
                } else {
                    (-64 * SCALE, 64 * SCALE, entry.best)
                }
            }
            None => (-64 * SCALE, 64 * SCALE, PASS as u8),
        };
        let new_alpha = alpha.max(lower);
        let new_beta = beta.min(upper);
        if new_alpha >= new_beta {
            return if alpha > upper {
                Some((upper, None))
            } else {
                Some((lower, None))
            };
        }
        let (res, best) = think_impl(
            board.clone(),
            new_alpha,
            new_beta,
            passed,
            evaluator,
            cache,
            old_best,
            timer,
            depth,
        )?;
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
            depth,
        };
        cache.update(entry);
        Some((res, Some(best)))
    }
}

pub fn think_with_move(
    board: Board,
    mut alpha: i16,
    beta: i16,
    passed: bool,
    evaluator: Arc<Evaluator>,
    cache: &mut EvalCacheTable,
    timer: &Option<Timer>,
    depth: i8,
) -> Option<(i16, usize)> {
    let (score, hand) = think(
        board,
        alpha,
        beta,
        passed,
        evaluator.clone(),
        cache,
        timer,
        depth,
    )?;

    match hand {
        Some(b) => return Some((score, b)),
        None => (),
    }

    let mut current_score = -64 * SCALE;
    let mut current_hand = PASS;
    let mut pass = true;
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                let s = -think(
                    next,
                    -beta,
                    -alpha,
                    false,
                    evaluator.clone(),
                    cache,
                    timer,
                    depth - 1,
                )?
                .0;
                if s > current_score {
                    current_hand = pos;
                    current_score = s;
                    alpha = max(alpha, current_score);
                }
                pass = false;
            }
            Err(_) => (),
        }
    }
    if pass {
        Some((score, PASS))
    } else {
        Some((score, current_hand))
    }
}

pub fn iterative_think(
    board: Board,
    alpha: i16,
    beta: i16,
    passed: bool,
    evaluator: Arc<Evaluator>,
    cache: &mut EvalCacheTable,
    time_limit: u128,
) -> (i16, usize, i8) {
    let start = Instant::now();
    let timer = Timer {
        period: start,
        time_limit,
    };
    let min_depth = 3;
    let mut current_depth = min_depth;

    let (mut score, mut hand) = think_with_move(
        board,
        alpha,
        beta,
        passed,
        evaluator.clone(),
        cache,
        &Some(timer.clone()),
        min_depth,
    )
    .unwrap();

    if !timer.is_ok() {
        return (score, hand, current_depth);
    }

    for depth in (min_depth + 1).. {
        let t = match think_with_move(
            board,
            alpha,
            beta,
            passed,
            evaluator.clone(),
            cache,
            &Some(timer.clone()),
            depth,
        ) {
            Some(t) => t,
            _ => return (score, hand, current_depth),
        };
        score = t.0;
        hand = t.1;
        current_depth = depth;
    }
    (score, hand, current_depth)
}
