#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::table::*;
use arrayvec::ArrayVec;
use core::arch::x86_64::_tzcnt_u64;
use std::cmp::max;

fn near_leaf(solve_obj: &mut SolveObj, board: Board) -> (i8, SolveStat) {
    let (score, node_count) = solve_obj.last_cache.solve_last(board);
    (
        score,
        SolveStat {
            node_count,
            st_cut_count: 0,
        },
    )
}

fn naive(solve_obj: &mut SolveObj, board: Board, (mut alpha, beta): (i8, i8), passed: bool) -> (i8, SolveStat) {
    let mut pass = true;
    let mut res = -(BOARD_SIZE as i8);
    let mut stat = SolveStat::one();
    for (next, _pos) in board.next_iter() {
        pass = false;
        let (child_res, child_stat) = solve_inner(solve_obj, next, (-beta, -alpha), false);
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
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass_unchecked(), (-beta, -alpha), true);
            stat.merge(child_stat);
            return (-child_res, stat);
        }
    }
    (res, stat)
}

fn static_order(solve_obj: &mut SolveObj, board: Board, (mut alpha, beta): (i8, i8), passed: bool) -> (i8, SolveStat) {
    let mut pass = true;
    let mut res = -(BOARD_SIZE as i8);
    let mut stat = SolveStat::one();
    const MASKS: [u64; 3] = [
        0x8100_0000_0000_0081, // Corner
        0x3C3C_FFFF_FFFF_3C3C, // Normal
        0x42C3_0000_0000_C342, // C + X
    ];
    let mobility_bits = board.mobility_bits();
    for mask in MASKS.iter() {
        let mut remain = mobility_bits & mask;
        while remain != 0 {
            let pos = unsafe { _tzcnt_u64(remain) } as usize;
            remain = remain & (remain - 1);
            if let Some(next) = board.play(pos) {
                pass = false;
                let (child_res, child_stat) = solve_inner(solve_obj, next, (-beta, -alpha), false);
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
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass_unchecked(), (-beta, -alpha), true);
            stat.merge(child_stat);
            return (-child_res, stat);
        }
    }
    (res, stat)
}

fn negascout_impl(solve_obj: &mut SolveObj, next: Board, (alpha, beta): (i8, i8), is_first: bool) -> (i8, SolveStat) {
    if is_first {
        solve_inner(solve_obj, next, (-beta, -alpha), false)
    } else {
        let (res, mut stat) = solve_inner(solve_obj, next, (-alpha - 1, -alpha), false);
        let mut neg_result = -res;
        if neg_result >= beta {
            return (res, stat);
        }
        if neg_result > alpha {
            let (res2, stat2) = solve_inner(solve_obj, next, (-beta, -neg_result), false);
            stat.merge(stat2);
            neg_result = -res2;
        }
        (-neg_result, stat)
    }
}

fn fastest_first(solve_obj: &mut SolveObj, board: Board, (mut alpha, beta): (i8, i8), passed: bool) -> (i8, SolveStat) {
    const MAX_FFS_NEXT: usize = 20;
    let mut nexts = ArrayVec::<_, MAX_FFS_NEXT>::new();
    for (next, _pos) in board.next_iter() {
        nexts.push((weighted_mobility(&next), next));
    }
    nexts.sort_by(|a, b| a.0.cmp(&b.0));
    let mut res = -(BOARD_SIZE as i8);
    let mut stat = SolveStat::one();
    for (i, &(_, next)) in nexts.iter().enumerate() {
        let (child_res, child_stat) = negascout_impl(solve_obj, next, (alpha, beta), i == 0);
        res = max(res, -child_res);
        stat.merge(child_stat);
        alpha = max(alpha, res);
        if alpha >= beta {
            return (res, stat);
        }
    }
    if nexts.is_empty() {
        if passed {
            return (board.score(), stat);
        } else {
            let (child_result, child_stat) = solve_inner(solve_obj, board.pass_unchecked(), (-beta, -alpha), true);
            stat.merge(child_stat);
            return (-child_result, stat);
        }
    }
    (res, stat)
}

fn move_ordering_by_eval(
    solve_obj: &mut SolveObj,
    board: Board,
    (mut alpha, beta): (i8, i8),
    passed: bool,
    old_best: Option<Hand>,
) -> (i8, Option<Hand>, SolveStat) {
    let v = move_ordering_impl(solve_obj, board, old_best);
    let mut res = -(BOARD_SIZE as i8);
    let mut best = None;
    let mut stat = SolveStat::one();
    for (i, &(pos, next)) in v.iter().enumerate() {
        let (child_res, child_stat) = negascout_impl(solve_obj, next, (alpha, beta), i == 0);
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
            let (child_res, child_stat) = solve_inner(solve_obj, board.pass_unchecked(), (-beta, -alpha), true);
            stat.merge(child_stat);
            return (-child_res, Some(Hand::Pass), stat);
        }
    }
    (res, best, stat)
}

pub fn solve_inner(
    solve_obj: &mut SolveObj,
    board: Board,
    (mut alpha, mut beta): (i8, i8),
    passed: bool,
) -> (i8, SolveStat) {
    let rem = popcnt(board.empty());
    if rem == 0 {
        (board.score(), SolveStat::zero())
    } else if rem == 1 {
        near_leaf(solve_obj, board)
    } else if rem < solve_obj.params.static_ordering_limit {
        naive(solve_obj, board, (alpha, beta), passed)
    } else if rem < solve_obj.params.ffs_ordering_limit {
        static_order(solve_obj, board, (alpha, beta), passed)
    } else {
        if rem >= solve_obj.params.stability_cut_limit {
            match stability_cut(board, (alpha, beta)) {
                CutType::NoCut => (),
                CutType::MoreThanBeta(v) => return (v, SolveStat::one_stcut()),
                CutType::LessThanAlpha(v) => return (v, SolveStat::one_stcut()),
            }
        }
        if rem < solve_obj.params.res_cache_limit {
            fastest_first(solve_obj, board, (alpha, beta), passed)
        } else if rem < solve_obj.params.eval_ordering_limit {
            let (lower, upper) = match lookup_table(solve_obj, board, (&mut alpha, &mut beta)) {
                CacheLookupResult::Cut(v) => return (v, SolveStat::zero()),
                CacheLookupResult::NoCut(l, u, _) => (l, u),
            };
            let (res, stat) = fastest_first(solve_obj, board, (alpha, beta), passed);
            update_table(
                solve_obj.res_cache.clone(),
                solve_obj.cache_gen,
                board,
                res,
                None,
                (alpha, beta),
                (lower, upper),
            );
            (res, stat)
        } else {
            let (lower, upper, old_best) = match lookup_table(solve_obj, board, (&mut alpha, &mut beta)) {
                CacheLookupResult::Cut(v) => return (v, SolveStat::zero()),
                CacheLookupResult::NoCut(l, u, b) => (l, u, b),
            };
            let (res, best, stat) = move_ordering_by_eval(solve_obj, board, (alpha, beta), passed, old_best);
            if rem >= solve_obj.params.res_cache_limit {
                update_table(
                    solve_obj.res_cache.clone(),
                    solve_obj.cache_gen,
                    board,
                    res,
                    best,
                    (alpha, beta),
                    (lower, upper),
                );
            }
            (res, stat)
        }
    }
}
