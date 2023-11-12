use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::endgame::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::table::*;
use crate::engine::think::*;
use futures::channel::mpsc;
use futures::channel::mpsc::UnboundedSender;
use futures::future::{BoxFuture, FutureExt};
use futures::StreamExt;
use std::cmp::max;
use std::collections::HashMap;
use std::future::*;

struct YBWCContext {
    tx: UnboundedSender<((i8, Option<Hand>), SolveStat)>,
    next: Board,
    pos: Hand,
    alpha: i8,
    beta: i8,
    res: i8,
    best: Option<Hand>,
}

async fn ybwc_child(mut solve_obj: SolveObj, sub_solver: SubSolver, depth: i8, mut cxt: YBWCContext) {
    let mut stat = SolveStat::zero();
    let next_depth = depth + solve_obj.params.ybwc_younger_add;
    let child_future = solve_outer(
        &mut solve_obj,
        &sub_solver,
        cxt.next,
        (-cxt.alpha - 1, -cxt.alpha),
        false,
        next_depth,
    );
    let (child_res, _child_best, child_stat) = child_future.await;
    stat.merge(child_stat);
    let mut tmp = -child_res;
    if cxt.alpha < tmp && tmp < cxt.beta {
        let child_future = solve_outer(
            &mut solve_obj,
            &sub_solver,
            cxt.next,
            (-cxt.beta, -tmp),
            false,
            next_depth,
        );
        let (child_res, _child_best, child_stat) = child_future.await;
        stat.merge(child_stat);
        tmp = -child_res;
    }
    if tmp > cxt.res {
        cxt.best = Some(cxt.pos);
        cxt.res = tmp;
    }
    let res_tuple = (cxt.res, cxt.best);
    let _ = cxt.tx.unbounded_send((res_tuple, stat));
}

async fn ybwc(
    solve_obj: &mut SolveObj,
    sub_solver: &SubSolver,
    board: Board,
    (mut alpha, beta): (i8, i8),
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
            let (child_res, _child_best, child_stat) = solve_outer(
                solve_obj,
                sub_solver,
                board.pass_unchecked(),
                (-beta, -alpha),
                true,
                depth,
            )
            .await;
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
            let (child_res, _child_best, child_stat) = solve_outer(
                solve_obj,
                sub_solver,
                next,
                (-beta, -alpha),
                false,
                next_depth,
            )
            .await;
            stat.merge(child_stat);
            res = -child_res;
            best = Some(pos);
            alpha = max(alpha, res);
            if alpha >= beta {
                return (res, best, stat);
            }
        } else if depth < solve_obj.params.ybwc_depth_limit {
            let tx = tx.clone();
            let context = YBWCContext {
                tx,
                next,
                pos,
                alpha,
                beta,
                res,
                best,
            };
            handles.push(tokio::task::spawn(ybwc_child(
                solve_obj.clone(),
                sub_solver.clone(),
                depth,
                context,
            )));
        } else {
            let (child_res, _child_best, child_stat) = solve_outer(
                solve_obj,
                sub_solver,
                next,
                (-alpha - 1, -alpha),
                false,
                depth,
            )
            .await;
            stat.merge(child_stat);
            let mut tmp = -child_res;
            if alpha < tmp && tmp < beta {
                let (child_res, _child_best, child_stat) =
                    solve_outer(solve_obj, sub_solver, next, (-beta, -tmp), false, depth).await;
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
                for handle in &handles {
                    handle.abort();
                }
                rx.close();
                return (res, best, stat);
            }
        }
    }
    (res, best, stat)
}

struct APHIDCache<'a> {
    estimated_score: i16,
    lower: i8,
    upper: i8,
    future: BoxFuture<'a, (i8, SolveStat)>,
}

async fn negascout_impl_2<'a, 'b>(
    solve_obj: &mut SolveObj,
    sub_solver: &SubSolver,
    next: Board,
    (alpha, beta): (i8, i8),
    is_first: bool,
    depth: i8,
    aphid_cache: &'a mut HashMap<Board, APHIDCache<'b>>,
) -> (usize, i8, Option<Hand>, SolveStat)
where
    'a: 'b,
{
    if is_first {
        aphid_impl(
            solve_obj,
            sub_solver,
            next,
            (-beta, -alpha),
            false,
            None,
            depth,
            aphid_cache,
        )
        .await
    } else {
        let (mut unsolved_leaves, res, mut hand, mut stat) = aphid_impl(
            solve_obj,
            sub_solver,
            next,
            (-alpha - 1, -alpha),
            false,
            None,
            depth,
            aphid_cache,
        )
        .await;
        let mut neg_result = -res;
        if neg_result >= beta {
            return (unsolved_leaves, res, hand, stat);
        }
        if neg_result > alpha {
            let (unsolved_leaves2, res2, hand2, stat2) = aphid_impl(
                solve_obj,
                sub_solver,
                next,
                (-beta, -neg_result),
                false,
                None,
                depth,
                aphid_cache,
            )
            .await;
            stat.merge(stat2);
            neg_result = -res2;
            hand = hand2;
            unsolved_leaves += unsolved_leaves2;
        }
        (unsolved_leaves, -neg_result, hand, stat)
    }
}

fn aphid_impl<'a, 'b, 'c, 'd>(
    solve_obj: &'a mut SolveObj,
    sub_solver: &'b SubSolver,
    board: Board,
    (mut alpha, beta): (i8, i8),
    passed: bool,
    old_best: Option<Hand>,
    depth: i8,
    aphid_cache: &'c mut HashMap<Board, APHIDCache<'c>>,
) -> BoxFuture<'d, (usize, i8, Option<Hand>, SolveStat)>
where
    'a: 'd,
    'b: 'd,
    'c: 'd,
{
    async move {
        if depth >= solve_obj.params.ybwc_depth_limit {
            if let Some(entry) = aphid_cache.get(&board) {
            } else {
                let mut searcher = Searcher {
                    evaluator: solve_obj.evaluator.clone(),
                    cache: solve_obj.eval_cache.clone(),
                    timer: None,
                    node_count: 0,
                    cache_gen: solve_obj.cache_gen,
                };
                let think_depth = 3;
                let score = searcher
                    .think(
                        board,
                        EVAL_SCORE_MIN,
                        EVAL_SCORE_MAX,
                        false,
                        think_depth as i32 * DEPTH_SCALE,
                    )
                    .unwrap()
                    .0;
                aphid_cache.insert(
                    board,
                    APHIDCache {
                        estimated_score: score,
                        lower: -(BOARD_SIZE as i8),
                        upper: BOARD_SIZE as i8,
                        future: sub_solver.solve_remote(board, (alpha, beta)),
                    },
                );
                return (1, score / SCALE, None, SolveStat::one());
            }
        }
        let v = move_ordering_impl(solve_obj, board, old_best);
        let mut res = -(BOARD_SIZE as i8);
        let mut best = None;
        let mut stat = SolveStat::one();
        let mut unsolved_leaves = 0;
        for (i, &(pos, next)) in v.iter().enumerate() {
            let (child_unsolved_leaves, child_res, _child_hand, child_stat) = negascout_impl_2(
                solve_obj,
                sub_solver,
                next,
                (alpha, beta),
                i == 0,
                depth,
                aphid_cache,
            )
            .await;
            if -child_res > res {
                res = -child_res;
                best = Some(pos);
            }
            stat.merge(child_stat);
            alpha = max(alpha, res);
            unsolved_leaves += child_unsolved_leaves;
            if res >= beta {
                return (unsolved_leaves, res, best, stat);
            }
        }
        if v.is_empty() {
            if passed {
                return (unsolved_leaves, board.score(), Some(Hand::Pass), stat);
            } else {
                let (child_unsolved_leaves, child_res, _child_hand, child_stat) = aphid_impl(
                    solve_obj,
                    sub_solver,
                    board.pass_unchecked(),
                    (-beta, -alpha),
                    true,
                    None,
                    depth,
                    aphid_cache,
                )
                .await;
                stat.merge(child_stat);
                return (child_unsolved_leaves, -child_res, Some(Hand::Pass), stat);
            }
        }
        (unsolved_leaves, res, best, stat)
    }
    .boxed()
}

async fn aphid(
    solve_obj: &mut SolveObj,
    sub_solver: &SubSolver,
    board: Board,
    (alpha, beta): (i8, i8),
    passed: bool,
    old_best: Option<Hand>,
    depth: i8,
) -> (i8, Option<Hand>, SolveStat) {
    let mut aphid_cache = HashMap::<Board, APHIDCache>::new();
    loop {
        let (unsolved_leaves, res, best, stat) = aphid_impl(
            solve_obj,
            sub_solver,
            board,
            (alpha, beta),
            passed,
            old_best,
            depth,
            &mut aphid_cache,
        )
        .await;
        if unsolved_leaves == 0 {
            return (res, best, stat);
        }
    }
}

pub fn solve_outer<'a, 'b, 'c>(
    solve_obj: &'a mut SolveObj,
    sub_solver: &'b SubSolver,
    board: Board,
    (mut alpha, mut beta): (i8, i8),
    passed: bool,
    depth: i8,
) -> BoxFuture<'c, (i8, Option<Hand>, SolveStat)>
where
    'a: 'c,
    'b: 'c,
{
    async move {
        let rem = popcnt(board.empty());
        if rem < solve_obj.params.ybwc_empties_limit {
            let (res, stat) = if sub_solver.workers.is_empty() {
                solve_inner(solve_obj, board, (alpha, beta), passed)
            } else {
                sub_solver.solve_remote(board, (alpha, beta)).await.unwrap()
            };
            return (res, None, stat);
        }
        match stability_cut(board, (alpha, beta)) {
            CutType::NoCut => (),
            CutType::MoreThanBeta(v) => return (v, None, SolveStat::one_stcut()),
            CutType::LessThanAlpha(v) => return (v, None, SolveStat::one_stcut()),
        }
        let (lower, upper, old_best) = match lookup_table(solve_obj, board, (&mut alpha, &mut beta)) {
            CacheLookupResult::Cut(v) => return (v, None, SolveStat::zero()),
            CacheLookupResult::NoCut(l, u, b) => (l, u, b),
        };
        let (res, best, stat) = ybwc(
            solve_obj,
            sub_solver,
            board,
            (alpha, beta),
            passed,
            old_best,
            depth,
        )
        .await;
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
        (res, best, stat)
    }
    .boxed()
}
