use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::endgame::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::table::*;
use futures::channel::mpsc;
use futures::channel::mpsc::UnboundedSender;
use futures::future::{BoxFuture, FutureExt};
use futures::StreamExt;
use std::cmp::max;
use std::sync::Arc;

struct YBWCContext {
    tx: UnboundedSender<((i8, Option<Hand>), SolveStat)>,
    next: Board,
    pos: Hand,
    alpha: i8,
    beta: i8,
    res: i8,
    best: Option<Hand>,
}

async fn ybwc_child(mut solve_obj: SolveObj, sub_solver: Arc<SubSolver>, depth: i8, mut cxt: YBWCContext) {
    let mut stat = SolveStat::zero();
    let next_depth = depth + solve_obj.params.ybwc_younger_add;
    let child_future = solve_outer(
        &mut solve_obj,
        &sub_solver,
        cxt.next,
        -cxt.alpha - 1,
        -cxt.alpha,
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
            -cxt.beta,
            -tmp,
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
    sub_solver: &Arc<SubSolver>,
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
            let (child_res, _child_best, child_stat) = solve_outer(
                solve_obj,
                sub_solver,
                board.pass(),
                -beta,
                -alpha,
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
                solve_obj, sub_solver, next, -beta, -alpha, false, next_depth,
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
                -alpha - 1,
                -alpha,
                false,
                depth,
            )
            .await;
            stat.merge(child_stat);
            let mut tmp = -child_res;
            if alpha < tmp && tmp < beta {
                let (child_res, _child_best, child_stat) =
                    solve_outer(solve_obj, sub_solver, next, -beta, -tmp, false, depth).await;
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

pub fn solve_outer(
    solve_obj: &mut SolveObj,
    sub_solver: &Arc<SubSolver>,
    board: Board,
    mut alpha: i8,
    mut beta: i8,
    passed: bool,
    depth: i8,
) -> BoxFuture<'static, (i8, Option<Hand>, SolveStat)> {
    let mut solve_obj = solve_obj.clone();
    let sub_solver = sub_solver.clone();
    async move {
        let rem = popcnt(board.empty());
        if rem < solve_obj.params.ybwc_empties_limit {
            let (res, stat) = if sub_solver.workers.is_empty() {
                solve_inner(&mut solve_obj, board, alpha, beta, passed)
            } else {
                sub_solver.solve_remote(board, alpha, beta).await.unwrap()
            };
            return (res, None, stat);
        }
        match stability_cut(board, alpha, beta) {
            CutType::NoCut => (),
            CutType::MoreThanBeta(v) => return (v, None, SolveStat::one_stcut()),
            CutType::LessThanAlpha(v) => return (v, None, SolveStat::one_stcut()),
        }
        let (lower, upper, old_best) = match lookup_table(&mut solve_obj, board, &mut alpha, &mut beta) {
            CacheLookupResult::Cut(v) => return (v, None, SolveStat::zero()),
            CacheLookupResult::NoCut(l, u, b) => (l, u, b),
        };
        let (res, best, stat) = ybwc(
            &mut solve_obj,
            &sub_solver,
            board,
            alpha,
            beta,
            passed,
            old_best,
            depth,
        )
        .await;
        if rem >= solve_obj.params.res_cache_limit {
            update_table(
                solve_obj.res_cache,
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
    .boxed()
}
