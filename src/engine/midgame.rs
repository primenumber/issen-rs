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
use num_cpus;
use std::cmp::max;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

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

fn simplified_abdada_young(
    solve_obj: &mut SolveObj,
    (alpha, beta): (i8, i8),
    stat: &mut SolveStat,
    deffered: &mut Vec<(Hand, Board)>,
    (pos, next): (Hand, Board),
    res: &mut i8,
    best: &mut Option<Hand>,
    depth: i8,
    cs_hash: &Arc<Mutex<HashMap<Board, usize>>>,
) -> Option<(i8, Option<Hand>, SolveStat)> {
    if defer_search(next, cs_hash) {
        deffered.push((pos, next));
        return None;
    }
    // NWS
    start_search(next, cs_hash);
    let (cres, _chand, cstat) = simplified_abdada_intro(
        solve_obj,
        next,
        (-alpha - 1, -alpha),
        false,
        depth + 1,
        cs_hash,
    );
    finish_search(next, cs_hash);
    stat.merge(cstat);
    let mut tmp = -cres;
    if alpha < tmp && tmp < beta {
        let (cres, _chand, cstat) = simplified_abdada_intro(solve_obj, next, (-beta, -tmp), false, depth + 1, cs_hash);
        stat.merge(cstat);
        tmp = -cres;
    }
    if tmp >= beta {
        return Some((tmp, Some(pos), *stat));
    }
    if tmp > *res {
        *best = Some(pos);
        *res = tmp;
    }
    None
}

fn simplified_abdada_body(
    solve_obj: &mut SolveObj,
    board: Board,
    (mut alpha, beta): (i8, i8),
    passed: bool,
    depth: i8,
    cs_hash: &Arc<Mutex<HashMap<Board, usize>>>,
) -> (i8, Option<Hand>, SolveStat) {
    let v = move_ordering_impl(solve_obj, board, None);
    let mut stat = SolveStat::one();
    if v.is_empty() {
        if passed {
            return (board.score(), Some(Hand::Pass), stat);
        } else {
            let (child_res, _child_best, child_stat) = simplified_abdada_intro(
                solve_obj,
                board.pass_unchecked(),
                (-beta, -alpha),
                true,
                depth,
                cs_hash,
            );
            stat.merge(child_stat);
            return (-child_res, Some(Hand::Pass), stat);
        }
    }
    let mut res = -(BOARD_SIZE as i8);
    let mut best = None;
    let mut stat = SolveStat::one();
    let mut deffered = Vec::new();
    for (i, (pos, next)) in v.into_iter().enumerate() {
        if i == 0 {
            let (cres, _chand, cstat) =
                simplified_abdada_intro(solve_obj, next, (-beta, -alpha), false, depth + 1, cs_hash);
            stat.merge(cstat);
            res = -cres;
            best = Some(pos);
            alpha = max(alpha, res);
            if alpha >= beta {
                return (res, best, stat);
            }
            continue;
        }
        if let Some(ret) = simplified_abdada_young(
            solve_obj,
            (alpha, beta),
            &mut stat,
            &mut deffered,
            (pos, next),
            &mut res,
            &mut best,
            depth,
            cs_hash,
        ) {
            return ret;
        }
    }
    for (pos, next) in deffered {
        // NWS
        let (cres, _chand, cstat) = simplified_abdada_intro(
            solve_obj,
            next,
            (-alpha - 1, -alpha),
            false,
            depth + 1,
            cs_hash,
        );
        stat.merge(cstat);
        let mut tmp = -cres;
        if alpha < tmp && tmp < beta {
            let (cres, _chand, cstat) =
                simplified_abdada_intro(solve_obj, next, (-beta, -tmp), false, depth + 1, cs_hash);
            stat.merge(cstat);
            tmp = -cres;
        }
        if tmp >= beta {
            return (tmp, Some(pos), stat);
        }
        if tmp > res {
            best = Some(pos);
            res = tmp;
        }
    }
    (res, best, stat)
}

fn simplified_abdada_intro(
    solve_obj: &mut SolveObj,
    board: Board,
    (mut alpha, mut beta): (i8, i8),
    passed: bool,
    depth: i8,
    cs_hash: &Arc<Mutex<HashMap<Board, usize>>>,
) -> (i8, Option<Hand>, SolveStat) {
    let rem = popcnt(board.empty());
    if depth >= solve_obj.params.ybwc_depth_limit || rem < solve_obj.params.ybwc_empties_limit {
        let (res, stat) = solve_inner(solve_obj, board, (alpha, beta), passed);
        return (res, None, stat);
    }
    match stability_cut(board, (alpha, beta)) {
        CutType::NoCut => (),
        CutType::MoreThanBeta(v) => return (v, None, SolveStat::one_stcut()),
        CutType::LessThanAlpha(v) => return (v, None, SolveStat::one_stcut()),
    }
    let (lower, upper, _old_best) = match lookup_table(solve_obj, board, (&mut alpha, &mut beta)) {
        CacheLookupResult::Cut(v) => return (v, None, SolveStat::zero()),
        CacheLookupResult::NoCut(l, u, b) => (l, u, b),
    };
    let (res, best, stat) = simplified_abdada_body(solve_obj, board, (alpha, beta), passed, depth, cs_hash);
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

fn start_search(board: Board, cs_hash: &Arc<Mutex<HashMap<Board, usize>>>) {
    let mut locked_table = cs_hash.lock().unwrap();
    match locked_table.get_mut(&board) {
        Some(nproc) => *nproc += 1,
        None => {
            locked_table.insert(board, 1);
        }
    }
}

fn finish_search(board: Board, cs_hash: &Arc<Mutex<HashMap<Board, usize>>>) {
    let mut locked_table = cs_hash.lock().unwrap();
    match locked_table.get_mut(&board) {
        Some(nproc) => {
            *nproc -= 1;
            if *nproc == 0 {
                locked_table.remove(&board);
            }
        }
        None => {
            panic!();
        }
    }
}

fn defer_search(board: Board, cs_hash: &Arc<Mutex<HashMap<Board, usize>>>) -> bool {
    cs_hash.lock().unwrap().contains_key(&board)
}

pub fn simplified_abdada(
    solve_obj: &mut SolveObj,
    board: Board,
    (alpha, beta): (i8, i8),
    passed: bool,
    depth: i8,
) -> (i8, Option<Hand>, SolveStat) {
    thread::scope(|s| {
        let mut handles = Vec::new();
        let cs_hash = Arc::new(Mutex::new(HashMap::new()));
        for _ in 0..num_cpus::get_physical() {
            let mut solve_obj = solve_obj.clone();
            let cs_hash = cs_hash.clone();
            handles.push(s.spawn(move || {
                simplified_abdada_intro(
                    &mut solve_obj,
                    board,
                    (alpha, beta),
                    passed,
                    depth,
                    &cs_hash,
                )
            }));
        }
        let mut stat = SolveStat::zero();
        let mut res = -(BOARD_SIZE as i8);
        let mut best = None;
        for h in handles {
            let (tres, tbest, tstat) = h.join().unwrap();
            stat.merge(tstat);
            res = tres;
            best = tbest;
        }
        (res, best, stat)
    })
}
