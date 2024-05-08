use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::endgame::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::table::*;
use dashmap::DashSet;
use std::cmp::max;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;

struct ABDADAContext<Eval: Evaluator> {
    solve_obj: SolveObj<Eval>,
    cs_hash: Arc<DashSet<Board>>,
    finished: Arc<AtomicBool>,
    stats: SolveStat,
}

fn simplified_abdada_body<Eval: Evaluator>(
    ctx: &mut ABDADAContext<Eval>,
    board: Board,
    (mut alpha, beta): (i8, i8),
    passed: bool,
    depth: i8,
) -> Option<(i8, Option<Hand>)> {
    let v = move_ordering_impl(&mut ctx.solve_obj, board, None);
    if v.is_empty() {
        if passed {
            return Some((board.score(), Some(Hand::Pass)));
        }
        let (child_res, _child_best) =
            simplified_abdada_intro(ctx, board.pass_unchecked(), (-beta, -alpha), true, depth)?;
        return Some((-child_res, Some(Hand::Pass)));
    }
    let mut res = -(BOARD_SIZE as i8);
    let mut best = None;
    let mut deffered = Vec::new();
    for (i, (pos, next)) in v.into_iter().enumerate() {
        if i == 0 {
            let (cres, _chand) = simplified_abdada_intro(ctx, next, (-beta, -alpha), false, depth + 1)?;
            res = -cres;
            best = Some(pos);
            alpha = max(alpha, res);
            if alpha >= beta {
                return Some((res, best));
            }
            continue;
        }
        if defer_search(next, &ctx.cs_hash) {
            deffered.push((pos, next));
            continue;
        }
        start_search(next, &ctx.cs_hash);
        let (cres, _chand) = simplified_abdada_intro(ctx, next, (-alpha - 1, -alpha), false, depth + 1)?;
        finish_search(next, &ctx.cs_hash);
        let mut tmp = -cres;
        if alpha < tmp && tmp < beta {
            let (cres, _chand) = simplified_abdada_intro(ctx, next, (-beta, -tmp), false, depth + 1)?;
            tmp = -cres;
        }
        if tmp >= beta {
            return Some((tmp, Some(pos)));
        }
        if tmp > res {
            best = Some(pos);
            res = tmp;
            alpha = max(alpha, res);
        }
    }
    for (pos, next) in deffered {
        // NWS
        let (cres, _chand) = simplified_abdada_intro(ctx, next, (-alpha - 1, -alpha), false, depth + 1)?;
        let mut tmp = -cres;
        if alpha < tmp && tmp < beta {
            let (cres, _chand) = simplified_abdada_intro(ctx, next, (-beta, -tmp), false, depth + 1)?;
            tmp = -cres;
        }
        if tmp >= beta {
            return Some((tmp, Some(pos)));
        }
        if tmp > res {
            best = Some(pos);
            res = tmp;
            alpha = max(alpha, res);
        }
    }
    Some((res, best))
}

fn simplified_abdada_intro<Eval: Evaluator>(
    ctx: &mut ABDADAContext<Eval>,
    board: Board,
    (mut alpha, mut beta): (i8, i8),
    passed: bool,
    depth: i8,
) -> Option<(i8, Option<Hand>)> {
    let rem = popcnt(board.empty());
    if depth >= ctx.solve_obj.params.parallel_depth_limit || rem < ctx.solve_obj.params.parallel_empties_limit {
        let (res, stat) = solve_inner(&mut ctx.solve_obj, board, (alpha, beta), passed);
        ctx.stats.merge(stat);
        ctx.solve_obj.local_cache_gen += 1;
        return Some((res, None));
    }
    ctx.stats.merge(SolveStat::one());
    if ctx.finished.load(std::sync::atomic::Ordering::SeqCst) {
        return None;
    }
    match stability_cut(board, (alpha, beta)) {
        CutType::NoCut => (),
        CutType::MoreThanBeta(v) => return Some((v, None)),
        CutType::LessThanAlpha(v) => return Some((v, None)),
    }
    let (lower, upper, _old_best) = match lookup_table(&mut ctx.solve_obj, board, (&mut alpha, &mut beta)) {
        CacheLookupResult::Cut(v) => return Some((v, None)),
        CacheLookupResult::NoCut(l, u, b) => (l, u, b),
    };
    let (res, best) = simplified_abdada_body(ctx, board, (alpha, beta), passed, depth)?;
    if rem >= ctx.solve_obj.params.res_cache_limit {
        update_table(
            ctx.solve_obj.res_cache.clone(),
            ctx.solve_obj.cache_gen,
            board,
            res,
            best,
            (alpha, beta),
            (lower, upper),
        );
    }
    Some((res, best))
}

fn start_search(board: Board, cs_hash: &Arc<DashSet<Board>>) {
    cs_hash.insert(board);
}

fn finish_search(board: Board, cs_hash: &Arc<DashSet<Board>>) {
    cs_hash.remove(&board);
}

fn defer_search(board: Board, cs_hash: &Arc<DashSet<Board>>) -> bool {
    cs_hash.contains(&board)
}

pub fn simplified_abdada<Eval: Evaluator>(
    solve_obj: &mut SolveObj<Eval>,
    board: Board,
    (alpha, beta): (i8, i8),
    passed: bool,
    depth: i8,
    num_threads: Option<usize>,
) -> (i8, Option<Hand>, SolveStat) {
    thread::scope(|s| {
        let mut handles = Vec::new();
        let cs_hash = Arc::new(DashSet::new());
        let finished = Arc::new(AtomicBool::new(false));
        let num_threads = num_threads.unwrap_or(num_cpus::get());
        for _ in 0..num_threads {
            let solve_obj = solve_obj.clone();
            let cs_hash = cs_hash.clone();
            let finished = finished.clone();
            let finished2 = finished.clone();
            handles.push(s.spawn(move || {
                let mut ctx = ABDADAContext {
                    solve_obj,
                    cs_hash,
                    finished,
                    stats: SolveStat::zero(),
                };
                let res = simplified_abdada_intro(&mut ctx, board, (alpha, beta), passed, depth);
                finished2.store(true, std::sync::atomic::Ordering::SeqCst);
                (res, ctx.stats)
            }));
        }
        let mut stat = SolveStat::zero();
        let mut res = -(BOARD_SIZE as i8);
        let mut best = None;
        for h in handles {
            let (tres, tstat) = h.join().unwrap();
            stat.merge(tstat);
            if let Some((tscore, tbest)) = tres {
                res = tscore;
                best = tbest;
            }
        }
        (res, best, stat)
    })
}
