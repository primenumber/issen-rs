use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::table::*;
use std::cmp::max;
use std::sync::Arc;
use std::time::Instant;

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
    let mut v = vec![(0i16, 0i16, 0i8, 0usize, board); 0];
    let mut w = vec![(0i8, 0usize, board); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        if let Ok(next) = board.play(pos) {
            let bonus = if pos as u8 == old_best {
                -16 * SCALE
            } else {
                0
            };
            v.push((bonus + weighted_mobility(&next) as i16, 0, 0, pos, next));
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
    for &(_, _, _, pos, next) in &v {
        nexts.push((pos, next));
    }
    for &(_, pos, next) in &w {
        nexts.push((pos, next));
    }
    let mut res = -64 * SCALE;
    let mut best = PASS;
    for (i, &(pos, next)) in nexts.iter().enumerate() {
        if i == 0 {
            res = -think(
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
            best = pos;
        } else {
            let reduce = if -evaluator.eval(next) < alpha - 16 * SCALE {
                2
            } else {
                1
            };
            let tmp = -think(
                next,
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
                best = pos;
            }
            if res >= beta {
                return Some((res, best));
            }
            if res > alpha {
                alpha = res;
                res = res.max(
                    -think(
                        next,
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

pub fn think(
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
        let res = evaluator.eval(board);
        Some((res, None))
    } else {
        if depth > 8 {
            if let Some(t) = timer {
                if !t.is_ok() {
                    return None;
                }
            }
        }
        let (lower, upper, old_best) = if depth > 2 {
            match cache.get(board) {
                Some(entry) => {
                    if entry.depth >= depth {
                        (entry.lower, entry.upper, entry.best)
                    } else {
                        (-64 * SCALE, 64 * SCALE, entry.best)
                    }
                }
                None => (-64 * SCALE, 64 * SCALE, PASS as u8),
            }
        } else {
            (-64 * SCALE, 64 * SCALE, PASS as u8)
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
            board, new_alpha, new_beta, passed, evaluator, cache, old_best, timer, depth,
        )?;
        let range = if res <= new_alpha {
            (-64 * SCALE, res)
        } else if res >= new_beta {
            (res, 64 * SCALE)
        } else {
            (res, res)
        };
        let entry = EvalCache {
            board,
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

    if let Some(b) = hand {
        return Some((score, b));
    }

    let mut current_score = -64 * SCALE;
    let mut current_hand = PASS;
    let mut pass = true;
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        if let Ok(next) = board.play(pos) {
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
        cache.inc_gen();
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
