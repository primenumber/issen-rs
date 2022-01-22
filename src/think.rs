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

pub struct Searcher {
    pub evaluator: Arc<Evaluator>,
    pub cache: EvalCacheTable,
    pub timer: Option<Timer>,
}

impl Searcher {
    fn think_impl(
        &mut self,
        board: Board,
        mut alpha: i16,
        beta: i16,
        passed: bool,
        old_best: Option<Hand>,
        depth: i8,
    ) -> Option<(i16, Hand)> {
        let mut v = vec![(0i16, 0i16, 0i8, Hand::Pass, board); 0];
        for (next, pos) in board.next_iter() {
            let bonus = if Some(pos) == old_best {
                -16 * SCALE
            } else {
                0
            };
            v.push((bonus + weighted_mobility(&next) as i16, 0, 0, pos, next));
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
        let mut nexts = Vec::new();
        for &(_, _, _, pos, next) in &v {
            nexts.push((pos, next));
        }
        let mut res = -64 * SCALE;
        let mut best = None;
        for (i, &(pos, next)) in nexts.iter().enumerate() {
            if i == 0 {
                res = -self.think(next, -beta, -alpha, false, depth - 1)?.0;
                best = Some(pos);
            } else {
                let reduce = if -self.evaluator.eval(next) < alpha - 16 * SCALE {
                    2
                } else {
                    1
                };
                let tmp = -self
                    .think(next, -alpha - 1, -alpha, false, depth - reduce)?
                    .0;
                if tmp > res {
                    res = tmp;
                    best = Some(pos);
                }
                if res >= beta {
                    return Some((res, best.unwrap()));
                }
                if res > alpha {
                    alpha = res;
                    res = res.max(-self.think(next, -beta, -alpha, false, depth - 1)?.0);
                }
            }
            alpha = alpha.max(res);
            if alpha >= beta {
                return Some((res, best.unwrap()));
            }
        }
        if nexts.is_empty() {
            if passed {
                return Some(((board.score() as i16) * SCALE, Hand::Pass));
            } else {
                return Some((
                    -self.think(board.pass(), -beta, -alpha, true, depth)?.0,
                    Hand::Pass,
                ));
            }
        }
        Some((res, best.unwrap()))
    }

    pub fn think(
        &mut self,
        board: Board,
        alpha: i16,
        beta: i16,
        passed: bool,
        depth: i8,
    ) -> Option<(i16, Option<Hand>)> {
        if depth <= 0 {
            let res = self.evaluator.eval(board);
            Some((res, None))
        } else {
            if depth > 8 {
                if let Some(t) = &self.timer {
                    if !t.is_ok() {
                        return None;
                    }
                }
            }
            let (lower, upper, old_best) = if depth > 2 {
                match self.cache.get(board) {
                    Some(entry) => {
                        if entry.depth >= depth {
                            (entry.lower, entry.upper, entry.best)
                        } else {
                            (-64 * SCALE, 64 * SCALE, entry.best)
                        }
                    }
                    None => (-64 * SCALE, 64 * SCALE, None),
                }
            } else {
                (-64 * SCALE, 64 * SCALE, None)
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
            let (res, best) =
                self.think_impl(board, new_alpha, new_beta, passed, old_best, depth)?;
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
                gen: self.cache.gen,
                best: Some(best),
                depth,
            };
            self.cache.update(entry);
            Some((res, Some(best)))
        }
    }

    pub fn think_with_move(
        &mut self,
        board: Board,
        mut alpha: i16,
        beta: i16,
        passed: bool,
        depth: i8,
    ) -> Option<(i16, Hand)> {
        let (score, hand) = self.think(board, alpha, beta, passed, depth)?;

        if let Some(b) = hand {
            return Some((score, b));
        }

        let mut current_score = -64 * SCALE - 1;
        let mut current_hand = None;
        let mut pass = true;
        for (next, hand) in board.next_iter() {
            let s = -self.think(next, -beta, -alpha, false, depth - 1)?.0;
            if s > current_score {
                current_hand = Some(hand);
                current_score = s;
                alpha = max(alpha, current_score);
            }
            pass = false;
        }
        if pass {
            Some((score, Hand::Pass))
        } else {
            Some((score, current_hand.unwrap()))
        }
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
) -> (i16, Hand, i8) {
    let start = Instant::now();
    let timer = Timer {
        period: start,
        time_limit,
    };
    let min_depth = 3;
    let mut current_depth = min_depth;

    let mut searcher = Searcher {
        evaluator,
        cache: cache.clone(),
        timer: Some(timer),
    };

    let (mut score, mut hand) = searcher
        .think_with_move(board, alpha, beta, passed, min_depth)
        .unwrap();

    if !searcher.timer.as_ref().unwrap().is_ok() {
        return (score, hand, current_depth);
    }

    for depth in (min_depth + 1).. {
        cache.inc_gen();
        let t = match searcher.think_with_move(board, alpha, beta, passed, depth) {
            Some(t) => t,
            _ => return (score, hand, current_depth),
        };
        score = t.0;
        hand = t.1;
        current_depth = depth;
    }
    (score, hand, current_depth)
}
