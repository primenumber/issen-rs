use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::table::*;
use std::cmp::max;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct Timer {
    pub period: Instant,
    pub time_limit: u128,
}

impl Timer {
    fn is_ok(&self) -> bool {
        self.period.elapsed().as_millis() <= self.time_limit
    }
}

pub struct Searcher {
    pub evaluator: Arc<Evaluator>,
    pub cache: Arc<EvalCacheTable>,
    pub timer: Option<Timer>,
    pub node_count: usize,
}

pub const DEPTH_SCALE: i32 = 256;

impl Searcher {
    fn think_naive(
        &mut self,
        board: Board,
        mut alpha: i16,
        beta: i16,
        passed: bool,
        depth: i32,
    ) -> Option<(i16, Hand)> {
        let mut res = EVAL_SCORE_MIN - 1;
        let mut best = None;
        for (i, (next, pos)) in board.next_iter().enumerate() {
            if i == 0 {
                res = res.max(
                    -self
                        .think(next, -beta, -alpha, false, depth - DEPTH_SCALE)?
                        .0,
                );
                best = Some(pos);
                if res >= beta {
                    return Some((res, pos));
                }
                alpha = max(alpha, res);
            } else {
                let tmp = -self
                    .think(next, -alpha - 1, -alpha, false, depth - DEPTH_SCALE)?
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
                    res = res.max(
                        -self
                            .think(next, -beta, -alpha, false, depth - DEPTH_SCALE)?
                            .0,
                    );
                }
            }
        }
        if best == None {
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

    fn think_impl(
        &mut self,
        board: Board,
        mut alpha: i16,
        beta: i16,
        passed: bool,
        old_best: Option<Hand>,
        depth: i32,
    ) -> Option<(i16, Hand)> {
        let mut v = Vec::with_capacity(16);
        for (next, pos) in board.next_iter() {
            let use_eval_depth = 4;
            let bonus = if Some(pos) == old_best {
                -16 * SCALE
            } else {
                0
            };
            let eval_score = if depth < use_eval_depth * DEPTH_SCALE {
                0
            } else {
                self.evaluator.eval(next)
            };
            v.push((
                next,
                pos,
                bonus + weighted_mobility(&next) as i16 * SCALE + eval_score as i16,
                eval_score,
                0,
            ));
        }
        v.sort_by(|a, b| (a.2, a.3, a.4).cmp(&(b.2, b.3, b.4)));
        let mut res = EVAL_SCORE_MIN;
        let mut best = None;
        for (i, &(next, pos, _, eval_score, _)) in v.iter().enumerate() {
            if i == 0 {
                res = -self
                    .think(next, -beta, -alpha, false, depth - DEPTH_SCALE)?
                    .0;
                best = Some(pos);
            } else {
                let reduce = if -eval_score < alpha - 8 * SCALE {
                    4 * DEPTH_SCALE
                } else if -eval_score < alpha - 5 * SCALE {
                    3 * DEPTH_SCALE
                } else if -eval_score < alpha - 3 * SCALE {
                    2 * DEPTH_SCALE
                } else {
                    DEPTH_SCALE
                };
                let next_depth = max(2 * DEPTH_SCALE, depth - reduce);
                let tmp = -self.think(next, -alpha - 1, -alpha, false, next_depth)?.0;
                if tmp > res {
                    res = tmp;
                    best = Some(pos);
                }
                if res >= beta {
                    return Some((res, best.unwrap()));
                }
                if res > alpha {
                    alpha = res;
                    res = res.max(
                        -self
                            .think(next, -beta, -alpha, false, depth - DEPTH_SCALE)?
                            .0,
                    );
                }
            }
            alpha = alpha.max(res);
            if alpha >= beta {
                return Some((res, best.unwrap()));
            }
        }
        if v.is_empty() {
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
        depth: i32,
    ) -> Option<(i16, Option<Hand>)> {
        self.node_count += 1;
        if depth <= 0 {
            let res = self.evaluator.eval(board);
            Some((res, None))
        } else if depth <= 3 * DEPTH_SCALE {
            let (res, best) = self.think_naive(board, alpha, beta, passed, depth)?;
            Some((res, Some(best)))
        } else {
            if depth > 8 * DEPTH_SCALE {
                if let Some(t) = &self.timer {
                    if !t.is_ok() {
                        return None;
                    }
                }
            }
            let min_cache_depth = 4 * DEPTH_SCALE;
            let (lower, upper, old_best) = if depth >= min_cache_depth {
                match self.cache.get(board) {
                    Some(entry) => {
                        if entry.depth >= depth {
                            (entry.lower, entry.upper, entry.best)
                        } else {
                            (EVAL_SCORE_MIN, EVAL_SCORE_MAX, entry.best)
                        }
                    }
                    None => (EVAL_SCORE_MIN, EVAL_SCORE_MAX, None),
                }
            } else {
                (EVAL_SCORE_MIN, EVAL_SCORE_MAX, None)
            };
            let new_alpha = alpha.max(lower);
            let new_beta = beta.min(upper);
            if new_alpha >= new_beta {
                return if alpha >= upper {
                    Some((upper, old_best))
                } else {
                    Some((lower, old_best))
                };
            }
            let (res, best) =
                self.think_impl(board, new_alpha, new_beta, passed, old_best, depth)?;
            if depth >= min_cache_depth {
                let range = if res <= new_alpha {
                    (EVAL_SCORE_MIN, res)
                } else if res >= new_beta {
                    (res, EVAL_SCORE_MAX)
                } else {
                    (res, res)
                };
                let entry = EvalCache {
                    board,
                    lower: range.0,
                    upper: range.1,
                    gen: self.cache.gen.load(std::sync::atomic::Ordering::SeqCst),
                    best: Some(best),
                    depth,
                };
                self.cache.update(entry);
            }
            Some((res, Some(best)))
        }
    }

    pub fn think_with_move(
        &mut self,
        board: Board,
        mut alpha: i16,
        beta: i16,
        passed: bool,
        depth: i32,
    ) -> Option<(i16, Hand)> {
        let (score, hand) = self.think(board, alpha, beta, passed, depth)?;

        if let Some(b) = hand {
            return Some((score, b));
        }

        let mut current_score = EVAL_SCORE_MIN - 1;
        let mut current_hand = None;
        let mut pass = true;
        for (next, hand) in board.next_iter() {
            let s = -self
                .think(next, -beta, -alpha, false, depth - DEPTH_SCALE)?
                .0;
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

    pub fn iterative_think(
        &mut self,
        board: Board,
        alpha: i16,
        beta: i16,
        passed: bool,
    ) -> (i16, Hand, i8) {
        let min_depth = 3;
        let mut current_depth = min_depth;

        let (mut score, mut hand) = self
            .think_with_move(board, alpha, beta, passed, min_depth)
            .unwrap();

        if !self.timer.as_ref().unwrap().is_ok() {
            return (score, hand, current_depth as i8);
        }

        for depth in (min_depth + 1).. {
            self.cache.inc_gen();
            let t = match self.think_with_move(board, alpha, beta, passed, depth * DEPTH_SCALE) {
                Some(t) => t,
                _ => return (score, hand, current_depth as i8),
            };
            score = t.0;
            hand = t.1;
            current_depth = depth;
        }
        (score, hand, current_depth as i8)
    }
}
