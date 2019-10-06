use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::cmp::{min, max};
use std::sync::mpsc;
use std::cell::Cell;
use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::table::*;

type ResCacheTable = Arc<Mutex<HashMap<Board, (i8, i8, u8)>>>;

pub struct SolveObj<'a> {
    res_cache: ResCacheTable,
    pub eval_cache: EvalCacheTable,
    evaluator: &'a Evaluator,
    pub count: Cell<usize>,
    pub st_cut: Cell<usize>,
    reduce: bool
}

enum CutType {
    NoCut,
    MoreThanBeta(i8),
    LessThanAlpha(i8)
}

impl SolveObj<'_> {
    pub fn new(res_cache: ResCacheTable, eval_cache: EvalCacheTable,
           evaluator: &Evaluator, reduce: bool) -> SolveObj {
        SolveObj {
            res_cache,
            eval_cache,
            evaluator,
            count: Cell::new(0),
            st_cut: Cell::new(0),
            reduce
        }
    }

    fn near_leaf(&self, board: Board) -> i8 {
        let bit = board.empty();
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => -next.score(),
            Err(_) => {
                self.count.set(self.count.get() + 1);
                match board.pass().play(pos) {
                    Ok(next) => next.score(),
                    Err(_) => board.score()
                }
            }
        }
    }

    fn naive(
            &self, board: Board, mut alpha: i8, beta: i8, passed: bool,
            depth: i8)-> i8 {
        let mut pass = true;
        let mut empties = board.empty();
        let mut res = -64;
        while empties != 0 {
            let bit = empties  & empties.wrapping_neg();
            empties = empties & (empties - 1);
            let pos = popcnt(bit - 1) as usize;
            match board.play(pos) {
                Ok(next) => {
                    pass = false;
                    res = max(res, -self.solve(
                            next, -beta, -alpha, false, depth+1));
                    alpha = max(alpha, res);
                    if alpha >= beta {
                        return res;
                    }
                },
                Err(_) => ()
            }
        }
        if pass {
            if passed {
                return board.score();
            } else {
                return -self.solve(board.pass(), -beta, -alpha, true, depth);
            }
        }
        res
    }

    fn fastest_first(
            &self, board: Board, mut alpha: i8, beta: i8, passed: bool,
            depth: i8) -> i8 {
        let mut v = vec![(0i8, board.clone()); 0];
        let mut empties = board.empty();
        while empties != 0 {
            let bit = empties  & empties.wrapping_neg();
            empties = empties & (empties - 1);
            let pos = popcnt(bit - 1) as usize;
            match board.play(pos) {
                Ok(next) => {
                    v.push((weighted_mobility(& next), next));
                },
                Err(_) => ()
            }
        }
        v.sort_by(|a, b| a.0.cmp(&b.0));
        let mut res = -64;
        for (i, &(_, ref next)) in v.iter().enumerate() {
            if i == 0 {
                res = max(res, -self.solve(
                        next.clone(), -beta, -alpha, false, depth+1));
            } else {
                let mut result = -self.solve(
                    next.clone(), -alpha-1, -alpha, false, depth+1);
                if result >= beta {
                    return result;
                }
                if result > alpha {
                    alpha = result;
                    result = -self.solve(
                        next.clone(), -beta, -alpha, false, depth+1);
                }
                res = max(res, result);
            }
            alpha = max(alpha, res);
            if alpha >= beta {
                return res;
            }
        }
        if v.is_empty() {
            if passed {
                return board.score();
            } else {
                return -self.solve(board.pass(), -beta, -alpha, true, depth);
            }
        }
        res
    }

    fn move_ordering_impl(&self, board: Board, old_best: u8, _depth: i8)
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
                    let new_res = think(next.clone(), alpha, beta, false, self.evaluator, &self.eval_cache, think_depth);
                    if new_res <= alpha {
                        let new_alpha = -64 * SCALE;
                        let new_beta = new_res;
                        res = think(next.clone(), new_alpha, new_beta, false, self.evaluator, &self.eval_cache, think_depth);
                        tmp.push((res + bonus, *pos, next.clone()));
                    } else if new_res >= beta {
                        let new_alpha = new_res;
                        let new_beta = 64 * SCALE;
                        res = think(next.clone(), new_alpha, new_beta, false, self.evaluator, &self.eval_cache, think_depth);
                        tmp.push((res + bonus, *pos, next.clone()));
                    } else {
                        res = new_res;
                        tmp.push((res + bonus, *pos, next.clone()));
                    }
                } else {
                    let new_res = think(next.clone(), res, res+1, false, self.evaluator, &self.eval_cache, think_depth);
                    if new_res < res {
                        let fixed_res = think(next.clone(), -64 * SCALE, new_res, false, self.evaluator, &self.eval_cache, think_depth);
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
        if nexts.len() > 0 && self.reduce {
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
            &self, board: Board, mut alpha: i8, beta: i8, passed: bool,
            old_best: u8, depth: i8) -> (i8, u8) {
        let v = self.move_ordering_impl(board.clone(), old_best, depth);
        let mut res = -64;
        let mut best = PASS as u8;
        for (i, (pos, next)) in v.iter().enumerate() {
            if i == 0 {
                res = -self.solve(next.clone(), -beta, -alpha, false, depth+1);
                best = *pos;
                alpha = max(alpha, res);
                if alpha >= beta {
                    return (res, best);
                }
            } else {
                let tmp = -self.solve(
                        next.clone(), -alpha-1, -alpha, false, depth+1);
                if tmp >= res {
                    res = tmp;
                    best = *pos;
                }
                if res >= beta {
                    return (res, best);
                }
                if res > alpha {
                    alpha = res;
                    res = max(res, -self.solve(
                            next.clone(), -beta, -alpha, false, depth+1));
                }
            }
        }
        if v.is_empty() {
            if passed {
                return (board.score(), PASS as u8);
            } else {
                return (-self.solve(board.pass(), -beta, -alpha, true, depth), PASS as u8);
            }
        }
        (res, best)
    }

    fn ybwc(
            &self, board: Board, mut alpha: i8, beta: i8, passed: bool,
            old_best: u8, depth: i8) -> (i8, u8) {
        let v = self.move_ordering_impl(board.clone(), old_best, depth);
        let (tx, rx) = mpsc::channel();
        let (txcount, rxcount) = mpsc::channel();
        let (res, best) = crossbeam::scope(|scope| {
            let mut handles = Vec::new();
            let mut res = -64;
            let mut best = PASS as u8;
            for (i, (pos, next)) in v.iter().enumerate() {
                if i == 0 {
                    res = -self.solve(next.clone(), -beta, -alpha, false, depth+1);
                    best = *pos;
                    alpha = max(alpha, res);
                    if alpha >= beta {
                        return (res, best);
                    }
                } else {
                    let tx = tx.clone();
                    let txcount = txcount.clone();
                    let child_obj = SolveObj::new(
                        self.res_cache.clone(),
                        self.eval_cache.clone(),
                        self.evaluator,
                        self.reduce);
                    handles.push(scope.spawn(move |_| {
                        let tmp = -child_obj.solve(
                                next.clone(), -alpha-1, -alpha, false, depth+2);
                        if tmp >= res {
                            res = tmp;
                            best = *pos;
                        }
                        if res >= beta {
                            let _ = tx.send((res, best));
                            let _ = txcount.send((child_obj.count.get(), child_obj.st_cut.get(), child_obj.eval_cache.cnt_get.get(), child_obj.eval_cache.cnt_update.get(), child_obj.eval_cache.cnt_hit.get()));
                            return;
                        }
                        if res > alpha {
                            alpha = res;
                            res = max(res, -child_obj.solve(
                                    next.clone(), -beta, -alpha, false, depth+2));
                        }
                        let _ = tx.send((res, best));
                        let _ = txcount.send((child_obj.count.get(), child_obj.st_cut.get(), child_obj.eval_cache.cnt_get.get(), child_obj.eval_cache.cnt_update.get(), child_obj.eval_cache.cnt_hit.get()));
                    }));
                }
            }
            for h in handles {
                let _ = h.join();
                let (cnt_solve, cnt_st_cut, cnt_get, cnt_update, cnt_hit) = rxcount.recv().unwrap();
                self.count.set(self.count.get() + cnt_solve);
                self.st_cut.set(self.st_cut.get() + cnt_st_cut);
                self.eval_cache.add_cnt_get(cnt_get);
                self.eval_cache.add_cnt_update(cnt_update);
                self.eval_cache.add_cnt_hit(cnt_hit);
                let (child_res, child_best) = rx.recv().unwrap();
                if child_res > res {
                    res = child_res;
                    best = child_best;
                }
            }
            alpha = max(alpha, res);
            (res, best)
        }).unwrap();
        if v.is_empty() {
            if passed {
                return (board.score(), PASS as u8);
            } else {
                return (-self.solve(board.pass(), -beta, -alpha, true, depth), PASS as u8);
            }
        }
        (res, best)
    }

    fn lookup_and_update_table(
            &self, board: Board, alpha: i8, beta: i8, passed: bool,
            depth: i8) -> i8 {
        let (lower, upper, old_best) = match self.res_cache.lock().unwrap().get(&board) {
            Some(t) => *t,
            None => (-64, 64, PASS as u8)
        };
        let new_alpha = max(lower, alpha);
        let new_beta = min(upper, beta);
        if new_alpha >= new_beta {
            return if alpha > upper {
                upper
            } else {
                lower
            }
        }
        let (res, best) = if depth >= 5 || popcnt(board.empty()) <= 16 {
            self.move_ordering_by_eval(
                board.clone(), alpha, beta, passed, old_best, depth)
        } else {
            self.ybwc(
                board.clone(), alpha, beta, passed, old_best, depth)
        };
        let range = if res <= new_alpha {
            (lower, min(upper, res))
        } else if res >= new_beta {
            (max(lower, res), upper)
        } else {
            (res, res)
        };
        self.res_cache.lock().unwrap().insert(board, (range.0, range.1, best));
        res
    }

    fn stability_cut(&self, board: Board, alpha: &mut i8, beta: &mut i8) -> CutType {
        let (bits_me, bits_op) = board.stable_partial();
        let lower = 2 * popcnt(bits_me) - 64;
        let upper = 64 - 2 * popcnt(bits_op);
        if upper <= *alpha {
            self.st_cut.set(self.st_cut.get() + 1);
            CutType::LessThanAlpha(upper)
        } else if lower >= *beta {
            self.st_cut.set(self.st_cut.get() + 1);
            CutType::MoreThanBeta(lower)
        } else {
            (*alpha).max(lower);
            (*beta).min(upper);
            CutType::NoCut
        }
    }

    pub fn solve(
            &self, board: Board, alpha: i8, beta: i8, passed: bool,
            depth: i8) -> i8 {
        self.count.set(self.count.get() + 1);
        let rem = popcnt(board.empty());
        if rem == 0 {
            board.score()
        } else if rem == 1 {
            self.near_leaf(board)
        } else if rem <= 6 {
            self.naive(board, alpha, beta, passed, depth)
        } else if rem <= 12 {
            let mut new_alpha = alpha;
            let mut new_beta = beta;
            match self.stability_cut(board.clone(), &mut new_alpha, &mut new_beta) {
                CutType::NoCut => self.fastest_first(board, new_alpha, new_beta, passed, depth),
                CutType::MoreThanBeta(v) => v,
                CutType::LessThanAlpha(v) => v
            }
        } else {
            let mut new_alpha = alpha;
            let mut new_beta = beta;
            match self.stability_cut(board.clone(), &mut new_alpha, &mut new_beta) {
                CutType::NoCut => self.lookup_and_update_table(board, new_alpha, new_beta, passed, depth),
                CutType::MoreThanBeta(v) => v,
                CutType::LessThanAlpha(v) => v
            }
        }
    }
}

fn think_impl(board: Board, mut alpha: i16, beta: i16, passed: bool,
         evaluator: & Evaluator,
         cache: &EvalCacheTable,
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
                    next.clone(), -beta, -alpha, false, evaluator,
                    cache, depth-1);
            best = *pos;
        } else {
            let reduce = if -evaluator.eval(next.clone()) < alpha - 16 * SCALE {
                2
            } else {
                1
            };
            let tmp = -think(
                    next.clone(), -alpha-1, -alpha, false, evaluator,
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
                        next.clone(), -beta, -alpha, false, evaluator,
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
         evaluator: & Evaluator,
         cache: &EvalCacheTable,
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
            gen: cache.gen.get(),
            best: best as u8,
            depth
        };
        cache.update(entry);
        res
    }
}
