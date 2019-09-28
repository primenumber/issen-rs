use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::cmp::{min, max};
use std::sync::mpsc;
use std::cell::Cell;
use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::table::*;

type ResCacheTable = Arc<Mutex<HashMap<Board, (i8, i8)>>>;

pub struct SolveObj<'a> {
    res_cache: ResCacheTable,
    pub eval_cache: EvalCacheTable,
    evaluator: &'a Evaluator,
    pub count: Cell<usize>
}

impl SolveObj<'_> {
    pub fn new(res_cache: ResCacheTable, eval_cache: EvalCacheTable,
           evaluator: &Evaluator) -> SolveObj {
        SolveObj {
            res_cache,
            eval_cache,
            evaluator,
            count: Cell::new(0)
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

    fn move_ordering_impl(&self, board: Board, depth: i8) -> Vec<Board> {
        let mut nexts = vec![(0i16, board.clone()); 0];
        let mut empties = board.empty();
        while empties != 0 {
            let bit = empties  & empties.wrapping_neg();
            empties = empties & (empties - 1);
            let pos = popcnt(bit - 1) as usize;
            match board.play(pos) {
                Ok(next) => {
                    nexts.push((0, next));
                },
                Err(_) => ()
            }
        }

        let rem = popcnt(board.empty());
        let max_depth = ((rem - 12) as f32 / 1.8) as i8;
        let min_depth = (max_depth - 6).max(0);
        for think_depth in min_depth..=max_depth {
            let mut tmp = vec![(0i16, board.clone()); 0];
            let mut res = 64 * SCALE;
            for (i, (score, next)) in nexts.iter().enumerate() {
                if i == 0 {
                    const WINDOW: i16 = 10;
                    let alpha = (score - WINDOW * SCALE).max(-64 * SCALE);
                    let beta = (score + WINDOW * SCALE).min(64 * SCALE);
                    let new_res = think(next.clone(), alpha, beta, false, self.evaluator, &self.eval_cache, think_depth);
                    if new_res <= alpha {
                        let new_alpha = -64 * SCALE;
                        let new_beta = new_res;
                        res = think(next.clone(), new_alpha, new_beta, false, self.evaluator, &self.eval_cache, think_depth);
                        tmp.push((res, next.clone()));
                    } else if new_res >= beta {
                        let new_alpha = new_res;
                        let new_beta = 64 * SCALE;
                        res = think(next.clone(), new_alpha, new_beta, false, self.evaluator, &self.eval_cache, think_depth);
                        tmp.push((res, next.clone()));
                    } else {
                        res = new_res;
                        tmp.push((res, next.clone()));
                    }
                } else {
                    let new_res = think(next.clone(), res, res+1, false, self.evaluator, &self.eval_cache, think_depth);
                    if new_res < res {
                        let fixed_res = think(next.clone(), -64 * SCALE, new_res, false, self.evaluator, &self.eval_cache, think_depth);
                        tmp.push((fixed_res, next.clone()));
                        res = fixed_res;
                    } else {
                        let score = new_res;// + weighted_mobility(next) as i16 * SCALE;
                        tmp.push((score, next.clone()));
                    }
                }
            }
            tmp.sort_by(|a, b| {
                a.0.cmp(&b.0)
            });
            nexts = tmp;
        }
        nexts.into_iter().map(|e| e.1).collect()
    }

    fn move_ordering_by_eval(
            &self, board: Board, mut alpha: i8, beta: i8, passed: bool,
            depth: i8) -> i8 {
        let v = self.move_ordering_impl(board.clone(), depth);
        let mut res = -64;
        for (i, next) in v.iter().enumerate() {
            if i == 0 {
                res = -self.solve(next.clone(), -beta, -alpha, false, depth+1);
                alpha = max(alpha, res);
                if alpha >= beta {
                    return res;
                }
            } else {
                res = max(res, -self.solve(
                        next.clone(), -alpha-1, -alpha, false, depth+1));
                if res >= beta {
                    return res;
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
                return board.score();
            } else {
                return -self.solve(board.pass(), -beta, -alpha, true, depth);
            }
        }
        res
    }

    fn ybwc(
            &self, board: Board, mut alpha: i8, beta: i8, passed: bool,
            depth: i8) -> i8 {
        let v = self.move_ordering_impl(board.clone(), depth);
        let (tx, rx) = mpsc::channel();
        let (txcount, rxcount) = mpsc::channel();
        let res = crossbeam::scope(|scope| {
            let mut handles = Vec::new();
            let mut res = -64;
            for (i, next) in v.iter().enumerate() {
                if i == 0 {
                    res = -self.solve(next.clone(), -beta, -alpha, false, depth+1);
                    alpha = max(alpha, res);
                    if alpha >= beta {
                        return res;
                    }
                } else {
                    let tx = tx.clone();
                    let txcount = txcount.clone();
                    let child_obj = SolveObj::new(
                        self.res_cache.clone(),
                        self.eval_cache.clone(),
                        self.evaluator);
                    handles.push(scope.spawn(move |_| {
                        res = max(res, -child_obj.solve(
                                next.clone(), -alpha-1, -alpha, false, depth+2));
                        if res >= beta {
                            let _ = tx.send(res);
                            let _ = txcount.send((child_obj.count.get(), child_obj.eval_cache.cnt_get.get(), child_obj.eval_cache.cnt_update.get(), child_obj.eval_cache.cnt_hit.get()));
                            return;
                        }
                        if res > alpha {
                            alpha = res;
                            res = max(res, -child_obj.solve(
                                    next.clone(), -beta, -alpha, false, depth+2));
                        }
                        let _ = tx.send(res);
                        let _ = txcount.send((child_obj.count.get(), child_obj.eval_cache.cnt_get.get(), child_obj.eval_cache.cnt_update.get(), child_obj.eval_cache.cnt_hit.get()));
                    }));
                }
            }
            for h in handles {
                let _ = h.join();
                let (cnt_solve, cnt_get, cnt_update, cnt_hit) = rxcount.recv().unwrap();
                self.count.set(self.count.get() + cnt_solve);
                self.eval_cache.add_cnt_get(cnt_get);
                self.eval_cache.add_cnt_update(cnt_update);
                self.eval_cache.add_cnt_hit(cnt_hit);
                res = max(res, rx.recv().unwrap());
            }
            alpha = max(alpha, res);
            res
        }).unwrap();
        if v.is_empty() {
            if passed {
                return board.score();
            } else {
                return -self.solve(board.pass(), -beta, -alpha, true, depth);
            }
        }
        res 
    }

    fn lookup_and_update_table(
            &self, board: Board, alpha: i8, beta: i8, passed: bool,
            depth: i8) -> i8 {
        let (lower, upper) = match self.res_cache.lock().unwrap().get(&board) {
            Some((lower, upper)) => (*lower, *upper),
            None => (-64, 64)
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
        let res = if depth >= 5 || popcnt(board.empty()) <= 16 {
            self.move_ordering_by_eval(
                board.clone(), alpha, beta, passed, depth)
        } else {
            self.ybwc(
                board.clone(), alpha, beta, passed, depth)
        };
        let range = if res <= new_alpha {
            (lower, min(upper, res))
        } else if res >= new_beta {
            (max(lower, res), upper)
        } else {
            (res, res)
        };
        self.res_cache.lock().unwrap().insert(board, range);
        res
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
            self.fastest_first(board, alpha, beta, passed, depth)
        } else {
            self.lookup_and_update_table(board, alpha, beta, passed, depth)
        }
    }
}

fn think_impl(board: Board, mut alpha: i16, beta: i16, passed: bool,
         evaluator: & Evaluator,
         cache: &EvalCacheTable,
         depth: i8) -> i16 {
    let mut v = vec![(0i16, 0i16, 0i8, board.clone()); 0];
    let mut w = vec![(0i8, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                match cache.get(next.clone()) {
                    Some(entry) => {
                        v.push((entry.upper, entry.lower, weighted_mobility(&next), next));
                    },
                    None => {
                        w.push((weighted_mobility(&next), next));
                    }
                }
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
    for &(_, _, _, ref next) in &v {
        nexts.push(next.clone());
    }
    for &(_, ref next) in &w {
        nexts.push(next.clone());
    }
    let mut res = -64 * SCALE;
    for (i, next) in nexts.iter().enumerate() {
        if i == 0 {
            res = -think(
                    next.clone(), -beta, -alpha, false, evaluator,
                    cache, depth-1);
        } else {
            let reduce = if -evaluator.eval(next.clone()) < alpha - 16 * SCALE {
                2
            } else {
                1
            };
            res = res.max(-think(
                    next.clone(), -alpha-1, -alpha, false, evaluator,
                    cache, depth-reduce));
            if res >= beta {
                return res;
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
            return res;
        }
    }
    if nexts.is_empty() {
        if passed {
            return (board.score() as i16) * SCALE;
        } else {
            return -think(
                board.pass(), -beta, -alpha, true, evaluator,
                cache, depth);
        }
    }
    res
}

pub fn think(board: Board, alpha: i16, beta: i16, passed: bool,
         evaluator: & Evaluator,
         cache: &EvalCacheTable,
         depth: i8) -> i16 {
    if depth <= 0 {
        let res = evaluator.eval(board.clone());
        let entry = EvalCache {
            board: board.clone(),
            lower: res,
            upper: res,
            depth: 0
        };
        cache.update(entry);
        res
    } else {
        let (lower, upper) = match cache.get(board.clone()) {
            Some(entry) => {
                if entry.depth >= depth {
                    (entry.lower, entry.upper)
                } else {
                    (-64 * SCALE, 64 * SCALE)
                }
            },
            None => (-64 * SCALE, 64 * SCALE)
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
        let res = think_impl(
            board.clone(), new_alpha, new_beta, passed, evaluator,
            cache, depth);
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
            depth
        };
        cache.update(entry);
        res
    }
}
