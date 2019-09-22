use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::cmp::{min, max};
use crate::bits::*;
use crate::board::*;
use crate::eval::*;

type Table<T> = Arc<Mutex<HashMap<Board, (T, T)>>>;

fn solve_1(board: Board, count: &mut usize) -> i8 {
    let bit = board.empty();
    let pos = popcnt(bit - 1) as usize;
    match board.play(pos) {
        Ok(next) => -next.score(),
        Err(_) => {
            *count += 1;
            match board.pass().play(pos) {
                Ok(next) => next.score(),
                Err(_) => board.score()
            }
        }
    }
}

fn solve_naive(board: Board, mut alpha: i8, beta: i8, passed: bool,
               table: &mut Table<i8>,
               table_order: & HashMap<Board, (f32, f32)>, count: &mut usize,
               depth: u8)-> i8 {
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
                res = max(res, -solve(
                        next, -beta, -alpha, false, table, table_order, count, depth+1));
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
            return -solve(board.pass(), -beta, -alpha, true, table, table_order, count, depth);
        }
    }
    res
}

fn solve_fastest_first(board: Board, mut alpha: i8, beta: i8, passed: bool,
                       table: &mut Table<i8>,
                       table_order: & HashMap<Board, (f32, f32)>,
                       count: &mut usize, depth: u8) -> i8 {
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
            res = max(res, -solve(
                    next.clone(), -beta, -alpha, false, table, table_order, count, depth+1));
        } else {
            let mut result = -solve(
                next.clone(), -alpha-1, -alpha, false, table, table_order, count, depth+1);
            if result >= beta {
                return result;
            }
            if result > alpha {
                alpha = result;
                result = -solve(
                    next.clone(), -beta, -alpha, false, table, table_order, count, depth+1);
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
            return -solve(board.pass(), -beta, -alpha, true, table, table_order, count, depth);
        }
    }
    res
}

fn solve_move_ordering_with_table(
    board: Board, mut alpha: i8, beta: i8, passed: bool,
    table: &mut Table<i8>,
    table_order: & HashMap<Board, (f32, f32)>, count: &mut usize,
    depth: u8) -> i8 {
    let mut v = vec![(0f32, 0f32, board.clone()); 0];
    let mut w = vec![(0i8, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                match table_order.get(&next) {
                    Some(&(lower, upper)) => {
                        v.push((upper, lower, next));
                    },
                    None => {
                        w.push((weighted_mobility(& next), next));
                    }
                }
            },
            Err(_) => ()
        }
    }
    v.sort_by(|a, b| {
        if a.0 == b.0 {
            a.1.partial_cmp(&b.1).unwrap()
        } else {
            a.0.partial_cmp(&b.0).unwrap()
        }
    });
    w.sort_by(|a, b| a.0.cmp(&b.0));
    let mut vw = Vec::<Board>::new();
    for &(_, _, ref next) in &v {
        vw.push(next.clone());
    }
    for &(_, ref next) in &w {
        vw.push(next.clone());
    }
    let mut res = -64;
    for (i, next) in vw.iter().enumerate() {
        if i == 0 {
            res = -solve(next.clone(), -beta, -alpha, false, table, table_order, count, depth+1);
            alpha = max(alpha, res);
            if alpha >= beta {
                return res;
            }
        } else {
            res = max(res, -solve(
                    next.clone(), -alpha-1, -alpha, false, table, table_order, count, depth+1));
            if res >= beta {
                return res;
            }
            if res > alpha {
                alpha = res;
                res = max(res, -solve(
                        next.clone(), -beta, -alpha, false, table, table_order, count, depth+1));
            }
        }
    }
    if v.is_empty() && w.is_empty() {
        if passed {
            return board.score();
        } else {
            return -solve(board.pass(), -beta, -alpha, true, table, table_order, count, depth);
        }
    }
    res
}

use std::sync::mpsc;

fn solve_ybwc(
    board: Board, mut alpha: i8, beta: i8, passed: bool,
    table: &mut Table<i8>,
    table_order: & HashMap<Board, (f32, f32)>, count: &mut usize,
    depth: u8) -> i8 {
    let mut v = vec![(0f32, 0f32, board.clone()); 0];
    let mut w = vec![(0i8, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                match table_order.get(&next) {
                    Some(&(lower, upper)) => {
                        v.push((upper, lower, next));
                    },
                    None => {
                        w.push((weighted_mobility(& next), next));
                    }
                }
            },
            Err(_) => ()
        }
    }
    v.sort_by(|a, b| {
        if a.0 == b.0 {
            a.1.partial_cmp(&b.1).unwrap()
        } else {
            a.0.partial_cmp(&b.0).unwrap()
        }
    });
    w.sort_by(|a, b| a.0.cmp(&b.0));
    let mut vw = Vec::<Board>::new();
    for &(_, _, ref next) in &v {
        vw.push(next.clone());
    }
    for &(_, ref next) in &w {
        vw.push(next.clone());
    }
    let (tx, rx) = mpsc::channel();
    let (txcount, rxcount) = mpsc::channel();
    let res = crossbeam::scope(|scope| {
        let mut handles = Vec::new();
        let mut res = -64;
        for (i, next) in vw.iter().enumerate() {
            if i == 0 {
                res = -solve(next.clone(), -beta, -alpha, false, table, table_order, count, depth+1);
                alpha = max(alpha, res);
                if alpha >= beta {
                    return res;
                }
            } else {
                let tx = tx.clone();
                let txcount = txcount.clone();
                let mut table = table.clone();
                handles.push(scope.spawn(move |_| {
                    let mut count = 0;
                    res = max(res, -solve(
                            next.clone(), -alpha-1, -alpha, false, &mut table, table_order, &mut count, depth+2));
                    if res >= beta {
                        let _ = tx.send(res);
                        let _ = txcount.send(count);
                        return;
                    }
                    if res > alpha {
                        alpha = res;
                        res = max(res, -solve(
                                next.clone(), -beta, -alpha, false, &mut table, table_order, &mut count, depth+2));
                    }
                    let _ = tx.send(res);
                    let _ = txcount.send(count);
                }));
            }
        }
        for h in handles {
            let _ = h.join();
            *count += rxcount.recv().unwrap();
            res = max(res, rx.recv().unwrap());
        }
        alpha = max(alpha, res);
        res
    }).unwrap();
    if v.is_empty() && w.is_empty() {
        if passed {
            return board.score();
        } else {
            return -solve(board.pass(), -beta, -alpha, true, table, table_order, count, depth);
        }
    }
    res 
}

fn solve_with_table(board: Board, alpha: i8, beta: i8, passed: bool,
                    table: &mut Table<i8>,
                    table_order: & HashMap<Board, (f32, f32)>, count: &mut usize,
                    depth: u8) -> i8 {
    let (lower, upper) = match table.lock().unwrap().get(&board) {
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
        solve_move_ordering_with_table(
            board.clone(), alpha, beta, passed, table, table_order, count, depth)
    } else {
        solve_ybwc(
            board.clone(), alpha, beta, passed, table, table_order, count, depth)
    };
    let range = if res <= new_alpha {
        (lower, min(upper, res))
    } else if res >= new_beta {
        (max(lower, res), upper)
    } else {
        (res, res)
    };
    table.lock().unwrap().insert(board, range);
    res
}

pub fn solve(board: Board, alpha: i8, beta: i8, passed: bool,
         table: &mut Table<i8>,
         table_order: & HashMap<Board, (f32, f32)>, count: &mut usize,
         depth: u8) -> i8 {
    *count += 1;
    if popcnt(board.empty()) == 0 {
        board.score()
    } else if popcnt(board.empty()) == 1 {
        solve_1(board, count)
    } else if popcnt(board.empty()) <= 6 {
        solve_naive(board, alpha, beta, passed, table, table_order, count, depth)
    } else if popcnt(board.empty()) <= 12 {
        solve_fastest_first(board, alpha, beta, passed, table, table_order, count, depth)
    } else {
        solve_with_table(board, alpha, beta, passed, table, table_order, count, depth)
    }
}

fn think_impl(board: Board, mut alpha: f32, beta: f32, passed: bool,
         evaluator: & Evaluator,
         table_cache: &mut HashMap<Board, (f32, f32)>,
         table_order: & HashMap<Board, (f32, f32)>, depth: i8) -> f32 {
    let mut v = vec![(0f32, 0f32, 0i8, board.clone()); 0];
    let mut w = vec![(0i8, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                match table_order.get(&next) {
                    Some(&(lower, upper)) => {
                        v.push((upper, lower, weighted_mobility(& next), next));
                    },
                    None => {
                        w.push((weighted_mobility(& next), next));
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
                a.1.partial_cmp(&b.1).unwrap()
            }
        } else {
            a.0.partial_cmp(&b.0).unwrap()
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
    let mut res = -std::f32::INFINITY;
    for (i, next) in nexts.iter().enumerate() {
        if i == 0 {
            res = res.max(-think(
                    next.clone(), -beta, -alpha, false, evaluator,
                    table_cache, table_order, depth-1));
        } else {
            let reduce = if -evaluator.eval(next.clone()) < alpha - 16.0 {
                2
            } else {
                1
            };
            res = res.max(-think(
                    next.clone(), -alpha-0.001, -alpha, false, evaluator,
                    table_cache, table_order, depth-reduce));
            if res >= beta {
                return res;
            }
            if res > alpha {
                alpha = res;
                res = res.max(-think(
                        next.clone(), -beta, -alpha, false, evaluator,
                        table_cache, table_order, depth-1));
            }
        }
        alpha = alpha.max(res);
        if alpha >= beta {
            return res;
        }
    }
    if nexts.is_empty() {
        if passed {
            return board.score() as f32;
        } else {
            return -think(
                board.pass(), -beta, -alpha, true, evaluator,
                table_cache, table_order, depth);
        }
    }
    res
}

pub fn think(board: Board, alpha: f32, beta: f32, passed: bool,
         evaluator: & Evaluator,
         table_cache: &mut HashMap<Board, (f32, f32)>,
         table_order: & HashMap<Board, (f32, f32)>, depth: i8) -> f32 {
    if depth <= 0 {
        let res = evaluator.eval(board.clone());
        table_cache.insert(board.clone(), (res, res));
        res
    } else {
        let (lower, upper) = match table_cache.get(&board) {
            Some(&(lower, upper)) => (lower, upper),
            None => (-std::f32::INFINITY, std::f32::INFINITY)
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
            table_cache, table_order, depth);
        let range = match table_cache.get(&board) {
            Some(&(lower, upper)) => {
                if res <= new_alpha {
                    (lower, res.min(upper))
                } else if res >= new_beta {
                    (lower.max(res), upper)
                } else {
                    (res, res)
                }
            },
            None => {
                if res <= new_alpha {
                    (-std::f32::INFINITY, res)
                } else if res >= new_beta {
                    (res, std::f32::INFINITY)
                } else {
                    (res, res)
                }
            }
        };
        table_cache.insert(board.clone(), range);
        res
    }
}

