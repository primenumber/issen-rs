use packed_simd::*;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Board {
    player: u64,
    opponent: u64,
    is_black: bool
}

fn upper_bit(mut x: u64x4) -> u64x4 {
    x = x | (x >>  1);
    x = x | (x >>  2);
    x = x | (x >>  4);
    x = x | (x >>  8);
    x = x | (x >> 16);
    x = x | (x >> 32);
    let lowers: u64x4 = x >> 1;
    x & !lowers
}

fn nonzero(x: u64x4) -> u64x4 {
    let zero = u64x4::new(0, 0, 0, 0);
    let mask = x.ne(zero);
    let one = u64x4::new(1, 1, 1, 1);
    one & u64x4::from_cast(mask)
}

fn popcnt(x: u64) -> i8 {
    x.count_ones() as i8
}

fn flip_vertical(mut x: u64) -> u64 {
    x = (x >> 32) | (x << 32);
    x = ((x >> 16) & 0x0000FFFF0000FFFFu64) | ((x << 16) & 0xFFFF0000FFFF0000u64);
    x = ((x >>  8) & 0x00FF00FF00FF00FFu64) | ((x <<  8) & 0xFF00FF00FF00FF00u64);
    x
}

fn delta_swap(x: u64, mask: u64, delta: isize) -> u64 {
    let tmp = mask & (x ^ (x << delta));
    x ^ tmp ^ (tmp >> delta)
}

fn flip_diag(mut x: u64) -> u64 {
    x = delta_swap(x, 0x0f0f0f0f00000000u64, 28);
    x = delta_swap(x, 0x3333000033330000u64, 14);
    x = delta_swap(x, 0x5500550055005500u64, 7);
    x
}

fn rot90(x: u64) -> u64 {
    flip_vertical(flip_diag(x))
}

use std::io::Write;

pub struct UnmovableError;

impl Board {
    fn flip_simd(&self, pos: usize) -> u64x4 {
        let p = u64x4::new(self.player, self.player, self.player, self.player);
        let o = u64x4::new(self.opponent, self.opponent, self.opponent, self.opponent);
        let omask = u64x4::new(0xFFFFFFFFFFFFFFFFu64,
                               0x7E7E7E7E7E7E7E7Eu64,
                               0x7E7E7E7E7E7E7E7Eu64,
                               0x7E7E7E7E7E7E7E7Eu64);
        let om = o & omask;
        let mask1 = u64x4::new(
            0x0080808080808080u64,
            0x7f00000000000000u64,
            0x0102040810204000u64,
            0x0040201008040201u64
        );
        let mut mask = mask1 >> (63 - pos) as u32;
        let mut outflank = upper_bit(!om & mask) & p;
        let mut flipped = u64x4::from_cast(-i64x4::from_cast(outflank) << 1) & mask;
        let mask2 = u64x4::new(
            0x0101010101010100u64,
            0x00000000000000feu64,
            0x0002040810204080u64,
            0x8040201008040200u64
        );
        mask = mask2 << pos as u32;
        outflank = mask & ((om | !mask) + 1) & p;
        flipped |= (outflank - nonzero(outflank)) & mask;
        return flipped;
    }

    pub fn flip(&self, pos: usize) -> u64 {
        let flips = self.flip_simd(pos);
        flips.or()
    }

    pub fn is_movable(&self, pos: usize) -> bool {
        if pos >= 64 {
            return false;
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return false;
        }
        self.flip(pos) != 0
    }

    pub fn play(&self, pos: usize) -> Result<Board, UnmovableError> {
        if pos >= 64 {
            return Err(UnmovableError{});
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return Err(UnmovableError{});
        }
        let flip_bits = self.flip(pos);
        if flip_bits == 0 {
            return Err(UnmovableError{});
        }
        Ok(Board {
            player: self.opponent ^ flip_bits,
            opponent: (self.player ^ flip_bits) | (1u64 << pos),
            is_black: !self.is_black
        })
    }

    pub fn pass(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
            is_black: !self.is_black
        }
    }

    pub fn empty(&self) -> u64 {
        !(self.player | self.opponent)
    }

    pub fn mobility(&self) -> Vec<usize> {
        let mut result = Vec::new();
        let mut empties = self.empty();
        while empties != 0 {
            let bit = empties  & empties.wrapping_neg();
            empties = empties & (empties - 1);
            let pos = popcnt(bit - 1) as usize;
            if self.is_movable(pos) {
                result.push(pos);
            }
        }
        result
    }

    pub fn is_gameover(&self) -> bool {
        if !self.mobility().is_empty() {
            return false;
        }
        self.pass().mobility().is_empty()
    }

    pub fn print(&self) -> () {
        for i in 0..64 {
            if ((self.player >> i) & 1) != 0 {
                if self.is_black {
                    write!(std::io::stdout(), "X").unwrap();
                } else {
                    write!(std::io::stdout(), "O").unwrap();
                }
            } else if ((self.opponent >> i) & 1) != 0 {
                if self.is_black {
                    write!(std::io::stdout(), "O").unwrap();
                } else {
                    write!(std::io::stdout(), "X").unwrap();
                }
            } else {
                write!(std::io::stdout(), "-").unwrap();
            }
            if i % 8 == 7 {
                write!(std::io::stdout(), "\n").unwrap();
            }
        }
    }

    pub fn score(&self) -> i8 {
        let pcnt = popcnt(self.player);
        let ocnt = popcnt(self.opponent);
        if pcnt == ocnt {
            0
        } else if pcnt > ocnt {
            64 - 2*ocnt
        } else {
            -64 + 2*pcnt
        }
    }

    pub fn flip_diag(&self) -> Board {
        Board {
            player: flip_diag(self.player),
            opponent: flip_diag(self.opponent),
            is_black: self.is_black
        }
    }

    pub fn rot90(&self) -> Board {
        Board {
            player: rot90(self.player),
            opponent: rot90(self.opponent),
            is_black: self.is_black
        }
    }
}

use std::str::FromStr;

pub struct BoardParseError;

impl FromStr for Board {
    type Err = BoardParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() <= 66 {
            return Err(BoardParseError{});
        }
        let mut black = 0u64;
        let mut white = 0u64;
        for (i, c) in s.chars().take(64).enumerate() {
            match c {
                'X' => black |= 1u64 << i,
                'O' => white |= 1u64 << i,
                _ => ()
            }
        }
        if s.chars().nth(65) == Some('X') {
            Ok(Board{ player: black, opponent: white, is_black: true })
        } else if s.chars().nth(65) == Some('O') {
            Ok(Board{ player: white, opponent: black, is_black: false })
        } else {
            Err(BoardParseError{})
        }
    }
}

use std::cmp::{min, max};
use std::collections::HashMap;

struct Evaluator {
    weights: Vec<Vec<f64>>,
    offsets: Vec<usize>,
    patterns: Vec<u64>,
    base3: Vec<usize>
}

fn pow3(x: i8) -> usize {
    if x == 0 {
        1
    } else {
        3 * pow3(x-1)
    }
}

fn mirror_under_8(mut x: u64) -> u64 {
    x = (x >> 4) | ((x << 4) & 0xF0);
    x = ((x >> 2) & 0x33) | ((x << 2) & 0xCC);
    x = ((x >> 1) & 0x55) | ((x << 1) & 0xAA);
    x
}

use bitintr::Pext;

fn pext(x: u64, mask: u64) -> u64 {
    x.pext(mask)
}

use std::mem;

impl Evaluator {
    pub fn new(subboard_filename: &str) -> Evaluator {
        let subboard_file = File::open(subboard_filename).unwrap();
        let subboard_reader = BufReader::new(subboard_file);
        let mut patterns = Vec::new();
        let mut offsets = Vec::new();
        let mut pattern_bits = 0;
        let mut length: usize = 0;
        let mut max_bits = 0;
        for (i, line) in subboard_reader.lines().enumerate() {
            if i == 0 {
                let _count = usize::from_str(&line.unwrap()).unwrap();
            } else if (i % 9) != 0 {
                let lane = (i % 9) - 1;
                let bits = u64::from_str_radix(&line.unwrap(), 2).unwrap();
                pattern_bits |= mirror_under_8(bits) << (lane * 8);
                if (i % 9) == 8 {
                    patterns.push(pattern_bits);
                    offsets.push(length);
                    length += pow3(popcnt(pattern_bits));
                    max_bits = max(max_bits, popcnt(pattern_bits));
                }
            } else {
                pattern_bits = 0;
            }
        }
        length += 1;

        let files: Vec<String> = (48..61).map(|i| format!("value{}", i)).collect();
        let mut weights = vec![vec![0f64; length]; files.len()];
        for (cnt, filename) in files.iter().enumerate() {
            let mut value_file = File::open(filename).unwrap();
            let mut buf = vec![0u8; length * 8];
            value_file.read(&mut buf).unwrap();
            for i in 0usize..length {
                let mut ary: [u8; 8] = Default::default();
                ary.copy_from_slice(&buf[(8*i)..(8*(i+1))]);
                weights[cnt][i] = unsafe { mem::transmute::<[u8; 8], f64>(ary) };
            }
        }

        let mut base3 = vec![0; 1<<max_bits];
        for i in 0usize..(1usize<<max_bits) {
            let mut sum = 0;
            for j in 0..max_bits {
                if ((i >> j) & 1) != 0 {
                    sum += pow3(j);
                }
            }
            base3[i] = sum;
        }
        Evaluator { weights, offsets, patterns, base3 }
    }

    fn eval_impl(&self, board: Board) -> f64 {
        let mut score = 0f64;
        let rem:usize = popcnt(board.empty()) as usize;
        for (i, pattern) in self.patterns.iter().enumerate() {
            let player_pattern = pext(board.player, *pattern) as usize;
            let opponent_pattern = pext(board.opponent, *pattern) as usize;
            score += self.weights[16 - rem][
                self.offsets[i] + self.base3[player_pattern]
                + 2*self.base3[opponent_pattern]];
        }
        score
    }

    pub fn eval(&self, mut board: Board) -> f64 {
        let mut score = 0f64;
        let rem:usize = popcnt(board.empty()) as usize;
        for _i in 0..4 {
            score += self.eval_impl(board.clone());
            score += self.eval_impl(board.flip_diag());
            board = board.rot90();
        }
        let raw_score = score + *self.weights[16 - rem].last().unwrap();
        if raw_score > 63.0 {
            64.0 - 1.0 / (raw_score - 62.0)
        } else if raw_score < -63.0 {
            -64.0 - 1.0 / (raw_score + 62.0)
        } else {
            raw_score
        }
    }
}

use std::sync::{Arc, Mutex};
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
               table_order: & HashMap<Board, (f64, f64)>, count: &mut usize,
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
                       table_order: & HashMap<Board, (f64, f64)>,
                       count: &mut usize, depth: u8) -> i8 {
    let mut v = vec![(0usize, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                v.push((next.mobility().len(), next));
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
    table_order: & HashMap<Board, (f64, f64)>, count: &mut usize,
    depth: u8) -> i8 {
    let mut v = vec![(0f64, 0f64, board.clone()); 0];
    let mut w = vec![(0usize, board.clone()); 0];
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
                        w.push((next.mobility().len(), next));
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
    table_order: & HashMap<Board, (f64, f64)>, count: &mut usize,
    depth: u8) -> i8 {
    let mut v = vec![(0f64, 0f64, board.clone()); 0];
    let mut w = vec![(0usize, board.clone()); 0];
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
                        w.push((next.mobility().len(), next));
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
                            next.clone(), -alpha-1, -alpha, false, &mut table, table_order, &mut count, depth+1));
                    if res >= beta {
                        let _ = tx.send(res);
                        let _ = txcount.send(count);
                        return;
                    }
                    if res > alpha {
                        alpha = res;
                        res = max(res, -solve(
                                next.clone(), -beta, -alpha, false, &mut table, table_order, &mut count, depth+1));
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
                    table_order: & HashMap<Board, (f64, f64)>, count: &mut usize,
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
    let res = if depth >= 3 {
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

fn solve(board: Board, alpha: i8, beta: i8, passed: bool,
         table: &mut Table<i8>,
         table_order: & HashMap<Board, (f64, f64)>, count: &mut usize,
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

fn think_impl(board: Board, mut alpha: f64, beta: f64, passed: bool,
         evaluator: & Evaluator,
         table_cache: &mut HashMap<Board, (f64, f64)>,
         table_order: & HashMap<Board, (f64, f64)>, depth: u8) -> f64 {
    let mut v = vec![(0f64, 0f64, 0usize, board.clone()); 0];
    let mut w = vec![(0usize, board.clone()); 0];
    let mut empties = board.empty();
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                match table_order.get(&next) {
                    Some(&(lower, upper)) => {
                        v.push((upper, lower, next.mobility().len(), next));
                    },
                    None => {
                        w.push((next.mobility().len(), next));
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
    let mut res = -std::f64::INFINITY;
    for (i, next) in nexts.iter().enumerate() {
        if i == 0 {
            res = res.max(-think(
                    next.clone(), -beta, -alpha, false, evaluator,
                    table_cache, table_order, depth-1));
        } else {
            res = res.max(-think(
                    next.clone(), -alpha-1.0, -alpha, false, evaluator,
                    table_cache, table_order, depth-1));
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
            return board.score() as f64;
        } else {
            return -think(
                board.pass(), -beta, -alpha, true, evaluator,
                table_cache, table_order, depth);
        }
    }
    res
}

fn think(board: Board, alpha: f64, beta: f64, passed: bool,
         evaluator: & Evaluator,
         table_cache: &mut HashMap<Board, (f64, f64)>,
         table_order: & HashMap<Board, (f64, f64)>, depth: u8) -> f64 {
    if depth == 0 {
        let res = evaluator.eval(board.clone());
        table_cache.insert(board.clone(), (res, res));
        res
    } else {
        let (lower, upper) = match table_cache.get(&board) {
            Some(&(lower, upper)) => (lower, upper),
            None => (-std::f64::INFINITY, std::f64::INFINITY)
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
                    (-std::f64::INFINITY, res)
                } else if res >= new_beta {
                    (res, std::f64::INFINITY)
                } else {
                    (res, res)
                }
            }
        };
        table_cache.insert(board.clone(), range);
        res
    }
}

use std::io::prelude::*;

pub struct HandParseError;

fn read_hand() -> Option<usize> {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();
    if s.len() < 2 {
        return None;
    }
    if s == "ps" {
        return Some(64);
    }
    let column_code = s.chars().nth(0).unwrap() as usize;
    if column_code < 'a' as usize || ('h' as usize) < column_code {
        return None;
    }
    let row_code = s.chars().nth(1).unwrap() as usize;
    if row_code < '1' as usize || ('8' as usize) < row_code {
        return None;
    }
    Some((row_code - '1' as usize) * 8 + (column_code - 'a' as usize))
}

fn play(mut board: Board) -> Board {
    while !board.is_gameover() {
        board.print();
        println!("Input move");
        let hand: usize;
        loop {
            match read_hand() {
                Some(h) => {
                    hand = h;
                    break;
                },
                None => ()
            }
        }
        if hand == 64 {
            board = board.pass();
        } else {
            match board.play(hand) {
                Ok(next) => board = next,
                Err(_) => println!("Invalid move")
            }
        }
    }
    board
}

use std::io::BufReader;
use std::fs::File;
use std::time::Instant;

fn iddfs(board: Board, evaluator: & Evaluator) -> HashMap<Board, (f64, f64)> {
    let mut table_order = HashMap::<Board, (f64, f64)>::new();
    let rem = popcnt(board.empty()) as u8;
    for cut in [16, 15, 14, 13, 12].iter() {
        let start2 = Instant::now();
        let mut table_cache = HashMap::<Board, (f64, f64)>::new();
        let tmp = think(board.clone(), -64.0, 64.0, false, & evaluator,
        &mut table_cache, &table_order, rem - cut);
        let end2 = start2.elapsed();
        println!("think: {}, nodes: {}nodes, time: {}.{:03}sec",
                 tmp, table_cache.len(),
                 end2.as_secs(), end2.subsec_nanos() / 1_000_000);
        mem::swap(&mut table_order, &mut table_cache);
    }
    table_order
}

fn solve_ffo(name: &str, begin_index: usize, evaluator: & Evaluator) -> () {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        match Board::from_str(&line.unwrap()) {
            Ok(board) => {
                let start = Instant::now();
                let mut table_order = HashMap::<Board, (f64, f64)>::new();
                let rem = popcnt(board.empty()) as u8;
                if rem > 16 {
                    table_order = iddfs(board.clone(), evaluator);
                }
                let mut table = Arc::new(Mutex::new(HashMap::<Board, (i8, i8)>::new()));
                let mut count = 0;
                let res = solve(
                    board, -64, 64, false, &mut table, &table_order, &mut count, 0);
                let end = start.elapsed();
                println!("number: {}, result: {}, count: {}, time: {}.{:03}sec",
                         i+begin_index, res, count, end.as_secs(),
                         end.subsec_nanos() / 1_000_000);
            },
            Err(_) => println!("Parse error")
        }

    }
}

fn main() {
    let evaluator = Evaluator::new("subboard.txt");
    solve_ffo("fforum-1-19.obf", 1, & evaluator);
    solve_ffo("fforum-20-39.obf", 20, & evaluator);
    solve_ffo("fforum-40-59.obf", 40, & evaluator);
    solve_ffo("fforum-60-79.obf", 60, & evaluator);
}
