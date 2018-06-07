use packed_simd::*;

#[derive(Clone)]
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

use std::cmp::max;

fn solve_naive(board: Board, mut alpha: i8, beta: i8, passed: bool) -> i8 {
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
                res = max(res, -solve(next, -beta, -alpha, false));
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
            return -solve(board.pass(), -beta, -alpha, true);
        }
    }
    res
}

fn solve_fastest_first(board: Board, mut alpha: i8, beta: i8, passed: bool) -> i8 {
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
            res = max(res, -solve(next.clone(), -beta, -alpha, false));
        } else {
            let mut result = -solve(next.clone(), -alpha-1, -alpha, false);
            if result >= beta {
                return result;
            }
            if result > alpha {
                alpha = result;
                result = -solve(next.clone(), -beta, -alpha, false);
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
            return -solve(board.pass(), -beta, -alpha, true);
        }
    }
    res
}

fn solve(board: Board, alpha: i8, beta: i8, passed: bool) -> i8 {
    if popcnt(board.empty()) <= 6 {
        solve_naive(board, alpha, beta, passed)
    } else {
        solve_fastest_first(board, alpha, beta, passed)
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

fn solve_ffo(name: &str, begin_index: usize) -> () {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        match Board::from_str(&line.unwrap()) {
            Ok(board) => {
                let start = Instant::now();
                let res = solve(board, -64, 64, false);
                let end = start.elapsed();
                println!("number: {}, result: {}, time: {}.{:03}sec",
                         i+begin_index, res, end.as_secs(),
                         end.subsec_nanos() / 1_000_000);
            },
            Err(_) => println!("Parse error")
        }

    }
}

fn main() {
    solve_ffo("fforum-1-19.obf", 1);
    solve_ffo("fforum-20-39.obf", 20);
    solve_ffo("fforum-40-59.obf", 40);
    solve_ffo("fforum-60-79.obf", 60);
}
