#[derive(Clone)]
pub struct Board {
    player: u64,
    opponent: u64,
    is_black: bool
}

fn upper_bit(x: u64) -> u64 {
    let mut y = x | (x >> 1);
    y = y | (y >>  2);
    y = y | (y >>  4);
    y = y | (y >>  8);
    y = y | (y >> 16);
    y = y | (y >> 32);
    return y & !(y >> 1);
}

fn nonzero(x: u64) -> u64 {
    if x != 0 {
        1
    } else {
        0
    }
}

fn popcnt(x: u64) -> i8 {
    let mut y = x;
    y = (y & 0x5555555555555555u64) + ((y & 0xAAAAAAAAAAAAAAAAu64) >>  1);
    y = (y & 0x3333333333333333u64) + ((y & 0xCCCCCCCCCCCCCCCCu64) >>  2);
    y = (y & 0x0F0F0F0F0F0F0F0Fu64) + ((y & 0xF0F0F0F0F0F0F0F0u64) >>  4);
    y = (y & 0x00FF00FF00FF00FFu64) + ((y & 0xFF00FF00FF00FF00u64) >>  8);
    y = (y & 0x0000FFFF0000FFFFu64) + ((y & 0xFFFF0000FFFF0000u64) >> 16);
    y = (y & 0x00000000FFFFFFFFu64) + ((y & 0xFFFFFFFF00000000u64) >> 32);
    y as i8
}

use std::io::Write;

pub struct UnmovableError;

impl Board {
    fn flip_impl(&self, pos: usize, simd_index: usize) -> u64 {
        let mut om = self.opponent;
        if simd_index != 0 {
            om &= 0x7E7E7E7E7E7E7E7Eu64;
        }
        const MASK1: [u64; 4] = [
            0x0080808080808080u64,
            0x7f00000000000000u64,
            0x0102040810204000u64,
            0x0040201008040201u64
        ];
        let mut mask = MASK1[simd_index] >> (63 - pos);
        let mut outflank = upper_bit(!om & mask) & self.player;
        let mut flipped = ((-(outflank as i64) << 1) as u64) & mask;
        const MASK2: [u64; 4] = [
            0x0101010101010100u64,
            0x00000000000000feu64,
            0x0002040810204080u64,
            0x8040201008040200u64
        ];
        mask = MASK2[simd_index] << pos;
        outflank = mask & ((om | !mask) + 1) & self.player;
        flipped |= (outflank - nonzero(outflank)) & mask;
        return flipped;
    }

    pub fn flip(&self, pos: usize) -> u64 {
        let mut res = 0;
        for i in 0usize..4usize {
            res |= self.flip_impl(pos, i);
        }
        res
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
    while empties != 0 {
        let bit = empties  & empties.wrapping_neg();
        empties = empties & (empties - 1);
        let pos = popcnt(bit - 1) as usize;
        match board.play(pos) {
            Ok(next) => {
                pass = false;
                alpha = max(alpha, -solve(next, -beta, -alpha, false));
                if alpha >= beta {
                    return alpha;
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
    alpha
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
    for &(_, ref next) in &v {
        alpha = max(alpha, -solve(next.clone(), -beta, -alpha, false));
        if alpha >= beta {
            return alpha;
        }
    }
    if v.is_empty() {
        if passed {
            return board.score();
        } else {
            return -solve(board.pass(), -beta, -alpha, true);
        }
    }
    alpha
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
