mod bits;
mod board;
mod eval;
mod search;

use std::io::prelude::*;
use std::mem;
use std::io::BufReader;
use std::fs::File;
use std::time::Instant;
use std::str::FromStr;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::search::*;

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

fn iddfs(board: Board, evaluator: & Evaluator) -> HashMap<Board, (i16, i16)> {
    let mut table_order = HashMap::<Board, (i16, i16)>::new();
    let rem = popcnt(board.empty());
    for cut in [16, 15, 14, 13, 12].iter() {
        let start2 = Instant::now();
        let mut table_cache = HashMap::<Board, (i16, i16)>::new();
        let tmp = think(board.clone(), -64 * scale, 64 * scale, false, & evaluator,
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
                let mut table_order = HashMap::<Board, (i16, i16)>::new();
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
