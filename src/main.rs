mod bits;
mod board;
mod eval;
mod table;
mod search;

use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::time::Instant;
use std::str::FromStr;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::table::*;
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

fn solve_ffo(name: &str, begin_index: usize, evaluator: &Evaluator) -> () {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        match Board::from_str(&line.unwrap()) {
            Ok(board) => {
                let rem = popcnt(board.empty());
                let start = Instant::now();
                let res_cache = Arc::new(Mutex::new(HashMap::<Board, (i8, i8)>::new()));
                let eval_cache = EvalCacheTable::new(256, 65536);
                let obj = SolveObj::new(
                    res_cache, eval_cache, evaluator);
                let res = obj.solve(
                    board, -64, 64, false, 0);
                let end = start.elapsed();
                println!("n: {}, rem: {}, res: {}, cnt: {}s/{}g/{}u/{}h, t: {}.{:03}s",
                         i+begin_index, rem, res,
                         obj.count.get(),
                         obj.eval_cache.cnt_get.get(),
                         obj.eval_cache.cnt_update.get(),
                         obj.eval_cache.cnt_hit.get(),
                         end.as_secs(),
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
