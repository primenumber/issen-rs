mod bits;
mod board;
mod eval;
mod table;
mod search;

#[macro_use]
extern crate lazy_static;

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

fn to_si(x: usize) -> String {
    if x == 0 {
        return "0".to_string();
    }
    let mut digits = 0;
    let mut y = x;
    while y > 0 {
        digits += 1;
        y /= 10;
    }
    if digits <= 3 {
        return format!("{}", x);
    }
    let mut x3 = x;
    for _ in 3..digits {
        x3 /= 10;
    }
    const SI: &str = " kMGTP";
    let base1000_digits = (digits - 1) / 3;
    let prefix = SI.chars().nth(base1000_digits).unwrap();
    match digits % 3 {
        0 => format!("{}{}", x3, prefix),
        1 => format!("{}.{}{}", x3 / 100, x3 % 100, prefix),
        2 => format!("{}.{}{}", x3 / 10, x3 % 10, prefix),
        _ => panic!()
    }
}

fn solve_ffo(name: &str, begin_index: usize, evaluator: &Evaluator, eval_cache: &EvalCacheTable) -> () {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        match Board::from_str(&line.unwrap()) {
            Ok(board) => {
                let rem = popcnt(board.empty());
                let start = Instant::now();
                let res_cache = Arc::new(Mutex::new(HashMap::<Board, (i8, i8, u8)>::new()));
                let obj = SolveObj::new(
                    res_cache, eval_cache.clone(), evaluator);
                let res = obj.solve(
                    board, -64, 64, false, 0);
                let end = start.elapsed();
                println!("n: {}, rem: {}, res: {}, cnt: {}s/{}c/{}g/{}u/{}h, t: {}.{:03}s",
                         i+begin_index, rem, res,
                         to_si(obj.count.get()),
                         to_si(obj.st_cut.get()),
                         to_si(obj.eval_cache.cnt_get.get()),
                         to_si(obj.eval_cache.cnt_update.get()),
                         to_si(obj.eval_cache.cnt_hit.get()),
                         end.as_secs(),
                         end.subsec_nanos() / 1_000_000);
                eval_cache.inc_gen();
            },
            Err(_) => println!("Parse error")
        }

    }
}

fn main() {
    let evaluator = Evaluator::new("subboard.txt");
    let eval_cache = EvalCacheTable::new(1024, 65536);
    solve_ffo("fforum-1-19.obf", 1, &evaluator, &eval_cache);
    solve_ffo("fforum-20-39.obf", 20, &evaluator, &eval_cache);
    solve_ffo("fforum-40-59.obf", 40, &evaluator, &eval_cache);
    solve_ffo("fforum-60-79.obf", 60, &evaluator, &eval_cache);
}
