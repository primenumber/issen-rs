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
use std::sync::Arc;
use futures::executor::ThreadPool;
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
        1 => format!("{}.{:02}{}", x3 / 100, x3 % 100, prefix),
        2 => format!("{}.{}{}", x3 / 10, x3 % 10, prefix),
        _ => panic!()
    }
}

struct Stat {
    nodes: usize,
    elapsed: f64,
    correct: bool
}

fn solve_ffo(name: &str, index: &mut usize,
             search_params: &SearchParams, evaluator: Arc<Evaluator>,
             res_cache: &mut ResCacheTable, eval_cache: &mut EvalCacheTable,
             pool: &ThreadPool) -> Vec<Stat> {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    println!("|No.|empties|result|answer|nodes|time|NPS|");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    let mut stats = Vec::new();
    for line in reader.lines() {
        let line_str = line.unwrap();
        let desired = line_str[71..].split(';').next().unwrap().parse().unwrap();
        match Board::from_str(&line_str) {
            Ok(board) => {
                let rem = popcnt(board.empty());
                let start = Instant::now();
                let mut obj = SolveObj::new(
                    res_cache.clone(), eval_cache.clone(), evaluator.clone(), search_params.clone(), pool.clone());
                let (res, stat) = solve(
                    &mut obj, board, -64, 64, false, 0);
                let end = start.elapsed();
                let milli_seconds = end.as_millis() + 1;  // ceil up, avoid zero-division
                let nodes_per_sec = (stat.node_count * 1000) as u128 / milli_seconds;
                println!("|{:2}|{:2}|{:+3}|{:+3}|{:>5}|{:4}.{:03}s|{}M/s|",
                         index, rem, res, desired,
                         to_si(stat.node_count),
                         end.as_secs(),
                         end.subsec_nanos() / 1_000_000,
                         nodes_per_sec / 1_000_000);
                eval_cache.inc_gen();
                res_cache.inc_gen();
                stats.push(Stat {
                    nodes: stat.node_count,
                    elapsed: end.as_secs_f64(), 
                    correct: res == desired
                });
                *index += 1;
            },
            Err(_) => println!("Parse error")
        }

    }
    stats
}

fn report_stats(stats: &[Stat]) -> () {
    let mut wrongs = 0;
    let mut nodes_sum = 0;
    let mut elapsed_sum = 0.0;
    for stat in stats {
        if !stat.correct {
            wrongs += 1;
        }
        nodes_sum += stat.nodes;
        elapsed_sum += stat.elapsed;
    }
    let nodes_per_sec = (nodes_sum as f64 / elapsed_sum) as usize;
    println!("Wrongs: {}, Nodes: {}, Elapsed: {:.3}, NPS: {}", wrongs, to_si(nodes_sum), elapsed_sum, to_si(nodes_per_sec));
}

fn main() {
    let search_params = SearchParams {
        reduce: false,
        ybwc_depth_limit: 10,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 17,
        eval_ordering_limit: 16,
        res_cache_limit: 11,
        stability_cut_limit: 12,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3
    };
    let evaluator = Arc::new(Evaluator::new("subboard.txt"));
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let mut index: usize = 1;
    let mut stats = Vec::new();
    //stats.extend(solve_ffo("problem/hard-20.obf",      &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    //stats.extend(solve_ffo("problem/hard-25.obf",      &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    //stats.extend(solve_ffo("problem/fforum-1-19.obf",  &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    //stats.extend(solve_ffo("problem/fforum-20-39.obf", &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    //stats.extend(solve_ffo("problem/fforum-40-59.obf", &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    //stats.extend(solve_ffo("problem/fforum-60-79.obf", &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    report_stats(&stats);
}
