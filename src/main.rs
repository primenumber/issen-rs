mod bits;
mod board;
mod eval;
mod search;
mod serialize;
mod table;
mod train;

use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::search::*;
use crate::table::*;
use crate::train::*;
use clap::{App, Arg, ArgMatches, SubCommand};
use futures::executor;
use futures::executor::ThreadPool;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::str;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

pub struct HandParseError;

fn read_hand() -> Option<usize> {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();
    if s.len() < 2 {
        return None;
    }
    if &s[0..2] == "ps" {
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

fn play(matches: &ArgMatches) -> Board {
    let player_turn = matches.value_of("player").unwrap() == "B";

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
        static_ordering_limit: 3,
    };
    let evaluator = Arc::new(Evaluator::new("table"));
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();

    let mut board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    while !board.is_gameover() {
        board.print_with_sides();
        let hand = if board.is_black == player_turn {
            let hand: usize;
            loop {
                println!("Input move");
                match read_hand() {
                    Some(h) => {
                        hand = h;
                        break;
                    }
                    None => (),
                }
            }
            hand
        } else {
            println!("Thinking...");
            let best = if popcnt(board.empty()) > 22 {
                let (score, best, depth) = iterative_think(
                    board,
                    -64 * SCALE,
                    64 * SCALE,
                    false,
                    evaluator.clone(),
                    &mut eval_cache,
                    1000,
                );
                eprintln!("Estimated result: {}, Depth: {}", score, depth);
                best
            } else {
                let mut obj = SolveObj::new(
                    res_cache.clone(),
                    eval_cache.clone(),
                    evaluator.clone(),
                    search_params.clone(),
                    pool.clone(),
                );
                executor::block_on(solve_with_move(board, &mut obj))
            };
            eval_cache.inc_gen();
            res_cache.inc_gen();
            best
        };
        if hand == 64 {
            board = board.pass();
        } else {
            match board.play(hand) {
                Ok(next) => board = next,
                Err(_) => println!("Invalid move"),
            }
        }
    }
    println!("Game over");
    board.print_with_sides();
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
        _ => panic!(),
    }
}

fn hand_to_string(hand: u8) -> String {
    if hand == PASS as u8 {
        "ps".to_string()
    } else {
        let row = hand / 8;
        let col = hand % 8;
        let row_char = 0x30 + row;
        let col_char = 0x41 + col;
        let s = [col_char, row_char];
        str::from_utf8(&s).unwrap().to_string()
    }
}

struct Stat {
    nodes: usize,
    elapsed: f64,
    correct: bool,
}

fn solve_ffo(
    name: &str,
    index: &mut usize,
    search_params: &SearchParams,
    evaluator: Arc<Evaluator>,
    res_cache: &mut ResCacheTable,
    eval_cache: &mut EvalCacheTable,
    pool: &ThreadPool,
) -> Vec<Stat> {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    println!("|No.|empties|result|answer|move|nodes|time|NPS|");
    println!("|---:|---:|---:|---:|---:|---:|:--:|---:|");
    let mut stats = Vec::new();
    for line in reader.lines() {
        let line_str = line.unwrap();
        let desired = line_str[71..].split(';').next().unwrap().parse().unwrap();
        match Board::from_str(&line_str) {
            Ok(board) => {
                let rem = popcnt(board.empty());
                let start = Instant::now();
                let mut obj = SolveObj::new(
                    res_cache.clone(),
                    eval_cache.clone(),
                    evaluator.clone(),
                    search_params.clone(),
                    pool.clone(),
                );
                let (res, best, stat) = solve(&mut obj, board, -64, 64, false, 0);
                let end = start.elapsed();
                let milli_seconds = end.as_millis() + 1; // ceil up, avoid zero-division
                let nodes_per_sec = (stat.node_count * 1000) as u128 / milli_seconds;
                println!(
                    "|{:2}|{:2}|{:+3}|{:+3}|{}|{:>5}|{:4}.{:03}s|{}M/s|",
                    index,
                    rem,
                    res,
                    desired,
                    best.map_or("XX".to_string(), |h| hand_to_string(h)),
                    to_si(stat.node_count),
                    end.as_secs(),
                    end.subsec_nanos() / 1_000_000,
                    nodes_per_sec / 1_000_000
                );
                eval_cache.inc_gen();
                res_cache.inc_gen();
                stats.push(Stat {
                    nodes: stat.node_count,
                    elapsed: end.as_secs_f64(),
                    correct: res == desired,
                });
                *index += 1;
            }
            Err(_) => println!("Parse error"),
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
    println!(
        "Wrongs: {}, Nodes: {}, Elapsed: {:.3}, NPS: {}",
        wrongs,
        to_si(nodes_sum),
        elapsed_sum,
        to_si(nodes_per_sec)
    );
}

fn ffo_benchmark() {
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
        static_ordering_limit: 3,
    };
    let evaluator = Arc::new(Evaluator::new("table"));
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let mut index: usize = 1;
    let mut stats = Vec::new();
    //stats.extend(solve_ffo("problem/hard-20.obf",      &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    //stats.extend(solve_ffo("problem/hard-25.obf",      &mut index, &search_params, evaluator.clone(), &mut res_cache, &mut eval_cache, &pool));
    stats.extend(solve_ffo(
        "problem/fforum-1-19.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
    ));
    stats.extend(solve_ffo(
        "problem/fforum-20-39.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
    ));
    stats.extend(solve_ffo(
        "problem/fforum-40-59.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
    ));
    stats.extend(solve_ffo(
        "problem/fforum-60-79.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
    ));
    report_stats(&stats);
}

fn main() {
    let matches = App::new("Issen-rs")
        .subcommand(SubCommand::with_name("ffobench").about("Run FFO benchmark 1-79"))
        .subcommand(
            SubCommand::with_name("play").about("Interactive play").arg(
                Arg::with_name("player")
                    .short("i")
                    .required(true)
                    .takes_value(true),
            ),
        )
        .subcommand(
            SubCommand::with_name("clean-record")
                .about("Cleaning record")
                .arg(
                    Arg::with_name("INPUT")
                        .short("i")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("update-record")
                .about("Update record by end-game search")
                .arg(
                    Arg::with_name("INPUT")
                        .short("i")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("DEPTH")
                        .short("d")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("gen-dataset")
                .about("Generate training dataset")
                .arg(
                    Arg::with_name("INPUT")
                        .short("i")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                )
                .arg(Arg::with_name("MAX_OUT").short("n").takes_value(true)),
        )
        .subcommand(
            SubCommand::with_name("train")
                .about("Train weights")
                .arg(
                    Arg::with_name("INPUT")
                        .short("i")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("gen-book")
                .about("Generate book")
                .arg(
                    Arg::with_name("INPUT")
                        .short("i")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("MAX_COUNT")
                        .short("c")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("pack")
                .about("Pack weights file")
                .arg(
                    Arg::with_name("INPUT")
                        .short("i")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .get_matches();
    match matches.subcommand() {
        ("ffobench", Some(_matches)) => {
            ffo_benchmark();
        }
        ("play", Some(matches)) => {
            play(matches);
        }
        ("clean-record", Some(matches)) => {
            clean_record(matches);
        }
        ("update-record", Some(matches)) => {
            update_record(matches);
        }
        ("gen-dataset", Some(matches)) => {
            gen_dataset(matches);
        }
        ("train", Some(matches)) => {
            train(matches);
        }
        ("gen-book", Some(matches)) => {
            gen_book(matches);
        }
        ("pack", Some(matches)) => {
            pack_weights(matches);
        }
        ("", None) => {
            eprintln!("Need subcommand");
        }
        _ => unreachable!(),
    }
}
