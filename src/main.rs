mod book;
mod engine;
mod play;
mod playout;
mod serialize;
mod sparse_mat;
mod train;

use crate::book::*;
use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::last_flip_cache::*;
use crate::engine::remote::*;
use crate::engine::search::*;
use crate::engine::table::*;
use crate::play::*;
use crate::serialize::*;
use crate::train::*;
use clap::{Arg, ArgMatches, Command};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::str;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

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

fn hand_to_string(hand: Hand) -> String {
    match hand {
        Hand::Pass => "ps".to_string(),
        Hand::Play(hand) => {
            let row = hand as u8 / 8;
            let col = hand as u8 % 8;
            let row_char = b'1' + row;
            let col_char = b'A' + col;
            let s = [col_char, row_char];
            str::from_utf8(&s).unwrap().to_string()
        }
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
    last_flip_cache: Arc<LastFlipCache>,
) -> Vec<Stat> {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    println!("|No.|empties|result|answer|move|nodes|time|NPS|");
    println!("|---:|---:|---:|---:|---:|---:|:--:|---:|");
    let mut stats = Vec::new();
    for line in reader.lines() {
        let line_str = line.unwrap();
        let desired: i8 = line_str[71..].split(';').next().unwrap().parse().unwrap();
        match Board::from_str(&line_str) {
            Ok(board) => {
                let rem = popcnt(board.empty());
                let start = Instant::now();
                let mut obj = SolveObj::new(
                    res_cache.clone(),
                    eval_cache.clone(),
                    evaluator.clone(),
                    last_flip_cache.clone(),
                    search_params.clone(),
                );
                let (res, best, stat) = solve(
                    &mut obj,
                    board,
                    -(BOARD_SIZE as i8),
                    BOARD_SIZE as i8,
                    false,
                    0,
                );
                let end = start.elapsed();
                let milli_seconds = end.as_millis() + 1; // ceil up, avoid zero-division
                let nodes_per_sec = (stat.node_count * 1000) as u128 / milli_seconds;
                println!(
                    "|{:2}|{:2}|{:+3}|{:+3}|{}|{:>5}|{:4}.{:03}s|{}M/s|",
                    index,
                    rem,
                    res,
                    desired,
                    best.map_or("XX".to_string(), hand_to_string),
                    to_si(stat.node_count),
                    end.as_secs(),
                    end.subsec_millis(),
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

fn report_stats(stats: &[Stat]) {
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
        ybwc_depth_limit: 12,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 18,
        eval_ordering_limit: 15,
        res_cache_limit: 11,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-220710"));
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let last_flip_cache = Arc::new(LastFlipCache::new());
    let mut index: usize = 1;
    let mut stats = Vec::new();
    //stats.extend(solve_ffo(
    //    "problem/hard-20.obf",
    //    &mut index,
    //    &search_params,
    //    evaluator.clone(),
    //    &mut res_cache,
    //    &mut eval_cache,
    //));
    //stats.extend(solve_ffo(
    //    "problem/hard-25.obf",
    //    &mut index,
    //    &search_params,
    //    evaluator.clone(),
    //    &mut res_cache,
    //    &mut eval_cache,
    //));
    stats.extend(solve_ffo(
        "problem/fforum-1-19.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        last_flip_cache.clone(),
    ));
    stats.extend(solve_ffo(
        "problem/fforum-20-39.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        last_flip_cache.clone(),
    ));
    stats.extend(solve_ffo(
        "problem/fforum-40-59.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        last_flip_cache.clone(),
    ));
    stats.extend(solve_ffo(
        "problem/fforum-60-79.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        last_flip_cache.clone(),
    ));
    report_stats(&stats);
}

fn send_query(_matches: &ArgMatches) {
    //let name = "problem/stress_test_54_10k.b81r";
    //let file = File::open(name).unwrap();
    //let reader = BufReader::new(file);
    //let client: Client = surf::Config::new()
    //    .set_base_url(Url::parse("http://localhost:7733").unwrap())
    //    .try_into()
    //    .unwrap();
    //let mut futures = Vec::new();
    //for (_idx, line) in reader.lines().enumerate() {
    //    let client = client.clone();
    //    futures.push(
    //        tokio::task::spawn(async move {
    //            let line_str = line.unwrap();
    //            let desired: i8 = line_str[17..].parse().unwrap();
    //            match Board::from_base81(&line_str[..16]) {
    //                Ok(board) => {
    //                    let data = SolveRequest {
    //                        board: board.to_base81(),
    //                        alpha: -(BOARD_SIZE as i8),
    //                        beta: BOARD_SIZE as i8,
    //                    };
    //                    let data_str = serde_json::json!(data);
    //                    let solve_res: SolveResponse = client
    //                        .post("/")
    //                        .body(http_types::Body::from_json(&data_str).unwrap())
    //                        .recv_json()
    //                        .await
    //                        .unwrap();
    //                    let res = solve_res.result;
    //                    if res != desired {
    //                        board.print();
    //                    }
    //                    assert_eq!(res, desired);
    //                }
    //                Err(_) => {
    //                    panic!();
    //                }
    //            }
    //        })
    //        .unwrap(),
    //    );
    //}
    //executor::block_on(future::join_all(futures));
}

fn main() {
    let arg_input_file = Arg::new("INPUT")
        .short('i')
        .required(true)
        .takes_value(true);
    let arg_output_file = Arg::new("OUTPUT")
        .short('o')
        .required(true)
        .takes_value(true);
    let matches = Command::new("Issen-rs")
        .subcommand(Command::new("ffobench").about("Run FFO benchmark 1-79"))
        .subcommand(
            Command::new("play").about("Interactive play").arg(
                Arg::new("player")
                    .short('i')
                    .required(true)
                    .takes_value(true),
            ),
        )
        .subcommand(Command::new("self-play").about("Automatic self play"))
        .subcommand(
            Command::new("gen-record")
                .about("Generate record")
                .arg(
                    Arg::new("DEPTH")
                        .short('d')
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::new("COUNT")
                        .short('n')
                        .required(true)
                        .takes_value(true),
                )
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("clean-record")
                .about("Cleaning record")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("update-record")
                .about("Update record by end-game search")
                .arg(arg_input_file.clone())
                .arg(
                    Arg::new("DEPTH")
                        .short('d')
                        .required(true)
                        .takes_value(true),
                )
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("minimax-record")
                .about("Minimax Record")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("gen-dataset")
                .about("Generate training dataset")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone())
                .arg(
                    Arg::new("MAX_OUT")
                        .short('n')
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            Command::new("train")
                .about("Train weights")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone())
                .arg(Arg::new("from").required(true).takes_value(true))
                .arg(Arg::new("to").required(true).takes_value(true))
                .arg(Arg::new("width").required(true).takes_value(true)),
        )
        .subcommand(
            Command::new("update-record-v2")
                .about("Update record iterative")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("gen-book")
                .about("Generate book")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone())
                .arg(
                    Arg::new("MAX_COUNT")
                        .short('c')
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            Command::new("binarize")
                .about("Binarize weights file")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("pack")
                .about("Pack weights file")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("pack-book")
                .about("Pack book file")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("last-table")
                .about("Generate last move flip count table")
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("last-mask")
                .about("Generate last move flip count mask")
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("parse-board")
                .about("Parse board")
                .arg(Arg::new("str").short('b').required(true).takes_value(true)),
        )
        .subcommand(
            Command::new("eval-stats")
                .about("Compute stats from dataset")
                .arg(arg_input_file.clone()),
        )
        .subcommand(Command::new("codingame").about("Codingame player"))
        .subcommand(Command::new("worker").about("worker mode"))
        .subcommand(Command::new("query").about("query mode"))
        .get_matches();
    match matches.subcommand() {
        Some(("ffobench", _matches)) => {
            ffo_benchmark();
        }
        Some(("play", matches)) => {
            play(matches);
        }
        Some(("self-play", matches)) => {
            self_play(matches);
        }
        Some(("gen-record", matches)) => {
            parallel_self_play(matches);
        }
        Some(("clean-record", matches)) => {
            clean_record(matches);
        }
        Some(("update-record", matches)) => {
            update_record(matches);
        }
        Some(("update-record-v2", matches)) => {
            iterative_update_book(matches);
        }
        Some(("minimax-record", matches)) => {
            minimax_record(matches);
        }
        Some(("gen-dataset", matches)) => {
            gen_dataset(matches);
        }
        Some(("train", matches)) => {
            train(matches);
        }
        Some(("gen-book", matches)) => {
            gen_book(matches);
        }
        Some(("binarize", matches)) => {
            binarize_weights(matches);
        }
        Some(("pack", matches)) => {
            pack_weights(matches);
        }
        Some(("pack-book", matches)) => {
            pack_book(matches);
        }
        Some(("last-table", matches)) => {
            gen_last_table(matches);
        }
        Some(("last-mask", matches)) => {
            gen_last_mask(matches);
        }
        Some(("parse-board", matches)) => {
            parse_board(matches);
        }
        Some(("eval-stats", matches)) => {
            eval_stats(matches);
        }
        Some(("codingame", matches)) => {
            codingame(matches).unwrap();
        }
        Some(("worker", matches)) => {
            worker(matches);
        }
        Some(("query", matches)) => {
            send_query(matches);
        }
        Some(_) => {
            eprintln!("Unknown subcommand");
        }
        None => {
            eprintln!("Need subcommand");
        }
    }
}
