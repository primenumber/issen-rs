#![feature(portable_simd)]
#![feature(iterator_try_collect)]
#![feature(test)]
mod book;
mod compression;
mod engine;
mod play;
mod record;
mod remote;
mod serializer;
mod setup;
mod sparse_mat;
mod train;

use crate::book::*;
use crate::compression::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::search::*;
use crate::play::*;
use crate::remote::*;
use crate::serializer::*;
use crate::setup::*;
use crate::train::*;
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use reqwest::Client;
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

struct Stat {
    nodes: usize,
    elapsed: f64,
    correct: bool,
}

fn solve_ffo<Eval: Evaluator>(
    name: &str,
    index: &mut usize,
    solve_obj: &mut SolveObj<Eval>,
    workers: &[String],
) -> Vec<Stat> {
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    let mut total_nodes = 0;
    println!("|No.|empties|result|answer|move|nodes|time|NPS|");
    println!("|---:|---:|---:|---:|---:|---:|:--:|---:|");
    let mut stats = Vec::new();
    let global_start = Instant::now();
    for line in reader.lines() {
        let line_str = line.unwrap();
        let desired: i8 = line_str[71..].split(';').next().unwrap().parse().unwrap();
        match BoardWithColor::from_str(&line_str) {
            Ok(board) => {
                let rem = board.empty().count_ones();
                let start = Instant::now();
                let (res, best, stat) = solve(
                    solve_obj,
                    workers,
                    board.board,
                    (-(BOARD_SIZE as i8), BOARD_SIZE as i8),
                    false,
                    0,
                    None,
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
                    best.map_or("XX".to_string(), |hand| format!("{}", hand)),
                    to_si(stat.node_count),
                    end.as_secs(),
                    end.subsec_millis(),
                    nodes_per_sec / 1_000_000
                );
                solve_obj.cache_generation += 1;
                stats.push(Stat {
                    nodes: stat.node_count,
                    elapsed: end.as_secs_f64(),
                    correct: res == desired,
                });
                *index += 1;
                total_nodes += stat.node_count;
            }
            Err(_) => println!("Parse error"),
        }
    }
    let micro_seconds = global_start.elapsed().as_micros();
    let nps = (total_nodes * 1000000) as u128 / micro_seconds;
    println!(
        "[Total] elapsed: {}us, node count: {}, NPS: {}nodes/sec",
        micro_seconds, total_nodes, nps,
    );

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

fn ffo_benchmark(matches: &ArgMatches) {
    let mut solve_obj = setup_default();
    let mut index: usize = 1;
    let mut stats = Vec::new();
    let worker_urls = match matches.get_many("workers") {
        Some(w) => w.cloned().collect::<Vec<String>>(),
        None => Vec::new(),
    };
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
        &mut solve_obj,
        &worker_urls,
    ));
    stats.extend(solve_ffo(
        "problem/fforum-20-39.obf",
        &mut index,
        &mut solve_obj,
        &worker_urls,
    ));
    stats.extend(solve_ffo(
        "problem/fforum-40-59.obf",
        &mut index,
        &mut solve_obj,
        &worker_urls,
    ));
    stats.extend(solve_ffo(
        "problem/fforum-60-79.obf",
        &mut index,
        &mut solve_obj,
        &worker_urls,
    ));
    report_stats(&stats);
}

async fn send_query_impl(board: Board, client: &Client) -> i8 {
    let data = engine::search::SolveRequest {
        board: board.to_base81(),
        alpha: -64,
        beta: 64,
    };
    let mut crc64 = crc64::Crc64::new();
    crc64.write(&board.player.to_le_bytes()).unwrap();
    crc64.write(&board.opponent.to_le_bytes()).unwrap();
    let data_json = serde_json::json!(data);
    let suffix = 192 + crc64.get() % 4;
    let uri = format!("http://192.168.10.{}:7733", suffix);
    let resp = match client.post(uri).json(&data_json).send().await {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("{:?}", e);
            panic!();
        }
    };
    let res = resp.json::<SolveResponse>().await.unwrap();
    res.result
}

fn load_stress_test_set() -> Vec<(Board, i8)> {
    let name = "problem/stress_test_54_1k.b81r";
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();
    for (_idx, line) in reader.lines().enumerate() {
        let line_str = line.unwrap();
        let desired: i8 = line_str[17..].parse().unwrap();
        match Board::from_base81(&line_str[..16]) {
            Ok(board) => {
                dataset.push((board, desired));
            }
            Err(_) => {
                panic!();
            }
        }
    }
    dataset
}

#[tokio::main]
async fn send_query(_matches: &ArgMatches) {
    let dataset = load_stress_test_set();
    let client = Arc::new(Client::new());
    let fut = futures::future::join_all(
        dataset
            .into_iter()
            .map(|(board, _)| {
                let client = client.clone();
                (board, client)
            })
            .map(|(board, client)| async move {
                let result = send_query_impl(board, &client).await;
                println!("{}", result);
            }),
    );
    let _ = tokio::spawn(fut).await;
}

fn main() {
    let arg_input_file = Arg::new("INPUT").short('i').required(true);
    let arg_output_file = Arg::new("OUTPUT").short('o').required(true);
    let arg_worker_urls = Arg::new("workers")
        .short('w')
        .action(ArgAction::Append)
        .value_parser(value_parser!(String));
    let matches = Command::new("Issen-rs")
        .subcommand(
            Command::new("ffobench")
                .about("Run FFO benchmark 1-79")
                .arg(arg_worker_urls.clone()),
        )
        .subcommand(
            Command::new("play")
                .about("Interactive play")
                .arg(Arg::new("player").short('i').required(true)),
        )
        .subcommand(
            Command::new("self-play")
                .about("Automatic self play")
                .arg(arg_worker_urls.clone()),
        )
        .subcommand(
            Command::new("gen-record")
                .about("Generate record")
                .arg(Arg::new("DEPTH").short('d').required(true))
                .arg(Arg::new("COUNT").short('n').required(true))
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("clean-record")
                .about("Cleaning record")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("gen-dataset")
                .about("Generate training dataset")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone())
                .arg(Arg::new("MAX_OUT").short('n').required(true)),
        )
        .subcommand(
            Command::new("train")
                .about("Train weights")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone())
                .arg(Arg::new("from").required(true))
                .arg(Arg::new("to").required(true))
                .arg(Arg::new("width").required(true)),
        )
        .subcommand(
            Command::new("grow-book")
                .about("Grow book")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone())
                .arg(Arg::new("REPEAT").short('r').required(true)),
        )
        .subcommand(
            Command::new("gen-book")
                .about("Generate book")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone()),
        )
        .subcommand(
            Command::new("gen-book-v2")
                .about("Generate book v2")
                .arg(arg_input_file.clone())
                .arg(arg_output_file.clone())
                .arg(Arg::new("MIN_COUNT").short('c').required(true)),
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
                .arg(Arg::new("str").short('b').required(true)),
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
        Some(("ffobench", matches)) => {
            ffo_benchmark(matches);
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
        Some(("gen-dataset", matches)) => {
            gen_dataset(matches);
        }
        Some(("train", matches)) => {
            train(matches);
        }
        Some(("gen-book", matches)) => {
            gen_book(matches);
        }
        Some(("gen-book-v2", matches)) => {
            gen_book_v2(matches);
        }
        Some(("grow-book", matches)) => {
            command_grow_book(matches);
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
