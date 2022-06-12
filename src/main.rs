mod bits;
mod board;
mod book;
mod eval;
mod play;
mod playout;
mod remote;
mod search;
mod serialize;
mod sparse_mat;
mod table;
mod think;
mod train;

use crate::bits::*;
use crate::board::*;
use crate::book::*;
use crate::eval::*;
use crate::play::*;
use crate::remote::*;
use crate::search::*;
use crate::table::*;
use crate::train::*;
use clap::{App, Arg, ArgMatches, SubCommand};
use futures::executor::ThreadPool;
use futures::task::SpawnExt;
use futures::{executor, future};
use std::convert::TryInto;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::str;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use surf::{Client, Url};
use tide::{Body, Request};

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
            let row_char = 0x30 + row;
            let col_char = 0x41 + col;
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
    pool: &ThreadPool,
    client: Arc<Client>,
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
                    search_params.clone(),
                    pool.clone(),
                    client.clone(),
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
        stability_cut_limit: 12,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-211122"));
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let client: Arc<Client> = Arc::new(
        surf::Config::new()
            .set_base_url(Url::parse("http://192.168.10.192:7733").unwrap())
            .try_into()
            .unwrap(),
    );
    let mut index: usize = 1;
    let mut stats = Vec::new();
    //stats.extend(solve_ffo(
    //    "problem/hard-20.obf",
    //    &mut index,
    //    &search_params,
    //    evaluator.clone(),
    //    &mut res_cache,
    //    &mut eval_cache,
    //    &pool,
    //));
    //stats.extend(solve_ffo(
    //    "problem/hard-25.obf",
    //    &mut index,
    //    &search_params,
    //    evaluator.clone(),
    //    &mut res_cache,
    //    &mut eval_cache,
    //    &pool,
    //));
    stats.extend(solve_ffo(
        "problem/fforum-1-19.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
        client.clone(),
    ));
    stats.extend(solve_ffo(
        "problem/fforum-20-39.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
        client.clone(),
    ));
    stats.extend(solve_ffo(
        "problem/fforum-40-59.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
        client.clone(),
    ));
    stats.extend(solve_ffo(
        "problem/fforum-60-79.obf",
        &mut index,
        &search_params,
        evaluator.clone(),
        &mut res_cache,
        &mut eval_cache,
        &pool,
        client.clone(),
    ));
    report_stats(&stats);
}

async fn worker_impl() -> tide::Result<()> {
    let search_params = SearchParams {
        reduce: false,
        ybwc_depth_limit: 12,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 16,
        eval_ordering_limit: 15,
        res_cache_limit: 11,
        stability_cut_limit: 12,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-211122"));
    let res_cache = ResCacheTable::new(256, 65536);
    let eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let client: Arc<Client> = Arc::new(
        surf::Config::new()
            .set_base_url(Url::parse("http://localhost:7733").unwrap())
            .try_into()
            .unwrap(),
    );
    let solve_obj = SolveObj::new(
        res_cache.clone(),
        eval_cache.clone(),
        evaluator.clone(),
        search_params.clone(),
        pool.clone(),
        client.clone(),
    );
    //tide::log::start();
    let mut app = tide::with_state(solve_obj);
    app.with(tide::log::LogMiddleware::new());
    app.at("/").post(|mut req: Request<SolveObj>| async move {
        let mut solve_obj = req.state().clone();
        let query: SolveRequest = req.body_json().await?;
        let board = Board::from_base81(&query.board).unwrap();
        let result = solve_inner(&mut solve_obj, board, query.alpha, query.beta, false);
        Body::from_json(&SolveResponse {
            result: result.0,
            node_count: result.1.node_count,
            st_cut_count: result.1.st_cut_count,
        })
    });
    app.listen("0.0.0.0:7733").await?;
    Ok(())
}

fn worker(_matches: &ArgMatches) {
    async_std::task::block_on(worker_impl()).unwrap();
}

fn send_query(_matches: &ArgMatches) {
    let name = "problem/stress_test_54_10k.b81r";
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    let client: Client = surf::Config::new()
        .set_base_url(Url::parse("http://localhost:7733").unwrap())
        .try_into()
        .unwrap();
    let pool = executor::ThreadPool::new().unwrap();
    let mut futures = Vec::new();
    for (_idx, line) in reader.lines().enumerate() {
        let client = client.clone();
        futures.push(
            pool.spawn_with_handle(async move {
                let line_str = line.unwrap();
                let desired: i8 = line_str[17..].parse().unwrap();
                match Board::from_base81(&line_str[..16]) {
                    Ok(board) => {
                        let data = SolveRequest {
                            board: board.to_base81(),
                            alpha: -64,
                            beta: 64,
                        };
                        let data_str = serde_json::json!(data);
                        let solve_res: SolveResponse = client
                            .post("/")
                            .body(http_types::Body::from_json(&data_str).unwrap())
                            .recv_json()
                            .await
                            .unwrap();
                        let res = solve_res.result;
                        if res != desired {
                            board.print();
                        }
                        assert_eq!(res, desired);
                    }
                    Err(_) => {
                        panic!();
                    }
                }
            })
            .unwrap(),
        );
    }
    executor::block_on(future::join_all(futures));
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
        .subcommand(SubCommand::with_name("self-play").about("Automatic self play"))
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
            SubCommand::with_name("minimax-record")
                .about("Minimax Record")
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
                )
                .arg(Arg::with_name("from").required(true).takes_value(true))
                .arg(Arg::with_name("to").required(true).takes_value(true))
                .arg(Arg::with_name("width").required(true).takes_value(true)),
        )
        .subcommand(
            SubCommand::with_name("update-record-v2")
                .about("Update record iterative")
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
            SubCommand::with_name("binarize")
                .about("Binarize weights file")
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
        .subcommand(
            SubCommand::with_name("pack-book")
                .about("Pack book file")
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
            SubCommand::with_name("last-table")
                .about("Generate last move flip count table")
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("last-mask")
                .about("Generate last move flip count mask")
                .arg(
                    Arg::with_name("OUTPUT")
                        .short("o")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("parse-board")
                .about("Parse board")
                .arg(
                    Arg::with_name("str")
                        .short("b")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("eval-stats")
                .about("Compute stats from dataset")
                .arg(
                    Arg::with_name("INPUT")
                        .short("i")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(SubCommand::with_name("codingame").about("Codingame player"))
        .subcommand(SubCommand::with_name("worker").about("worker mode"))
        .subcommand(SubCommand::with_name("query").about("query mode"))
        .get_matches();
    match matches.subcommand() {
        ("ffobench", Some(_matches)) => {
            ffo_benchmark();
        }
        ("play", Some(matches)) => {
            play(matches);
        }
        ("self-play", Some(matches)) => {
            self_play(matches);
        }
        ("clean-record", Some(matches)) => {
            clean_record(matches);
        }
        ("update-record", Some(matches)) => {
            update_record(matches);
        }
        ("update-record-v2", Some(matches)) => {
            iterative_update_book(matches);
        }
        ("minimax-record", Some(matches)) => {
            minimax_record(matches);
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
        ("binarize", Some(matches)) => {
            binarize_weights(matches);
        }
        ("pack", Some(matches)) => {
            pack_weights(matches);
        }
        ("pack-book", Some(matches)) => {
            pack_book(matches);
        }
        ("last-table", Some(matches)) => {
            gen_last_table(matches);
        }
        ("last-mask", Some(matches)) => {
            gen_last_mask(matches);
        }
        ("parse-board", Some(matches)) => {
            parse_board(matches);
        }
        ("eval-stats", Some(matches)) => {
            eval_stats(matches);
        }
        ("codingame", Some(matches)) => {
            codingame(matches).unwrap();
        }
        ("worker", Some(matches)) => {
            worker(matches);
        }
        ("query", Some(matches)) => {
            send_query(matches);
        }
        ("", None) => {
            eprintln!("Need subcommand");
        }
        _ => unreachable!(),
    }
}
