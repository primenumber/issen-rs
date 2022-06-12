use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::search::*;
use crate::table::*;
use crate::think::*;
use crate::train::*;
use clap::ArgMatches;
use futures::executor;
use futures::executor::ThreadPool;
use std::io::prelude::*;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;
use surf::{Client, Url};

fn read_hand() -> Option<usize> {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();
    if s.len() < 2 {
        return None;
    }
    if &s[0..2] == "ps" {
        return Some(64);
    }
    let mut itr = s.chars();
    let column_code = itr.next().unwrap() as usize;
    if column_code < 'a' as usize || ('h' as usize) < column_code {
        return None;
    }
    let row_code = itr.next().unwrap() as usize;
    if row_code < '1' as usize || ('8' as usize) < row_code {
        return None;
    }
    Some((row_code - '1' as usize) * 8 + (column_code - 'a' as usize))
}

pub fn play(matches: &ArgMatches) -> Board {
    let player_turn = matches.value_of("player").unwrap() == "B";

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
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let client: Arc<Client> = Arc::new(
        surf::Config::new()
            .set_base_url(Url::parse("http://localhost:7733").unwrap())
            .try_into()
            .unwrap(),
    );

    let mut board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    while !board.is_gameover() {
        board.print_with_sides();
        let hand = if board.is_black == player_turn {
            let hand: Hand;
            loop {
                println!("Input move");
                if let Some(h) = read_hand() {
                    hand = Hand::Play(h);
                    break;
                }
            }
            hand
        } else {
            println!("Thinking...");
            let best = if popcnt(board.empty()) > 22 {
                let time_limit = 1000;
                let start = Instant::now();
                let timer = Timer {
                    period: start,
                    time_limit,
                };
                let mut searcher = Searcher {
                    evaluator: evaluator.clone(),
                    cache: eval_cache.clone(),
                    timer: Some(timer),
                    node_count: 0,
                };
                let (score, best, depth) =
                    searcher.iterative_think(board, -64 * SCALE, 64 * SCALE, false);
                eprintln!("Estimated result: {}, Depth: {}", score, depth);
                best
            } else {
                let mut obj = SolveObj::new(
                    res_cache.clone(),
                    eval_cache.clone(),
                    evaluator.clone(),
                    search_params.clone(),
                    pool.clone(),
                    client.clone(),
                );
                executor::block_on(solve_with_move(board, &mut obj))
            };
            eval_cache.inc_gen();
            res_cache.inc_gen();
            best
        };
        match hand {
            Hand::Pass => board = board.pass(),
            Hand::Play(hand) => match board.play(hand) {
                Ok(next) => board = next,
                Err(_) => println!("Invalid move"),
            },
        }
    }
    println!("Game over");
    board.print_with_sides();
    board
}

pub fn self_play(_matches: &ArgMatches) -> Board {
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
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let client: Arc<Client> = Arc::new(
        surf::Config::new()
            .set_base_url(Url::parse("http://localhost:7733").unwrap())
            .try_into()
            .unwrap(),
    );

    let mut board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    while !board.is_gameover() {
        board.print_with_sides();
        println!("Thinking...");
        let best = if popcnt(board.empty()) > 22 {
            let time_limit = 1000;
            let start = Instant::now();
            let timer = Timer {
                period: start,
                time_limit,
            };
            let mut searcher = Searcher {
                evaluator: evaluator.clone(),
                cache: eval_cache.clone(),
                timer: Some(timer),
                node_count: 0,
            };
            let (score, best, depth) =
                searcher.iterative_think(board, -64 * SCALE, 64 * SCALE, false);
            eprintln!(
                "Estimated result: {}, Depth: {}, Nodes: {}",
                score, depth, searcher.node_count
            );
            best
        } else {
            let mut obj = SolveObj::new(
                res_cache.clone(),
                eval_cache.clone(),
                evaluator.clone(),
                search_params.clone(),
                pool.clone(),
                client.clone(),
            );
            executor::block_on(solve_with_move(board, &mut obj))
        };
        eval_cache.inc_gen();
        res_cache.inc_gen();
        let hand = best;
        match hand {
            Hand::Pass => board = board.pass(),
            Hand::Play(hand) => match board.play(hand) {
                Ok(next) => board = next,
                Err(_) => println!("Invalid move"),
            },
        }
    }
    println!("Game over, score: {}", board.score());
    board.print_with_sides();
    board
}

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
}

pub fn codingame(_matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
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
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let client: Arc<Client> = Arc::new(
        surf::Config::new()
            .set_base_url(Url::parse("http://localhost:7733")?)
            .try_into()?,
    );
    let mut reader = BufReader::new(std::io::stdin());

    // read initial states
    let id = {
        let mut buf = String::new();
        reader.read_line(&mut buf)?;
        parse_input!(buf, usize)
    };
    let rows = {
        let mut buf = String::new();
        reader.read_line(&mut buf)?;
        parse_input!(buf, usize)
    };
    let cols = rows;
    loop {
        // read board
        let mut black = 0u64;
        let mut white = 0u64;
        for row in 0..rows {
            let mut buf = String::new();
            reader.read_line(&mut buf)?;
            for (col, ch) in buf.chars().enumerate() {
                match ch {
                    '0' => black |= 1u64 << (row * cols + col),
                    '1' => white |= 1u64 << (row * cols + col),
                    _ => (),
                }
            }
        }
        let board = if id == 0 {
            Board {
                player: black,
                opponent: white,
                is_black: true,
            }
        } else {
            Board {
                player: white,
                opponent: black,
                is_black: false,
            }
        };
        // read actions
        let actions = {
            let mut buf = String::new();
            reader.read_line(&mut buf)?;
            parse_input!(buf, usize)
        };
        for _ in 0..actions {
            let mut buf = String::new();
            reader.read_line(&mut buf)?;
        }
        // search
        let best = if popcnt(board.empty()) > 16 {
            let time_limit = 130;
            let start = Instant::now();
            let timer = Timer {
                period: start,
                time_limit,
            };
            let mut searcher = Searcher {
                evaluator: evaluator.clone(),
                cache: eval_cache.clone(),
                timer: Some(timer),
                node_count: 0,
            };
            let (score, best, depth) =
                searcher.iterative_think(board, -64 * SCALE, 64 * SCALE, false);
            eprintln!(
                "Estimated result: {}, Depth: {}, Nodes: {}",
                score, depth, searcher.node_count
            );
            best
        } else {
            let mut obj = SolveObj::new(
                res_cache.clone(),
                eval_cache.clone(),
                evaluator.clone(),
                search_params.clone(),
                pool.clone(),
                client.clone(),
            );
            executor::block_on(solve_with_move(board, &mut obj))
        };
        eval_cache.inc_gen();
        res_cache.inc_gen();
        match best {
            Hand::Play(pos) => {
                println!("{}", pos_to_str(pos).to_ascii_lowercase());
            }
            _ => panic!(),
        }
    }
}
