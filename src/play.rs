use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use crate::engine::think::*;
use crate::train::*;
use clap::ArgMatches;
use futures::executor;
use futures::executor::ThreadPool;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

fn read_hand() -> Option<Hand> {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();
    if s.len() < 2 {
        return None;
    }
    if &s[0..2] == "ps" {
        return Some(Hand::Pass);
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
    Some(Hand::Play(
        (row_code - '1' as usize) * 8 + (column_code - 'a' as usize),
    ))
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
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-220710"));
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
            let hand: Hand;
            loop {
                println!("Input move");
                if let Some(h) = read_hand() {
                    hand = h;
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
                    searcher.iterative_think(board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
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
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-220710"));
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
                searcher.iterative_think(board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
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

fn self_play_worker(mut solve_obj: SolveObj, initial_record: &[Hand]) -> (String, i8) {
    use std::fmt::Write;
    let mut board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    let mut record_str = String::new();
    for hand in initial_record {
        match hand {
            Hand::Pass => board = board.pass(),
            Hand::Play(pos) => match board.play(*pos) {
                Ok(next) => {
                    write!(&mut record_str, "{}", hand).unwrap();
                    board = next;
                }
                Err(_) => panic!(),
            },
        }
    }
    while !board.is_gameover() {
        let best = if popcnt(board.empty()) > 16 {
            let time_limit = 1000;
            let start = Instant::now();
            let timer = Timer {
                period: start,
                time_limit,
            };
            let mut searcher = Searcher {
                evaluator: solve_obj.evaluator.clone(),
                cache: solve_obj.eval_cache.clone(),
                timer: Some(timer),
                node_count: 0,
            };
            let (_score, best, _depth) =
                searcher.iterative_think(board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
            best
        } else {
            let mut obj = solve_obj.clone();
            executor::block_on(solve_with_move(board, &mut obj))
        };
        solve_obj.eval_cache.inc_gen();
        solve_obj.res_cache.inc_gen();
        let hand = best;
        match hand {
            Hand::Pass => board = board.pass(),
            Hand::Play(pos) => match board.play(pos) {
                Ok(next) => {
                    write!(&mut record_str, "{}", hand).unwrap();
                    board = next;
                }
                Err(_) => panic!(),
            },
        }
    }
    let result = if board.is_black {
        board.score()
    } else {
        -board.score()
    };
    (record_str, result)
}

fn generate_depth_n(
    board: Board,
    depth: usize,
    prev_passed: bool,
    record: &mut Vec<Hand>,
) -> Vec<Vec<Hand>> {
    if depth == 0 {
        return vec![record.clone()];
    }
    let mut result = Vec::new();
    let mut is_pass = true;
    for (next, hand) in board.next_iter() {
        is_pass = false;
        record.push(hand);
        result.extend(generate_depth_n(next, depth - 1, false, record));
        record.pop();
    }
    if is_pass {
        if prev_passed {
            return vec![];
        } else {
            record.push(Hand::Pass);
            result = generate_depth_n(board.pass(), depth, true, record);
            record.pop();
        }
    }
    result
}

pub fn parallel_self_play(matches: &ArgMatches) {
    let output_path = matches.value_of("OUTPUT").unwrap();
    let random_depth = matches.value_of("DEPTH").unwrap().parse().unwrap();
    let take_count = matches.value_of("COUNT").unwrap().parse().unwrap();

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let search_params = SearchParams {
        reduce: false,
        ybwc_depth_limit: 12,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 64,
        eval_ordering_limit: 15,
        res_cache_limit: 11,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-220710"));
    let res_cache = ResCacheTable::new(256, 65536);
    let eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let obj = SolveObj::new(res_cache, eval_cache, evaluator, search_params, pool);
    let initial_board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    let mut record = Vec::new();
    let initial_records = generate_depth_n(initial_board, random_depth, false, &mut record);
    eprintln!("{}", initial_records.len());
    let mut rng = SmallRng::from_entropy();
    let initial_records = initial_records
        .choose_multiple(&mut rng, take_count)
        .collect::<Vec<_>>();
    let mut results = Vec::new();
    initial_records
        .par_iter()
        .map(|r| self_play_worker(obj.clone(), r))
        .collect_into_vec(&mut results);
    for (record, score) in results {
        writeln!(writer, "{} {}", record, score).unwrap();
    }
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
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-220710"));
    let mut res_cache = ResCacheTable::new(256, 65536);
    let mut eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
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
                searcher.iterative_think(board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
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
