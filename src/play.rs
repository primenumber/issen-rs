use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::think::*;
use crate::setup::*;
use crate::train::*;
use clap::ArgMatches;
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
use tokio::runtime::Runtime;

pub fn play(matches: &ArgMatches) -> Board {
    let player_turn = matches.get_one::<String>("player").unwrap() == "B";

    let solve_obj = setup_default();
    let sub_solver = Arc::new(SubSolver::new(&[]));

    let mut board = BoardWithColor::initial_state();

    while !board.is_gameover() {
        board.print_with_sides();
        let hand = if board.is_black == player_turn {
            let hand: Hand;
            loop {
                println!("Input move");
                let mut s = String::new();
                std::io::stdin().read_line(&mut s).unwrap();
                if let Ok(h) = s.trim().parse::<Hand>() {
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
                    evaluator: solve_obj.evaluator.clone(),
                    cache: solve_obj.eval_cache.clone(),
                    timer: Some(timer),
                    node_count: 0,
                };
                let (score, best, depth) = searcher.iterative_think(board.board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
                let scaled_score = score as f64 / SCALE as f64;
                eprintln!("Estimated result: {}, Depth: {}", scaled_score, depth);
                best
            } else {
                let mut solve_obj = solve_obj.clone();
                Runtime::new()
                    .unwrap()
                    .block_on(solve_with_move(board.board, &mut solve_obj, &sub_solver))
            };
            solve_obj.eval_cache.inc_gen();
            solve_obj.res_cache.inc_gen();
            best
        };
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
            Hand::Play(hand) => match board.play(hand) {
                Ok(next) => board = next,
                Err(_) => println!("Invalid move"),
            },
        }
    }
    println!("Game over");
    board.print_with_sides();
    board.board
}

pub fn self_play(matches: &ArgMatches) -> Board {
    let solve_obj = setup_default();
    let worker_urls = matches
        .get_many("workers")
        .unwrap()
        .cloned()
        .collect::<Vec<String>>();
    let sub_solver = Arc::new(SubSolver::new(&worker_urls));

    let mut board = BoardWithColor::initial_state();
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
                evaluator: solve_obj.evaluator.clone(),
                cache: solve_obj.eval_cache.clone(),
                timer: Some(timer),
                node_count: 0,
            };
            let (score, best, depth) = searcher.iterative_think(board.board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
            let secs = start.elapsed().as_secs_f64();
            let nps = (searcher.node_count as f64 / secs) as u64;
            eprintln!(
                "Estimated result: {}, Depth: {}, Nodes: {}, NPS: {}",
                score, depth, searcher.node_count, nps
            );
            best
        } else {
            let mut solve_obj = solve_obj.clone();
            Runtime::new()
                .unwrap()
                .block_on(solve_with_move(board.board, &mut solve_obj, &sub_solver))
        };
        solve_obj.eval_cache.inc_gen();
        solve_obj.res_cache.inc_gen();
        let hand = best;
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
            Hand::Play(hand) => match board.play(hand) {
                Ok(next) => board = next,
                Err(_) => println!("Invalid move"),
            },
        }
    }
    println!("Game over, score: {}", board.score());
    board.print_with_sides();
    board.board
}

fn self_play_worker(solve_obj: SolveObj, sub_solver: Arc<SubSolver>, initial_record: &[Hand]) -> (String, i8) {
    use std::fmt::Write;
    let mut board = BoardWithColor::initial_state();
    let mut record_str = String::new();
    for hand in initial_record {
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
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
            let (_score, best, _depth) = searcher.iterative_think(board.board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
            best
        } else {
            let mut obj = solve_obj.clone();
            Runtime::new()
                .unwrap()
                .block_on(solve_with_move(board.board, &mut obj, &sub_solver))
        };
        solve_obj.eval_cache.inc_gen();
        solve_obj.res_cache.inc_gen();
        let hand = best;
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
            Hand::Play(pos) => match board.play(pos) {
                Ok(next) => {
                    write!(&mut record_str, "{}", hand).unwrap();
                    board = next;
                }
                Err(_) => panic!(),
            },
        }
    }
    let result = board.score();
    (record_str, result)
}

fn generate_depth_n(board: Board, depth: usize, prev_passed: bool, record: &mut Vec<Hand>) -> Vec<Vec<Hand>> {
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
            result = generate_depth_n(board.pass_unchecked(), depth, true, record);
            record.pop();
        }
    }
    result
}

pub fn parallel_self_play(matches: &ArgMatches) {
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();
    let random_depth = *matches.get_one::<usize>("DEPTH").unwrap();
    let take_count = *matches.get_one::<usize>("COUNT").unwrap();

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let solve_obj = setup_default();
    let sub_solver = Arc::new(SubSolver::new(&[]));
    let initial_board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
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
        .map(|r| self_play_worker(solve_obj.clone(), sub_solver.clone(), r))
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
    let solve_obj = setup_default();
    let sub_solver = Arc::new(SubSolver::new(&[]));
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
        let board = 
            BoardWithColor {
                board: Board {
                player: black,
                opponent: white,
                },
                is_black: id == 0,
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
                evaluator: solve_obj.evaluator.clone(),
                cache: solve_obj.eval_cache.clone(),
                timer: Some(timer),
                node_count: 0,
            };
            let (score, best, depth) = searcher.iterative_think(board.board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
            eprintln!(
                "Estimated result: {}, Depth: {}, Nodes: {}",
                score, depth, searcher.node_count
            );
            best
        } else {
            let mut solve_obj = solve_obj.clone();
            Runtime::new()
                .unwrap()
                .block_on(solve_with_move(board.board, &mut solve_obj, &sub_solver))
        };
        solve_obj.eval_cache.inc_gen();
        solve_obj.res_cache.inc_gen();
        match best {
            Hand::Play(pos) => {
                println!("{}", pos_to_str(pos).to_ascii_lowercase());
            }
            _ => panic!(),
        }
    }
}
