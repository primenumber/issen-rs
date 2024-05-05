use crate::book::*;
use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::pattern_eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use crate::engine::think::*;
use crate::setup::*;
use crate::train::*;
use clap::ArgMatches;
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub fn play(matches: &ArgMatches) -> Board {
    let player_turn = matches.get_one::<String>("player").unwrap() == "B";

    let mut solve_obj = setup_default();
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
                    cache_gen: solve_obj.cache_gen,
                };
                let (score, best, depth) = searcher.iterative_think(
                    board.board,
                    searcher.evaluator.score_min(),
                    searcher.evaluator.score_max(),
                    false,
                    3,
                    0,
                );
                let scaled_score = score as f64 / solve_obj.evaluator.score_scale() as f64;
                eprintln!("Estimated result: {}, Depth: {}", scaled_score, depth);
                best
            } else {
                let mut solve_obj = solve_obj.clone();
                solve_with_move(board.board, &mut solve_obj, &sub_solver, None)
            };
            solve_obj.cache_gen += 1;
            best
        };
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
            Hand::Play(hand) => match board.play(hand) {
                Some(next) => board = next,
                None => println!("Invalid move"),
            },
        }
    }
    println!("Game over");
    board.print_with_sides();
    board.board
}

pub fn self_play(matches: &ArgMatches) -> Board {
    let mut solve_obj = setup_default();
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
                cache_gen: solve_obj.cache_gen,
            };
            let (score, best, depth) = searcher.iterative_think(
                board.board,
                searcher.evaluator.score_min(),
                searcher.evaluator.score_max(),
                false,
                3,
                0,
            );
            let secs = start.elapsed().as_secs_f64();
            let nps = (searcher.node_count as f64 / secs) as u64;
            eprintln!(
                "Estimated result: {}, Depth: {}, Nodes: {}, NPS: {}",
                score as f32 / searcher.evaluator.score_scale() as f32,
                depth,
                searcher.node_count,
                nps
            );
            best
        } else {
            let mut solve_obj = solve_obj.clone();
            solve_with_move(board.board, &mut solve_obj, &sub_solver, None)
        };
        solve_obj.cache_gen += 1;
        let hand = best;
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
            Hand::Play(hand) => match board.play(hand) {
                Some(next) => board = next,
                None => println!("Invalid move"),
            },
        }
    }
    println!("Game over, score: {}", board.score());
    board.print_with_sides();
    board.board
}

fn self_play_worker<Eval: Evaluator>(
    mut solve_obj: SolveObj<Eval>,
    sub_solver: Arc<SubSolver>,
    initial_record: &[Hand],
) -> (String, i8) {
    use std::fmt::Write;
    let mut board = BoardWithColor::initial_state();
    let mut record_str = String::new();
    for hand in initial_record {
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
            Hand::Play(pos) => match board.play(*pos) {
                Some(next) => {
                    write!(&mut record_str, "{}", hand).unwrap();
                    board = next;
                }
                None => panic!(),
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
                cache_gen: solve_obj.cache_gen,
            };
            let (_score, best, _depth) = searcher.iterative_think(
                board.board,
                searcher.evaluator.score_min(),
                searcher.evaluator.score_max(),
                false,
                3,
                0,
            );
            best
        } else {
            let mut obj = solve_obj.clone();
            solve_with_move(board.board, &mut obj, &sub_solver, Some(1))
        };
        solve_obj.cache_gen += 1;
        let hand = best;
        match hand {
            Hand::Pass => board = board.pass_unchecked(),
            Hand::Play(pos) => match board.play(pos) {
                Some(next) => {
                    write!(&mut record_str, "{}", hand).unwrap();
                    board = next;
                }
                None => panic!(),
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
    let res_cache = Arc::new(ResCacheTable::new(512, 4096));
    let eval_cache = Arc::new(EvalCacheTable::new(512, 4096));
    let evaluator = Arc::new(PatternLinearEvaluator::load(Path::new("table-220710")).unwrap());
    let search_params = SearchParams {
        reduce: false,
        parallel_depth_limit: 16,
        parallel_empties_limit: 16,
        eval_ordering_limit: 15,
        res_cache_limit: 12,
        local_res_cache_limit: 9,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 5,
    };
    let mut solve_obj = SolveObj::new(res_cache, eval_cache, evaluator, search_params, 0);
    let sub_solver = Arc::new(SubSolver::new(&[]));
    let mut reader = BufReader::new(std::io::stdin());
    let book = Book::import(Path::new("medium2.book"))?;

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
        let is_black = id == 0;
        let board = BoardWithColor {
            board: Board {
                player: if is_black { black } else { white },
                opponent: if is_black { white } else { black },
            },
            is_black,
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
        // book or search
        let best = if let Some((hands, _score)) = book.lookup_with_symmetry(board.board) {
            *hands.first().unwrap()
        } else if popcnt(board.empty()) > 16 {
            let time_limit = 130;
            let start = Instant::now();
            let timer = Timer {
                period: start,
                time_limit,
            };
            let searcher = Searcher {
                evaluator: solve_obj.evaluator.clone(),
                cache: solve_obj.eval_cache.clone(),
                timer: Some(timer),
                node_count: 0,
                cache_gen: solve_obj.cache_gen,
            };
            let (score, best, depth, node_count) = think_parallel(
                &searcher,
                board.board,
                searcher.evaluator.score_min(),
                searcher.evaluator.score_max(),
                false,
            );
            eprintln!(
                "Estimated result: {}, Depth: {}, Nodes: {}",
                score, depth, node_count
            );
            best
        } else {
            let mut solve_obj = solve_obj.clone();
            solve_with_move(board.board, &mut solve_obj, &sub_solver, None)
        };
        solve_obj.cache_gen += 1;
        match best {
            Hand::Play(pos) => {
                println!("{}", pos_to_str(pos).to_ascii_lowercase());
            }
            _ => panic!(),
        }
    }
}
