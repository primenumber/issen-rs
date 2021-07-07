use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::search::*;
use crate::table::*;
use clap::ArgMatches;
use futures::executor;
use futures::executor::ThreadPool;
use futures::task::SpawnExt;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::Arc;

// parse pos string [A-H][1-8]
fn parse_pos(s: &[u8]) -> Option<usize> {
    const CODE_1: u8 = '1' as u32 as u8;
    const CODE_8: u8 = '8' as u32 as u8;
    const CODE_A: u8 = 'A' as u32 as u8;
    const CODE_H: u8 = 'H' as u32 as u8;
    if s.len() != 2 {
        None
    } else if s[0] < CODE_A || s[0] > CODE_H {
        None
    } else if s[1] < CODE_1 || s[1] > CODE_8 {
        None
    } else {
        Some(((s[0] - CODE_A) + (s[1] - CODE_1) * 8) as usize)
    }
}

fn parse_record(line: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for chunk in line.as_bytes().chunks(2) {
        match parse_pos(chunk) {
            Some(pos) => result.push(pos),
            None => {
                return result;
            }
        }
    }
    result
}

fn step_by_pos(board: &Board, pos: usize) -> Option<Board> {
    match board.play(pos) {
        Ok(next) => Some(next),
        Err(_) => {
            if !board.mobility().is_empty() {
                None
            } else {
                match board.pass().play(pos) {
                    Ok(next) => Some(next),
                    Err(_) => None,
                }
            }
        }
    }
}

fn collect_boards(record: &[usize]) -> Option<Vec<Board>> {
    let mut board = Board {
        player: 0x0000_0008_1000_0000,
        opponent: 0x0000_0010_0800_0000,
        is_black: true,
    };
    let mut boards = Vec::with_capacity(70); // enough large
    for &pos in record {
        boards.push(board);
        board = match step_by_pos(&board, pos) {
            Some(next) => next,
            None => {
                return None;
            }
        };
    }
    if !board.is_gameover() {
        return None;
    }
    boards.push(board);
    Some(boards)
}

fn load_records(input_path: &str) -> Vec<Vec<Board>> {
    let in_f = File::open(input_path).unwrap();
    let mut reader = BufReader::new(in_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_records = input_line.trim().parse().unwrap();
    let mut result = Vec::new();
    for _i in 0..num_records {
        let mut input_line = String::new();
        reader.read_line(&mut input_line).unwrap();
        let record = parse_record(&input_line);
        if let Some(boards) = collect_boards(&record) {
            result.push(boards);
        }
    }
    result
}

pub fn clean_record(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

    let in_f = File::open(input_path).unwrap();
    let mut reader = BufReader::new(in_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_records = input_line.trim().parse().unwrap();
    let mut result = Vec::new();
    for _i in 0..num_records {
        let mut input_line = String::new();
        reader.read_line(&mut input_line).unwrap();
        let record = parse_record(&input_line);
        if let Some(_boards) = collect_boards(&record) {
            result.push(input_line);
        }
    }

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    write!(writer, "{}\n", result.len()).unwrap();
    for line in result {
        write!(writer, "{}", line).unwrap();
    }
}

async fn solve_with_move(board: Board, solve_obj: &mut SolveObj) -> usize {
    match solve_outer(solve_obj, board, -64, 64, false, 0).await.1 {
        Some(best) => best as usize,
        None => {
            let mut best_pos = None;
            let mut result = -65;
            for pos in board.mobility() {
                let next = board.play(pos).unwrap();
                let res = -solve_outer(solve_obj, next, -64, -result, false, 0).await.0;
                if res > result {
                    result = res;
                    best_pos = Some(pos);
                }
            }
            match best_pos {
                Some(pos) => pos,
                None => PASS,
            }
        }
    }
}

fn pos_to_str(pos: usize) -> String {
    let row = pos / 8;
    let col = pos % 8;
    let first = (col as u8) + ('A' as u8);
    let second = (row as u8) + ('1' as u8);
    let mut result = String::new();
    result.push(first as char);
    result.push(second as char);
    result
}

async fn create_record_by_solve(mut board: Board, solve_obj: &mut SolveObj) -> (String, Board) {
    let mut result = String::new();
    while !board.is_gameover() {
        let pos = solve_with_move(board, solve_obj).await;
        if pos != PASS {
            result += &pos_to_str(pos);
            board = board.play(pos).unwrap();
        } else {
            board = board.pass();
        }
    }
    (result, board)
}

async fn update_record_impl(
    record: &[usize],
    solve_obj: &mut SolveObj,
    depth: usize,
) -> Option<(String, i8)> {
    let mut board = Board {
        player: 0x0000_0008_1000_0000,
        opponent: 0x0000_0010_0800_0000,
        is_black: true,
    };
    let mut updated_record = String::new();
    for &pos in record {
        if popcnt(board.empty()) as usize <= depth {
            let (s, b) = create_record_by_solve(board, solve_obj).await;
            board = b;
            updated_record += &s;
            break;
        } else {
            board = match step_by_pos(&board, pos) {
                Some(next) => next,
                None => {
                    return None;
                }
            };
            updated_record += &pos_to_str(pos);
        }
    }
    let score = if board.is_black {
        board.score()
    } else {
        -board.score()
    };
    Some((updated_record, score))
}

pub fn update_record(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let depth = matches.value_of("DEPTH").unwrap().parse().unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

    let in_f = File::open(input_path).unwrap();
    let mut reader = BufReader::new(in_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_records = input_line.trim().parse().unwrap();

    let res_cache = ResCacheTable::new(256, 65536);
    let eval_cache = EvalCacheTable::new(256, 65536);
    let evaluator = Arc::new(Evaluator::new("table"));
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
    let pool = ThreadPool::new().unwrap();

    let mut handles = Vec::new();
    for i in 0..num_records {
        let mut solve_obj = SolveObj::new(
            res_cache.clone(),
            eval_cache.clone(),
            evaluator.clone(),
            search_params.clone(),
            pool.clone(),
        );

        let mut input_line = String::new();
        reader.read_line(&mut input_line).unwrap();

        handles.push(
            pool.spawn_with_handle(async move {
                if i % 1000 == 0 {
                    eprintln!("{}", i);
                }
                let record = parse_record(&input_line);
                if let Some(updated_record) =
                    update_record_impl(&record, &mut solve_obj, depth).await
                {
                    Some(updated_record)
                } else {
                    None
                }
            })
            .unwrap(),
        );
    }

    let mut result = Vec::new();
    for handle in handles {
        if let Some((line, score)) = executor::block_on(handle) {
            result.push(format!("{} {}\n", line, score));
        }
    }

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    write!(writer, "{}\n", result.len()).unwrap();
    for line in result {
        write!(writer, "{}", line).unwrap();
    }
}

pub fn gen_dataset(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();
    let max_output = matches.value_of("MAX_OUT").unwrap().parse().unwrap();

    eprintln!("Parse input...");
    let boards_list = load_records(input_path);

    eprintln!("Generate boards_set...");
    let mut boards_set = vec![HashSet::new(); 64];
    for boards in boards_list {
        for board in boards {
            boards_set[popcnt(board.empty()) as usize].insert(board);
        }
    }

    let mut total_boards = 0;
    for boards in &boards_set {
        total_boards += boards.len();
    }
    eprintln!("Total board count = {}", total_boards);

    eprintln!("Minimax-ing results...");
    let mut boards_with_results = HashMap::new();
    for boards in &boards_set {
        for &board in boards {
            let mut current = board;
            let mut mobility = current.mobility();
            let is_pass = mobility.is_empty();
            if is_pass {
                current = current.pass();
                mobility = current.mobility();
                if mobility.is_empty() {
                    boards_with_results.insert(board, board.score());
                    continue;
                }
            }
            let mut best_score = None;
            for pos in mobility {
                let next = current.play(pos).unwrap();
                if let Some(score) = boards_with_results.get(&next) {
                    best_score = Some(max(-score, best_score.unwrap_or(-64)));
                }
            }
            if is_pass {
                boards_with_results.insert(board, -best_score.unwrap());
            } else {
                boards_with_results.insert(current, best_score.unwrap());
            }
        }
    }

    boards_with_results.retain(|&k, _| popcnt(k.empty()) >= 8);
    eprintln!("Remaining board count = {}", boards_with_results.len());

    eprintln!("Writing to file...");
    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    write!(
        &mut writer,
        "{}\n",
        min(boards_with_results.len(), max_output)
    );
    for (idx, (board, score)) in boards_with_results.iter().enumerate() {
        if idx >= max_output {
            break;
        }
        write!(
            &mut writer,
            "{:016x} {:016x} {}\n",
            board.player, board.opponent, score
        )
        .unwrap();
    }
    eprintln!("Finished!");
}
