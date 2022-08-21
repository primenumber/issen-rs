use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use crate::playout::*;
use crate::serialize::*;
use crate::train::*;
use clap::ArgMatches;
use futures::executor;
use futures::executor::ThreadPool;
use futures::task::SpawnExt;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

fn write_record<W: Write>(current: &mut Vec<Hand>, writer: &mut W) {
    for hand in current {
        if let Hand::Play(pos) = hand {
            write!(writer, "{}", pos_to_str(*pos)).unwrap();
        }
    }
    writeln!(writer).unwrap();
}

fn minimax_record_impl<W: Write>(
    board: Board,
    tree: &HashMap<Board, (i8, Vec<Hand>)>,
    current: &mut Vec<Hand>,
    writer: &mut W,
) {
    let board_2 = if board.mobility_bits() == 0 {
        if board.pass().mobility_bits() == 0 {
            write_record(current, writer);
            return;
        }
        board.pass()
    } else {
        board
    };
    for rotate in 0..4 {
        for mirror in [false, true] {
            if let Some((_, hands)) = tree.get(&board.transform(rotate, mirror)) {
                for &hand in hands {
                    let t_hand = hand.transform(0, mirror).transform((4 - rotate) % 4, false);
                    match t_hand {
                        Hand::Play(pos) => {
                            current.push(t_hand);
                            let next = board_2.play(pos).unwrap();
                            minimax_record_impl(next, tree, current, writer);
                            current.pop();
                        }
                        _ => panic!(),
                    }
                }
                return;
            }
        }
    }
}

fn get_best_records(
    board: Board,
    tree: &HashMap<Board, (i8, Vec<Hand>)>,
    current: &mut Vec<Hand>,
) -> Vec<Vec<Hand>> {
    let mut result = Vec::new();
    let board_2 = if board.mobility_bits() == 0 {
        if board.pass().mobility_bits() == 0 {
            return vec![current.clone()];
        }
        board.pass()
    } else {
        board
    };
    for rotate in 0..4 {
        for mirror in [false, true] {
            let t_board = board.transform(rotate, mirror);
            if let Some((_, hands)) = tree.get(&t_board) {
                for &hand in hands {
                    let t_hand = hand.transform(0, mirror).transform((4 - rotate) % 4, false);
                    match t_hand {
                        Hand::Play(pos) => {
                            current.push(t_hand);
                            let next = board_2.play(pos).unwrap();
                            result.extend(get_best_records(next, tree, current));
                            current.pop();
                        }
                        _ => panic!(),
                    }
                }
                return result;
            }
        }
    }
    result
}

fn load_records_from_file(input_path: &str) -> Vec<HashSet<Board>> {
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

    boards_set
}

fn minimax_record_body(boards_set: &Vec<HashSet<Board>>) -> HashMap<Board, (i8, Vec<Hand>)> {
    eprintln!("Minimax-ing results...");
    let mut boards_with_results_all = HashMap::<Board, (i8, Vec<Hand>)>::new();
    for boards in boards_set {
        let boards_with_results_next = [
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new())),
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new())),
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new())),
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new())),
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new())),
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new())),
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new())),
        ];
        boards.par_iter().for_each(|&board| {
            let mut current = board;
            let mut mobility = current.mobility_bits();
            let is_pass = mobility == 0;
            if is_pass {
                current = current.pass();
                mobility = current.mobility_bits();
                if mobility == 0 {
                    let s = vec![];
                    let idx = (board.player % 7) as usize;
                    boards_with_results_next[idx]
                        .lock()
                        .unwrap()
                        .insert(board, (board.score(), s));
                    return;
                }
            }
            let mut best_score = None;
            let mut best_poses = HashSet::new();
            for (next, pos) in current.next_iter() {
                for next_sym in next.sym_boards() {
                    if let Some((score, _)) = boards_with_results_all.get(&next_sym) {
                        let new_score = -score;
                        let old_score = best_score.unwrap_or(-65);
                        match new_score.cmp(&old_score) {
                            Ordering::Greater => {
                                best_score = Some(new_score);
                                best_poses.clear();
                                best_poses.insert(pos);
                            }
                            Ordering::Equal => {
                                best_poses.insert(pos);
                            }
                            _ => (),
                        }
                    }
                }
            }
            let best_poses: Vec<_> = best_poses.into_iter().collect();
            let score = if is_pass {
                -best_score.unwrap()
            } else {
                best_score.unwrap()
            };
            let idx = (board.player % 7) as usize;
            boards_with_results_next[idx]
                .lock()
                .unwrap()
                .insert(board, (score, best_poses));
        });
        for next in boards_with_results_next {
            for (&k, v) in next.lock().unwrap().iter() {
                boards_with_results_all.insert(k, v.clone());
            }
        }
    }

    boards_with_results_all
}

pub fn minimax_record(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

    let boards_set = load_records_from_file(input_path);
    let boards_with_results_all = minimax_record_body(&boards_set);

    eprintln!("Writing to file...");
    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let initial_board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    minimax_record_impl(
        initial_board,
        &boards_with_results_all,
        &mut vec![],
        &mut writer,
    );
}

pub fn iterative_update_book(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

    let mut boards_set = load_records_from_file(input_path);

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);
    let initial_board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    let res_cache = ResCacheTable::new(256, 65536);
    let eval_cache = EvalCacheTable::new(256, 65536);
    let evaluator = Arc::new(Evaluator::new("table-220710"));
    let search_params = SearchParams {
        reduce: false,
        ybwc_depth_limit: 10,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 17,
        eval_ordering_limit: 16,
        res_cache_limit: 11,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let depth = 18;
    let pool = ThreadPool::new().unwrap();
    for _ in 0..100 {
        let boards_with_results_all = minimax_record_body(&boards_set);
        eprintln!("Get best records...");
        let best_records = get_best_records(initial_board, &boards_with_results_all, &mut vec![]);

        eprintln!("Thinking {} records...", best_records.len());
        let mut handles = Vec::new();
        let finished = Arc::new(AtomicUsize::new(0));
        for record in best_records {
            let mut solve_obj = SolveObj::new(
                res_cache.clone(),
                eval_cache.clone(),
                evaluator.clone(),
                search_params.clone(),
                pool.clone(),
            );
            let finished = finished.clone();

            handles.push(
                pool.spawn_with_handle(async move {
                    let mut updateds = Vec::new();
                    for (idx, _) in record.iter().enumerate() {
                        let sub_record: Vec<_> = record[0..=idx]
                            .iter()
                            .map(|&x| match x {
                                Hand::Play(pos) => pos,
                                _ => panic!(),
                            })
                            .collect();
                        let updated = playout(&sub_record, &mut solve_obj, 200, depth).await;
                        let count = finished.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if count % 1000 == 0 {
                            eprintln!("{}", count);
                        }
                        updateds.push(updated);
                    }
                    updateds
                })
                .unwrap(),
            );
        }

        for handle in handles {
            let results = executor::block_on(handle);
            for (line, score) in results.iter().flatten() {
                writeln!(writer, "{} {}", line, score).unwrap();
                let record = parse_record(line);
                let boards = collect_boards(&record).unwrap();
                for board in boards {
                    boards_set[popcnt(board.empty()) as usize].insert(board);
                }
            }
        }
    }
}

pub fn gen_book(matches: &ArgMatches) -> Option<()> {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();
    let max_count = matches.value_of("MAX_COUNT").unwrap().parse().unwrap();

    let in_f = File::open(input_path).ok()?;
    let mut reader = BufReader::new(in_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_boards = input_line.trim().parse().unwrap();
    let mut records = Vec::new();
    for _i in 0..num_boards {
        input_line.clear();
        reader.read_line(&mut input_line).unwrap();
        let data: Vec<&str> = input_line.split(' ').collect();
        let player = u64::from_str_radix(data[0], 16).ok()?;
        let opponent = u64::from_str_radix(data[1], 16).ok()?;
        let board = Board {
            player,
            opponent,
            is_black: true, // dummy
        };
        if 64 - popcnt(board.empty()) > max_count {
            continue;
        }
        records.push((
            board,
            data[2].trim().parse::<i8>().unwrap(),
            data[3].trim().parse::<usize>().unwrap(),
        ));
    }

    records.sort_unstable_by_key(|k| popcnt(k.0.empty()));
    //let book = HashMap::new();

    for (board, _score, pos) in records {
        let _next = match board.play(pos) {
            Ok(n) => n,
            Err(_) => continue,
        };
    }

    let out_f = File::create(output_path).ok()?;
    let mut _writer = BufWriter::new(out_f);

    Some(())
}

pub fn pack_book(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

    let in_f = File::open(input_path).unwrap();
    let reader = BufReader::new(in_f);

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    for line in reader.lines() {
        write!(writer, ">").unwrap();
        for pos_bytes in line.unwrap().trim().as_bytes().chunks(2) {
            let pos = if pos_bytes[0] < 0x60 {
                (pos_bytes[0] - 0x41) + (pos_bytes[1] - 0x31) * 8
            } else {
                (pos_bytes[0] - 0x61) + (pos_bytes[1] - 0x31) * 8
            };
            write!(writer, "{}", encode_base64_impl(pos).unwrap() as char).unwrap();
        }
    }
    write!(writer, "\n").unwrap();
}
