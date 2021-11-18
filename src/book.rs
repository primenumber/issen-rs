use crate::bits::*;
use crate::board::*;
use crate::serialize::*;
use crate::train::*;
use clap::ArgMatches;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
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
    board: &Board,
    tree: &HashMap<Board, (i8, Vec<Hand>)>,
    current: &mut Vec<Hand>,
    writer: &mut W,
) {
    if let Some((_, hands)) = tree.get(board) {
        for &hand in hands {
            match hand {
                Hand::Pass => {
                    write_record(current, writer);
                }
                Hand::Play(pos) => {
                    current.push(hand);
                    let next = match board.play(pos) {
                        Ok(n) => n,
                        Err(_) => board.pass().play(pos).unwrap(),
                    };
                    let next = next;
                    minimax_record_impl(&next, tree, current, writer);
                    current.pop();
                }
            }
        }
    }
}

fn sym_boards(mut board: Board) -> Vec<Board> {
    let mut boards = Vec::new();
    for _ in 0..4 {
        board = board.rot90();
        boards.push(board);
        boards.push(board.flip_diag());
    }
    boards
}

pub fn minimax_record(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

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
    let mut boards_with_results_all = HashMap::<Board, (i8, Vec<Hand>)>::new();
    for boards in &boards_set {
        let boards_with_results_next =
            Arc::new(Mutex::new(HashMap::<Board, (i8, Vec<Hand>)>::new()));
        {
            boards.par_iter().for_each(|&board| {
                let mut current = board;
                let mut mobility = current.mobility();
                let is_pass = mobility.is_empty();
                if is_pass {
                    current = current.pass();
                    mobility = current.mobility();
                    if mobility.is_empty() {
                        let s = vec![Hand::Pass];
                        boards_with_results_next
                            .lock()
                            .unwrap()
                            .insert(board, (board.score(), s));
                        return;
                    }
                }
                let mut best_score = None;
                let mut best_poses = HashSet::new();
                for (next, pos) in current.next_iter() {
                    for next_sym in sym_boards(next) {
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
                if is_pass {
                    boards_with_results_next
                        .lock()
                        .unwrap()
                        .insert(board, (-best_score.unwrap(), best_poses));
                } else {
                    boards_with_results_next
                        .lock()
                        .unwrap()
                        .insert(current, (best_score.unwrap(), best_poses));
                }
            });
        }
        for (&k, v) in boards_with_results_next.lock().unwrap().iter() {
            boards_with_results_all.insert(k, v.clone());
        }
    }

    eprintln!("Writing to file...");
    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let initial_board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
        is_black: true,
    };
    minimax_record_impl(
        &initial_board,
        &boards_with_results_all,
        &mut vec![],
        &mut writer,
    );
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
