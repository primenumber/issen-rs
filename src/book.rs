use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::think::*;
use crate::playout::*;
use crate::record::*;
use crate::serialize::*;
use crate::setup::*;
use crate::train::*;
use anyhow::Result;
use clap::ArgMatches;
use rand::prelude::*;
use rayon::prelude::*;
use std::cmp::max;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::runtime::Runtime;

pub struct Book {
    records: Vec<Record>,
    minimax_map: HashMap<Board, (Hand, i16)>,
}

impl Book {
    pub fn new() -> Book {
        Book {
            records: Vec::new(),
            minimax_map: HashMap::new(),
        }
    }

    pub fn import(path: &Path) -> Result<Book> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut book = Book::new();
        for line in reader.lines() {
            book.append(Record::parse(&line?)?)?;
        }
        Ok(book)
    }

    pub fn export(&self, path: &Path) -> Result<()> {
        let f = File::create(path)?;
        let mut writer = BufWriter::new(f);
        for record in &self.records {
            writeln!(writer, "{}", record)?;
        }
        Ok(())
    }

    pub fn lookup(&self, board: Board) -> Option<(Hand, i16)> {
        self.minimax_map.get(&board).copied()
    }

    pub fn append(&mut self, record: Record) -> Result<()> {
        let mut timeline = record.timeline()?;
        self.records.push(record);
        timeline.reverse();
        for (board, hand, score) in timeline {
            let mut best_hand = hand;
            let mut best_score = -(BOARD_SIZE as i16);
            if board.is_gameover() {
                best_score = score;
            } else if board.mobility_bits() == 0 {
                let next = board.pass_unchecked();
                if let Some((_, next_score)) = self.lookup(next) {
                    best_score = max(best_score, -next_score);
                }
            } else {
                for (next, h) in board.next_iter() {
                    if let Some((_, next_score)) = self.lookup(next) {
                        if -next_score > best_score {
                            best_score = -next_score;
                            best_hand = h;
                        }
                    }
                }
            }
            self.minimax_map.insert(board, (best_hand, best_score));
        }
        Ok(())
    }

    pub fn filter_record(&self, min_count: usize) -> Book {
        let mut records = self.records.clone();
        records.sort_unstable();
        records.dedup();
        let mut count_map = HashMap::<Board, usize>::new();
        for rec in &records {
            for (board, _hand, _score) in rec.timeline().unwrap() {
                if let Some(num) = count_map.get(&board) {
                    count_map.insert(board, num + 1);
                } else {
                    count_map.insert(board, 1);
                }
            }
        }
        let mut new_records = Vec::new();
        for rec in &records {
            let mut hands = Vec::new();
            let mut last_score = None;
            for (board, hand, _score) in rec.timeline().unwrap() {
                last_score = Some(self.lookup(board).unwrap().1);
                let &num = count_map.get(&board).unwrap();
                if num < min_count {
                    break;
                }
                hands.push(hand);
            }
            new_records.push(Record::new(rec.get_initial(), &hands, last_score.unwrap()));
        }
        new_records.dedup();
        let mut new_book = Book::new();
        for rec in new_records {
            new_book.append(rec).unwrap();
        }
        new_book
    }
}

fn grow_book(in_book_path: &Path, out_book_path: &Path, repeat: usize) -> Result<()> {
    let book = Arc::new(Mutex::new(Book::import(in_book_path)?));
    let mut solve_obj = setup_default();
    solve_obj.params.ybwc_empties_limit = 64;
    let rt = Runtime::new().unwrap();
    (0..repeat).into_par_iter().for_each(|i| {
        let mut rng = rand::thread_rng();
        let think_time_limit = 1 << rng.gen_range(7..=12);
        eprintln!("i={}, tl={}", i, think_time_limit);
        let mut board = Board::initial_state();
        let mut hands = Vec::new();
        let initial_hand = "F5".parse().unwrap();
        hands.push(initial_hand);
        board = board.play_hand(initial_hand).unwrap();
        while !board.is_gameover() {
            if let Some((hand, score)) = book.lock().unwrap().lookup(board) {
                let from_book = if score > 0 {
                    true
                } else if score == 0 {
                    rng.gen_bool(0.5)
                } else {
                    false
                };
                if from_book {
                    //eprintln!("i={}: book {} {} {}", i, board.empty().count_ones(), hand, score);
                    hands.push(hand);
                    board = board.play_hand(hand).unwrap();
                    continue;
                }
            }
            solve_obj.eval_cache.inc_gen();
            let hand = if board.empty().count_ones() <= 20 {
                let mut solve_obj = solve_obj.clone();
                rt.block_on(async move {
                    let sub_solver = Arc::new(SubSolver::new(&[]));
                    solve_with_move(board, &mut solve_obj, &sub_solver).await
                })
            } else {
                let start = Instant::now();
                let timer = Timer {
                    period: start,
                    time_limit: think_time_limit,
                };
                let mut searcher = Searcher {
                    evaluator: solve_obj.evaluator.clone(),
                    cache: solve_obj.eval_cache.clone(),
                    timer: Some(timer),
                    node_count: 0,
                };
                let (_score, hand, _depth) = searcher.iterative_think(board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
                //eprintln!("i={}, search {} {} {}", i, board.empty().count_ones(), hand, score as f64 / SCALE as f64);
                hand
            };
            hands.push(hand);
            board = board.play_hand(hand).unwrap();
        }
        book.lock()
            .unwrap()
            .append(Record::new(
                Board::initial_state(),
                &hands,
                board.score().into(),
            ))
            .unwrap();
    });
    book.lock().unwrap().export(out_book_path)?;
    Ok(())
}

pub fn command_grow_book(matches: &ArgMatches) {
    let in_book_path = matches.get_one::<String>("INPUT").unwrap();
    let out_book_path = matches.get_one::<String>("OUTPUT").unwrap();
    let repeat = matches
        .get_one::<String>("REPEAT")
        .unwrap()
        .parse()
        .unwrap();
    grow_book(Path::new(in_book_path), Path::new(out_book_path), repeat).unwrap();
}

pub fn gen_book_v2(matches: &ArgMatches) -> Option<()> {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();
    let min_count = matches
        .get_one::<String>("MIN_COUNT")
        .unwrap()
        .parse()
        .unwrap();

    let orig_book = Book::import(Path::new(input_path)).unwrap();
    let new_book = orig_book.filter_record(min_count);
    new_book.export(Path::new(output_path)).unwrap();
    Some(())
}

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
        if board.pass_unchecked().mobility_bits() == 0 {
            write_record(current, writer);
            return;
        }
        board.pass_unchecked()
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

fn get_best_records(board: Board, tree: &HashMap<Board, (i8, Vec<Hand>)>, current: &mut Vec<Hand>) -> Vec<Vec<Hand>> {
    let mut result = Vec::new();
    let board_2 = if board.mobility_bits() == 0 {
        if board.pass_unchecked().mobility_bits() == 0 {
            return vec![current.clone()];
        }
        board.pass_unchecked()
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
    let mut boards_set = vec![HashSet::new(); BOARD_SIZE];
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
                current = current.pass_unchecked();
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
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

    let boards_set = load_records_from_file(input_path);
    let boards_with_results_all = minimax_record_body(&boards_set);

    eprintln!("Writing to file...");
    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let initial_board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
    };
    minimax_record_impl(
        initial_board,
        &boards_with_results_all,
        &mut vec![],
        &mut writer,
    );
}

pub fn iterative_update_book(matches: &ArgMatches) {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

    let mut boards_set = load_records_from_file(input_path);

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);
    let initial_board = Board {
        player: 0x0000000810000000,
        opponent: 0x0000001008000000,
    };
    let solve_obj = setup_default();
    let sub_solver = Arc::new(SubSolver::new(&[]));
    let depth = 18;
    for _ in 0..100 {
        let boards_with_results_all = minimax_record_body(&boards_set);
        eprintln!("Get best records...");
        let best_records = get_best_records(initial_board, &boards_with_results_all, &mut vec![]);

        eprintln!("Thinking {} records...", best_records.len());
        let mut handles = Vec::new();
        let finished = Arc::new(AtomicUsize::new(0));
        for record in best_records {
            let mut solve_obj = solve_obj.clone();
            let sub_solver = sub_solver.clone();
            let finished = finished.clone();

            handles.push(tokio::task::spawn(async move {
                let mut updateds = Vec::new();
                for (idx, _) in record.iter().enumerate() {
                    let sub_record: Vec<_> = record[0..=idx]
                        .iter()
                        .map(|&x| match x {
                            Hand::Play(pos) => pos,
                            _ => panic!(),
                        })
                        .collect();
                    let updated = playout(&sub_record, &mut solve_obj, &sub_solver, 200, depth).await;
                    let count = finished.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if count % 1000 == 0 {
                        eprintln!("{}", count);
                    }
                    updateds.push(updated);
                }
                updateds
            }));
        }

        for handle in handles {
            let results = Runtime::new().unwrap().block_on(handle).unwrap();
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
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();
    let max_count = *matches.get_one::<i8>("MAX_COUNT").unwrap();

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
        let board = Board { player, opponent };
        if BOARD_SIZE as i8 - popcnt(board.empty()) > max_count {
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
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

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
