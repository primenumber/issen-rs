use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::think::*;
use crate::record::*;
use crate::setup::*;
use anyhow::Result;
use clap::ArgMatches;
use rand::prelude::*;
use rayon::prelude::*;
use std::cmp::max;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
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

    pub fn from_records(records: &[Record]) -> Book {
        let mut book = Book::new();
        for rec in records {
            book.append(rec.clone()).unwrap();
        }
        book
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

    pub fn lookup_with_symmetry(&self, mut board: Board) -> Option<(Hand, i16)> {
        let mut fboard = board.flip_diag();
        for i in 0..4 {
            if let Some((hand, score)) = self.lookup(board) {
                return Some((hand.transform(4 - i, false), score));
            }
            if let Some((hand, score)) = self.lookup(fboard) {
                return Some((hand.transform(4 - i, true), score));
            }
            board = board.rot90();
            fboard = fboard.rot90();
        }
        None
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

    pub fn minimax_record(&self) -> Book {
        let mut records = self.records.clone();
        records.sort_unstable();
        records.dedup();
        let mut new_records = Vec::new();
        for rec in records {
            let mut is_best_record = true;
            for (board, _hand, score) in rec.timeline().unwrap() {
                if self.lookup(board).unwrap().1 > score {
                    is_best_record = false;
                    break;
                }
            }
            if is_best_record {
                new_records.push(rec);
            }
        }
        Book::from_records(&new_records)
    }

    pub fn filter_record(&self, min_count: usize) -> Book {
        let mut records = self.records.clone();
        records.sort_unstable();
        records.dedup();
        let mut count_map = HashMap::<Board, usize>::new();
        for rec in &records {
            for (board, _hand, _score) in rec.timeline().unwrap() {
                count_map.insert(board, count_map.get(&board).unwrap_or(&0) + 1);
            }
        }
        let mut new_records = Vec::new();
        for rec in &records {
            let mut hands = Vec::new();
            let mut last_score = None;
            for (board, hand, _score) in rec.timeline().unwrap() {
                last_score = Some(self.lookup(board).unwrap().1);
                if *count_map.get(&board).unwrap() < min_count {
                    break;
                }
                hands.push(hand);
            }
            new_records.push(Record::new(rec.get_initial(), &hands, last_score.unwrap()));
        }
        new_records.dedup();
        Book::from_records(&new_records)
    }
}

fn search(board: Board, think_time_limit: u128, solve_obj: &SolveObj, rt: &Runtime) -> Hand {
    solve_obj.eval_cache.inc_gen();
    if board.empty().count_ones() <= 20 {
        solve_obj.res_cache.inc_gen();
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
        hand
    }
}

fn gen_opening(rng: &mut SmallRng) -> (Board, Vec<Hand>) {
    let mut board = Board::initial_state();
    let mut hands = Vec::new();
    let initial_hand = "F5".parse().unwrap();
    hands.push(initial_hand);
    board = board.play_hand(initial_hand).unwrap();
    // NOTE: Prevent searcher from always choosing D6
    let second_hand = if rng.gen_bool(0.5) { "D6" } else { "F6" }.parse().unwrap();
    hands.push(second_hand);
    board = board.play_hand(second_hand).unwrap();
    (board, hands)
}

fn play_with_book(
    book: Arc<Mutex<Book>>,
    think_time_limit: u128,
    solve_obj: &SolveObj,
    rt: &Runtime,
    rng: &mut SmallRng,
) {
    let (mut board, mut hands) = gen_opening(rng);
    while !board.is_gameover() {
        if let Some((hand, score)) = book.lock().unwrap().lookup(board) {
            let from_book = match score.cmp(&0) {
                Ordering::Less => false,
                Ordering::Equal => rng.gen_bool(0.8),
                Ordering::Greater => true,
            };
            if from_book {
                hands.push(hand);
                board = board.play_hand(hand).unwrap();
                continue;
            }
        }
        let hand = search(board, think_time_limit, solve_obj, rt);
        hands.push(hand);
        board = board.play_hand(hand).unwrap();
    }
    let record = Record::new(Board::initial_state(), &hands, board.score().into());
    eprintln!("{}", record);
    book.lock().unwrap().append(record).unwrap();
}

fn grow_book(in_book_path: &Path, out_book_path: &Path, repeat: usize) -> Result<()> {
    let book = Arc::new(Mutex::new(Book::import(in_book_path)?));
    let mut solve_obj = setup_default();
    solve_obj.params.ybwc_empties_limit = 64;
    let rt = Runtime::new().unwrap();
    (0..repeat).into_par_iter().for_each(|i| {
        let mut rng = SmallRng::from_entropy();
        let think_time_limit = 1 << rng.gen_range(7..=12);
        eprintln!("i={}, tl={}", i, think_time_limit);
        play_with_book(book.clone(), think_time_limit, &solve_obj, &rt, &mut rng);
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
