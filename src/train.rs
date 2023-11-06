use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::table::*;
use crate::engine::think::*;
use crate::serialize::*;
use crate::sparse_mat::*;
use clap::ArgMatches;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::str;
use std::sync::Arc;

const PACKED_SCALE: i32 = 256;

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

pub fn parse_record(line: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for chunk in line.as_bytes().chunks(2) {
        if chunk == "ps".as_bytes() {
            continue;
        }
        match parse_pos(chunk) {
            Some(pos) => result.push(pos),
            None => {
                return result;
            }
        }
    }
    result
}

pub fn step_by_pos(board: &Board, pos: usize) -> Option<Board> {
    match board.play(pos) {
        Some(next) => Some(next),
        None => {
            if !board.mobility().is_empty() {
                None
            } else {
                board.pass_unchecked().play(pos)
            }
        }
    }
}

pub fn collect_boards(record: &[usize]) -> Option<Vec<Board>> {
    let mut board = Board {
        player: 0x0000_0008_1000_0000,
        opponent: 0x0000_0010_0800_0000,
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

fn boards_from_record_impl(board: Board, record: &[usize]) -> (Vec<(Board, i8, Hand)>, i8) {
    match record.first() {
        Some(&first) => {
            let ((mut boards, score), hand) = if board.mobility_bits() == 0 {
                (
                    boards_from_record_impl(board.pass_unchecked(), record),
                    Hand::Pass,
                )
            } else {
                (
                    boards_from_record_impl(step_by_pos(&board, first).unwrap(), &record[1..]),
                    Hand::Play(first),
                )
            };
            boards.insert(0, (board, -score, hand));
            (boards, -score)
        }
        None => (vec![(board, board.score(), Hand::Pass)], board.score()),
    }
}

fn boards_from_record(line: &str) -> Vec<(Board, i8, Hand)> {
    let record = parse_record(line);
    let board = Board::initial_state();
    boards_from_record_impl(board, &record).0
}

pub fn clean_record(matches: &ArgMatches) {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

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

    writeln!(writer, "{}", result.len()).unwrap();
    for line in result {
        write!(writer, "{}", line).unwrap();
    }
}

pub fn pos_to_str(pos: usize) -> String {
    let row = pos / 8;
    let col = pos % 8;
    let first = (col as u8) + b'A';
    let second = (row as u8) + b'1';
    let mut result = String::new();
    result.push(first as char);
    result.push(second as char);
    result
}

pub fn gen_dataset(matches: &ArgMatches) {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();
    let max_output = *matches.get_one::<usize>("MAX_OUT").unwrap();

    eprintln!("Parse input...");
    let in_f = File::open(input_path).unwrap();
    let mut reader = BufReader::new(in_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_records = input_line.trim().parse().unwrap();
    let mut boards_with_results = Vec::new();
    for _i in 0..num_records {
        let mut input_line = String::new();
        reader.read_line(&mut input_line).unwrap();
        boards_with_results.append(&mut boards_from_record(&input_line));
    }

    eprintln!("Total board count = {}", boards_with_results.len());

    eprintln!("Writing to file...");
    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    writeln!(
        &mut writer,
        "{}",
        min(boards_with_results.len(), max_output)
    )
    .unwrap();
    for (idx, (board, score, hand)) in boards_with_results.iter().enumerate() {
        if idx >= max_output {
            break;
        }
        if let Hand::Play(pos) = hand {
            writeln!(
                &mut writer,
                "{:016x} {:016x} {} {}",
                board.player, board.opponent, score, pos,
            )
            .unwrap();
        }
    }
    eprintln!("Finished!");
}

struct Base3 {
    table: Vec<usize>,
}

impl Base3 {
    fn new(max_bits: usize) -> Base3 {
        let n = 1 << max_bits;
        let table: Vec<usize> = (0..n)
            .map(|i| {
                let mut v = 0;
                let mut p3 = 1;
                for j in 0..max_bits {
                    if (i >> j) & 1 == 1 {
                        v += p3;
                    }
                    p3 *= 3;
                }
                v
            })
            .collect();
        Base3 { table }
    }

    fn to_base3(&self, bit1: usize, bit2: usize) -> usize {
        self.table[bit1] + 2 * self.table[bit2]
    }
}

struct WeightedPattern {
    patterns: Vec<u64>,
    weight: Vec<f64>,
    pattern_starts: Vec<usize>,
    base3_converter: Base3,
}

impl WeightedPattern {
    fn new(patterns: &[u64]) -> WeightedPattern {
        let max_bits = patterns.iter().map(|x| popcnt(*x)).max().unwrap() as usize;
        let conv = Base3::new(max_bits);
        let mut vp3 = Vec::new();
        let mut p3 = 1;
        for _ in 0..=max_bits {
            vp3.push(p3);
            p3 *= 3;
        }
        let mut pattern_starts = Vec::new();
        pattern_starts.push(0);
        let mut offset = 0;
        for pattern in patterns.iter() {
            offset += vp3[popcnt(*pattern) as usize];
            pattern_starts.push(offset);
        }
        // pcnt, ocnt, parity, const
        offset += 4;
        pattern_starts.push(offset);
        WeightedPattern {
            patterns: patterns.to_vec(),
            weight: vec![0.; offset],
            pattern_starts,
            base3_converter: conv,
        }
    }

    fn generate_indices_impl(&self, board: &Board) -> Vec<u32> {
        let mut indices = Vec::new();
        for (idx, pattern) in self.patterns.iter().enumerate() {
            let pbit = pext(board.player, *pattern) as usize;
            let obit = pext(board.opponent, *pattern) as usize;
            let pattern_index = self.base3_converter.to_base3(pbit, obit) + self.pattern_starts[idx];
            indices.push(pattern_index as u32);
        }
        indices
    }

    fn generate_indices(&self, board: &Board) -> Vec<u32> {
        let mut board_rot = *board;
        let mut indices = Vec::with_capacity(self.patterns.len() * 8 + 4);
        for _i in 0..4 {
            indices.extend(self.generate_indices_impl(&board_rot));
            let board_rev = board_rot.reverse_vertical();
            indices.extend(self.generate_indices_impl(&board_rev));
            board_rot = board_rot.rot90();
        }
        indices
    }

    fn train(&mut self, boards: &[Board], scores: &[f64]) {
        let expected_size = boards.len() * (4 + 8 * self.patterns.len());
        let mut row_starts = Vec::with_capacity(boards.len());
        row_starts.push(0);
        let mut mat_weights = Vec::with_capacity(expected_size);
        let mut cols = Vec::with_capacity(expected_size);
        let other_params_offset = self.pattern_starts[self.patterns.len()] as u32;
        // construct matrix
        for board in boards {
            let indices = self.generate_indices(board);
            cols.extend(indices);
            mat_weights.resize(cols.len(), 1.0);
            // pcnt, ocnt, parity, const
            cols.push(other_params_offset + 0);
            mat_weights.push(popcnt(board.mobility_bits()) as f64);
            cols.push(other_params_offset + 1);
            mat_weights.push(popcnt(board.pass_unchecked().mobility_bits()) as f64);
            cols.push(other_params_offset + 2);
            mat_weights.push((popcnt(board.empty()) % 2) as f64);
            cols.push(other_params_offset + 3);
            mat_weights.push(1.0);
            row_starts.push(cols.len());
        }
        // L2 normalization
        let col_size = *self.pattern_starts.last().unwrap();
        for col in 0..col_size {
            cols.push(col as u32);
            mat_weights.push(8.0);
            row_starts.push(cols.len());
        }
        let mut scores_vec: Vec<_> = scores.iter().copied().collect();
        scores_vec.resize(scores_vec.len() + col_size, 0.0);
        let spm = SparseMat::new(mat_weights, col_size, row_starts, cols);
        cgls(&spm, &mut self.weight, &scores_vec, 300);
    }
}

#[allow(dead_code)]
const PATTERNS: [u64; 10] = [
    0x0000_0000_0000_00ff,
    0x0000_0000_0000_ff00,
    0x0000_0000_00ff_0000,
    0x0000_0000_ff00_0000,
    0x0000_0000_0303_0303,
    0x0000_0000_0102_0408,
    0x0000_0001_0204_0810,
    0x0000_0102_0408_1020,
    0x0001_0204_0810_2040,
    0x0102_0408_1020_4080,
];

const PATTERNS_LARGE: [u64; 10] = [
    0x0000_0000_0000_02ff,
    0x0000_0000_0000_ff00,
    0x0000_0000_00ff_0000,
    0x0000_0000_ff00_0000,
    0x0000_0001_0303_0303,
    0x0000_0000_0102_0408,
    0x0000_0001_0204_0810,
    0x0000_0102_0408_1020,
    0x0001_0204_0810_2040,
    0x0102_0408_1020_4080,
];

pub fn train(matches: &ArgMatches) -> Option<()> {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();
    let range_min = *matches.get_one::<i8>("from").unwrap();
    let range_max = *matches.get_one::<i8>("to").unwrap();
    let width = matches.get_one::<i8>("width").unwrap();

    let in_f = File::open(input_path).ok()?;
    let mut reader = BufReader::new(in_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_boards = input_line.trim().parse().unwrap();
    let mut boards = Vec::new();
    let mut scores = Vec::new();
    for _i in 0..num_boards {
        input_line.clear();
        reader.read_line(&mut input_line).unwrap();
        let data: Vec<&str> = input_line.split(' ').collect();
        let player = u64::from_str_radix(&data[0], 16).ok()?;
        let opponent = u64::from_str_radix(&data[1], 16).ok()?;
        boards.push(Board { player, opponent });
        scores.push(data[2].trim().parse().unwrap());
    }
    let mut wp = WeightedPattern::new(&PATTERNS_LARGE);
    for stone_count in range_min..=range_max {
        eprintln!("Stone count = {}", stone_count);
        let stones_min = stone_count - width + 1;
        let stones_max = stone_count + width - 1;
        let mut using_boards = Vec::new();
        let mut using_scores = Vec::new();
        for (&board, &score) in boards.iter().zip(scores.iter()) {
            let stones = popcnt(board.player | board.opponent);
            if stones < stones_min || stones > stones_max {
                continue;
            }
            using_boards.push(board);
            using_scores.push(score);
        }
        wp.train(&using_boards, &using_scores);

        let out_f = File::create(format!("{}_{}.txt", output_path, stone_count)).ok()?;
        let mut writer = BufWriter::new(out_f);

        write!(&mut writer, "{}\n", wp.weight.len()).ok()?;
        for &w in &wp.weight {
            write!(&mut writer, "{}\n", w).ok()?;
        }
    }
    Some(())
}

pub fn binarize_weights(matches: &ArgMatches) {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

    let in_f = File::open(input_path).unwrap();
    let mut reader = BufReader::new(in_f);

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_weight = input_line.trim().parse().unwrap();

    for _i in 0..num_weight {
        input_line.clear();
        reader.read_line(&mut input_line).unwrap();
        let weight = input_line.trim().parse::<f64>().unwrap();
        writer.write(&weight.to_le_bytes()).unwrap();
    }
}

pub fn pack_weights(matches: &ArgMatches) {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

    let in_f = File::open(input_path).unwrap();
    let mut reader = BufReader::new(in_f);

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_weight = input_line.trim().parse().unwrap();
    let scale = PACKED_SCALE as f64;

    let mut weights = Vec::new();
    for _i in 0..num_weight {
        input_line.clear();
        reader.read_line(&mut input_line).unwrap();
        let weight = input_line.trim().parse::<f64>().unwrap();
        let weight_scaled = (weight * scale).round() as i16;
        weights.push(weight_scaled);
    }

    let orig_len = weights.len();
    let mut compressed = compress(&weights);

    write!(&mut writer, "{}\n", orig_len).unwrap();

    while compressed.len() % 15 != 0 {
        compressed.push(false);
    }
    for chunk in compressed.chunks(15) {
        let mut bits = 0u32;
        for (idx, &bit) in chunk.iter().enumerate() {
            if bit {
                bits |= 1 << idx;
            }
        }
        let c = encode_utf16(bits).unwrap();
        write!(&mut writer, "{}", c).unwrap();
    }
}

pub fn eval_stats(matches: &ArgMatches) -> Option<()> {
    let input_path = matches.get_one::<String>("INPUT").unwrap();

    let in_f = File::open(input_path).ok()?;
    let mut reader = BufReader::new(in_f);

    eprintln!("Loading...");
    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_boards = input_line.trim().parse().unwrap();
    let mut dataset = HashSet::new();
    for _i in 0..num_boards {
        input_line.clear();
        reader.read_line(&mut input_line).unwrap();
        let data: Vec<&str> = input_line.split(' ').collect();
        let player = u64::from_str_radix(&data[0], 16).ok()?;
        let opponent = u64::from_str_radix(&data[1], 16).ok()?;
        dataset.insert((
            Board { player, opponent },
            data[2].trim().parse::<i16>().unwrap(),
        ));
    }

    let dataset: Vec<_> = dataset.into_iter().take(8192).collect();

    eprintln!("Computing...");
    let eval_cache = Arc::new(EvalCacheTable::new(256, 65536));
    let evaluator = Arc::new(Evaluator::new("table-single"));
    let depth_max = 8;
    let scores: Vec<_> = dataset
        .par_iter()
        .map(|&(board, _)| {
            let mut scores = Vec::new();
            let eval_cache = eval_cache.clone();
            let mut searcher = Searcher {
                evaluator: evaluator.clone(),
                cache: eval_cache.clone(),
                timer: None,
                node_count: 0,
            };
            for depth in 1..=depth_max {
                eval_cache.inc_gen();
                if let Some((evaluated, _)) = searcher.think(
                    board,
                    EVAL_SCORE_MIN,
                    EVAL_SCORE_MAX,
                    false,
                    depth as i32 * DEPTH_SCALE,
                ) {
                    scores.push(evaluated);
                } else {
                    panic!()
                }
            }
            scores
        })
        .collect();
    let mut diff = vec![vec![Vec::new(); depth_max]; depth_max];
    for vs in scores.iter() {
        for (i, s1) in vs.iter().enumerate() {
            for (j, s2) in vs.iter().enumerate() {
                if i >= j {
                    continue;
                }
                diff[i][j].push((s1 - s2).abs());
            }
        }
    }
    for (i, sub) in diff.into_iter().enumerate() {
        for (j, mut vd) in sub.into_iter().enumerate() {
            if i >= j {
                continue;
            }
            let total = vd.len();
            if total == 0 {
                continue;
            }
            vd.sort_unstable();
            println!("{} {}", i, j);
            for idx in 1..=15 {
                let ratio = 1.0 - 0.7f32.powi(idx);
                let index = (total as f32 * ratio) as usize;
                println!("{} {}", ratio, vd[index] as f32 / SCALE as f32);
            }
        }
    }
    Some(())
}
