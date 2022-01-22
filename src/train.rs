use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::playout::*;
use crate::search::*;
use crate::serialize::*;
use crate::table::*;
use crate::think::*;
use clap::ArgMatches;
use futures::executor;
use futures::executor::ThreadPool;
use futures::task::SpawnExt;
use rayon::prelude::*;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::str;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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

pub fn collect_boards(record: &[usize]) -> Option<Vec<Board>> {
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

pub fn load_records(input_path: &str) -> Vec<Vec<Board>> {
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

pub fn pos_to_str(pos: usize) -> String {
    let row = pos / 8;
    let col = pos % 8;
    let first = (col as u8) + ('A' as u8);
    let second = (row as u8) + ('1' as u8);
    let mut result = String::new();
    result.push(first as char);
    result.push(second as char);
    result
}

pub async fn create_record_by_solve(mut board: Board, solve_obj: &mut SolveObj) -> (String, Board) {
    let mut result = String::new();
    while !board.is_gameover() {
        let pos = solve_with_move(board, solve_obj).await;
        if let Hand::Play(pos) = pos {
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
    let finished = Arc::new(AtomicUsize::new(0));
    for _i in 0..num_records {
        let mut solve_obj = SolveObj::new(
            res_cache.clone(),
            eval_cache.clone(),
            evaluator.clone(),
            search_params.clone(),
            pool.clone(),
        );
        let mut input_line = String::new();
        reader.read_line(&mut input_line).unwrap();
        let finished = finished.clone();

        handles.push(
            pool.spawn_with_handle(async move {
                let record = parse_record(&input_line);
                let mut updateds = Vec::new();
                for (idx, _) in record.iter().enumerate() {
                    let sub_record = record.clone();
                    let sub_record = &sub_record[0..=idx];
                    //let updated = update_record_impl(&record, &mut solve_obj, depth).await;
                    let updated = playout(sub_record, &mut solve_obj, 100, depth).await;
                    let count = finished.fetch_add(1, Ordering::SeqCst);
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

    let mut result = Vec::new();
    for handle in handles {
        let results = executor::block_on(handle);
        for (line, score) in results.iter().flatten() {
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
        let boards_with_results_next = Arc::new(Mutex::new(HashMap::new()));
        boards.par_iter().for_each(|&board| {
            let mut current = board;
            let mut mobility = current.mobility();
            let is_pass = mobility.is_empty();
            if is_pass {
                current = current.pass();
                mobility = current.mobility();
                if mobility.is_empty() {
                    boards_with_results_next
                        .lock()
                        .unwrap()
                        .insert(board, (board.score(), PASS));
                    return;
                }
            }
            let mut best_score = None;
            let mut best_pos = None;
            for pos in mobility {
                let next = current.play(pos).unwrap();
                for next_sym in next.sym_boards() {
                    if let Some((score, _)) = boards_with_results.get(&next_sym) {
                        best_score = Some(max(-score, best_score.unwrap_or(-64)));
                        best_pos = Some(pos);
                    }
                }
            }
            if is_pass {
                boards_with_results_next
                    .lock()
                    .unwrap()
                    .insert(board, (-best_score.unwrap(), best_pos.unwrap()));
            } else {
                boards_with_results_next
                    .lock()
                    .unwrap()
                    .insert(current, (best_score.unwrap(), best_pos.unwrap()));
            }
        });
        for (&k, v) in boards_with_results_next.lock().unwrap().iter() {
            boards_with_results.insert(k, v.clone());
        }
    }

    eprintln!("Remaining board count = {}", boards_with_results.len());

    eprintln!("Writing to file...");
    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    write!(
        &mut writer,
        "{}\n",
        min(boards_with_results.len(), max_output)
    )
    .unwrap();
    for (idx, (board, (score, pos))) in boards_with_results.iter().enumerate() {
        if idx >= max_output {
            break;
        }
        write!(
            &mut writer,
            "{:016x} {:016x} {} {}\n",
            board.player, board.opponent, score, pos,
        )
        .unwrap();
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

struct SparseMat {
    weight: Vec<f64>,
    col_size: usize,
    row_starts: Vec<usize>,
    cols: Vec<u32>,
}

impl SparseMat {
    fn row_size(&self) -> usize {
        self.row_starts.len() - 1
    }

    #[allow(dead_code)]
    fn transpose(&self) -> SparseMat {
        let mut weight_t = vec![Vec::new(); self.col_size];
        let mut cols_t = vec![Vec::new(); self.col_size];

        for row in 0..self.row_size() {
            let row_start = self.row_starts[row];
            let row_end = self.row_starts[row + 1];
            for (&col, &w) in self.cols[row_start..row_end]
                .iter()
                .zip(&self.weight[row_start..row_end])
            {
                cols_t[col as usize].push(row as u32);
                weight_t[col as usize].push(w);
            }
        }
        let mut row_starts_t = Vec::new();
        let mut offset = 0;
        for col_t in &cols_t {
            row_starts_t.push(offset);
            offset += col_t.len();
        }
        row_starts_t.push(offset);
        SparseMat {
            weight: weight_t.into_iter().flatten().collect(),
            col_size: self.row_size(),
            row_starts: row_starts_t,
            cols: cols_t.into_iter().flatten().collect(),
        }
    }

    // y = A*x
    fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        y.par_iter_mut().enumerate().for_each(|(row, elem)| {
            *elem = unsafe {
                let row_start = *self.row_starts.get_unchecked(row);
                let row_end = *self.row_starts.get_unchecked(row + 1);
                let mut ans = 0.;
                for (col, w) in self.cols[row_start..row_end]
                    .iter()
                    .zip(&self.weight[row_start..row_end])
                {
                    ans += w * x.get_unchecked(*col as usize);
                }
                ans
            };
        });
    }

    // x = A^t*y
    // Unparallelized
    fn mul_vec_transposed_naive(&self, y: &[f64], row_offset: usize, x: &mut [f64]) {
        for e in x.iter_mut() {
            *e = 0.;
        }
        unsafe {
            for (row_local, elem) in y.iter().enumerate() {
                let row = row_local + row_offset;
                let row_start = *self.row_starts.get_unchecked(row);
                let row_end = *self.row_starts.get_unchecked(row + 1);
                let val = *elem;
                for (col, w) in self.cols[row_start..row_end]
                    .iter()
                    .zip(&self.weight[row_start..row_end])
                {
                    *x.get_unchecked_mut(*col as usize) += w * val;
                }
            }
        }
    }

    fn mul_vec_transposed_(&self, y: &[f64], row_offset: usize, x: &mut [f64]) {
        if y.len() < 65536 {
            self.mul_vec_transposed_naive(y, row_offset, x);
        } else {
            let mid = y.len() / 2;
            let mut another_x = x.to_vec();
            rayon::join(
                || self.mul_vec_transposed_(&y[..mid], row_offset, x),
                || self.mul_vec_transposed_(&y[mid..], row_offset + mid, &mut another_x),
            );
            for (elem, another_elem) in x.iter_mut().zip(&another_x) {
                *elem += another_elem;
            }
        }
    }

    fn mul_vec_transposed(&self, y: &[f64], x: &mut [f64]) {
        self.mul_vec_transposed_(y, 0, x);
    }
}

fn norm(x: &[f64]) -> f64 {
    let mut ans = 0.;
    for a in x {
        ans += a * a;
    }
    ans
}

fn cgls(spm: &SparseMat, a: &mut [f64], b: &[f64], iter_num: usize) {
    let mut pa = vec![0.; spm.row_size()];
    spm.mul_vec(&a, &mut pa);
    let mut r = vec![0.; spm.row_size()];
    for i in 0..spm.row_size() {
        r[i] = b[i] - pa[i];
    }
    let mut p = vec![0.; spm.col_size];
    //let spm_t = spm.transpose();
    spm.mul_vec_transposed(&r, &mut p);
    let mut s = p.clone();
    let mut old_s_norm = norm(&s);
    let mut q = vec![0.; spm.row_size()];
    let mut d = vec![0.; spm.row_size()];
    for i in 0..iter_num {
        spm.mul_vec(&p, &mut q);
        let alpha = old_s_norm / norm(&q);
        for idx in 0..spm.col_size {
            a[idx] += alpha * p[idx];
        }
        for idx in 0..spm.row_size() {
            r[idx] -= alpha * q[idx];
        }
        spm.mul_vec_transposed(&r, &mut s);
        let new_s_norm = norm(&s);
        if i % 10 == 0 {
            spm.mul_vec(&a, &mut pa);
            let except_l2_norm_len = spm.row_size() - spm.col_size;
            for j in 0..except_l2_norm_len {
                d[j] = b[j] - pa[j];
            }
            eprintln!(
                "Step: {}, CGLS Diff: {}",
                i,
                (norm(&d[0..except_l2_norm_len]) / except_l2_norm_len as f64).sqrt()
            );
        }
        if new_s_norm < 1.0 {
            break;
        }
        let beta = new_s_norm / old_s_norm;
        for idx in 0..spm.col_size {
            p[idx] = s[idx] + beta * p[idx];
        }
        old_s_norm = new_s_norm;
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
            let pattern_index =
                self.base3_converter.to_base3(pbit, obit) + self.pattern_starts[idx];
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
            mat_weights.push(popcnt(board.pass().mobility_bits()) as f64);
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
        let spm = SparseMat {
            weight: mat_weights,
            col_size,
            row_starts,
            cols,
        };
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
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();
    let range_min = matches.value_of("from").unwrap().parse::<i8>().unwrap();
    let range_max = matches.value_of("to").unwrap().parse::<i8>().unwrap();
    let width = matches.value_of("width").unwrap().parse::<i8>().unwrap();

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
        boards.push(Board {
            player,
            opponent,
            is_black: true, // dummy
        });
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
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

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
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

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
    let input_path = matches.value_of("INPUT").unwrap();

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
            Board {
                player,
                opponent,
                is_black: true, // dummy
            },
            data[2].trim().parse::<i16>().unwrap(),
        ));
    }

    let dataset: Vec<_> = dataset.into_iter().take(8192).collect();

    eprintln!("Computing...");
    let eval_cache = EvalCacheTable::new(256, 65536);
    let evaluator = Arc::new(Evaluator::new("table-single"));
    let depth_max = 8;
    let scores: Vec<_> = dataset
        .par_iter()
        .map(|&(board, _)| {
            let mut scores = Vec::new();
            let mut eval_cache = eval_cache.clone();
            let mut searcher = Searcher {
                evaluator: evaluator.clone(),
                cache: eval_cache.clone(),
                timer: None,
                node_count: 0,
            };
            for depth in 1..=depth_max {
                eval_cache.inc_gen();
                if let Some((evaluated, _)) =
                    searcher.think(board, -64 * SCALE, 64 * SCALE, false, depth as i8)
                {
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
