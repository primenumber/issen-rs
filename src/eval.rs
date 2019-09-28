
pub struct Evaluator {
    weights: Vec<Vec<i16>>,
    offsets: Vec<usize>,
    patterns: Vec<u64>,
    base3: Vec<usize>
}

fn pow3(x: i8) -> usize {
    if x == 0 {
        1
    } else {
        3 * pow3(x-1)
    }
}

pub const SCALE: i16 = 128;

use std::mem;
use std::cmp::max;
use std::io::{BufRead,BufReader,Read};
use std::fs::File;
use std::str::FromStr;
use crate::bits::*;
use crate::board::*;

impl Evaluator {
    pub fn new(subboard_filename: &str) -> Evaluator {
        let subboard_file = File::open(subboard_filename).unwrap();
        let subboard_reader = BufReader::new(subboard_file);
        let mut patterns = Vec::new();
        let mut offsets = Vec::new();
        let mut pattern_bits = 0;
        let mut length: usize = 0;
        let mut max_bits = 0;
        for (i, line) in subboard_reader.lines().enumerate() {
            if i == 0 {
                let _count = usize::from_str(&line.unwrap()).unwrap();
            } else if (i % 9) != 0 {
                let lane = (i % 9) - 1;
                let bits = u64::from_str_radix(&line.unwrap(), 2).unwrap();
                pattern_bits |= mirror_under_8(bits) << (lane * 8);
                if (i % 9) == 8 {
                    patterns.push(pattern_bits);
                    offsets.push(length);
                    length += pow3(popcnt(pattern_bits));
                    max_bits = max(max_bits, popcnt(pattern_bits));
                }
            } else {
                pattern_bits = 0;
            }
        }
        length += 1;

        let files: Vec<String> = (30..61).map(|i| format!("table/value{}", i)).collect();
        let mut weights = vec![vec![0i16; length]; files.len()];
        for (cnt, filename) in files.iter().enumerate() {
            let mut value_file = File::open(filename).unwrap();
            let mut buf = vec![0u8; length * 8];
            value_file.read(&mut buf).unwrap();
            for i in 0usize..length {
                let mut ary: [u8; 8] = Default::default();
                ary.copy_from_slice(&buf[(8*i)..(8*(i+1))]);
                weights[cnt][i] = (SCALE as f64 * unsafe { mem::transmute::<[u8; 8], f64>(ary) }).max(SCALE as f64 * -64.0).min(SCALE as f64 * 64.0).round() as i16;
            }
        }

        let mut base3 = vec![0; 1<<max_bits];
        for i in 0usize..(1usize<<max_bits) {
            let mut sum = 0;
            for j in 0..max_bits {
                if ((i >> j) & 1) != 0 {
                    sum += pow3(j);
                }
            }
            base3[i] = sum;
        }
        Evaluator { weights, offsets, patterns, base3 }
    }

    fn eval_impl(&self, board: Board, index: usize) -> i32 {
        let mut score = 0i32;
        for (i, pattern) in self.patterns.iter().enumerate() {
            let player_pattern = pext(board.player, *pattern) as usize;
            let opponent_pattern = pext(board.opponent, *pattern) as usize;
            score += self.weights[index][
                self.offsets[i] + self.base3[player_pattern]
                + 2*self.base3[opponent_pattern]] as i32;
        }
        score
    }

    pub fn eval(&self, mut board: Board) -> i16 {
        let mut score = 0i32;
        let rem: usize = popcnt(board.empty()) as usize;
        let index = (34 - rem).max(0).min(30);
        for _i in 0..4 {
            score += self.eval_impl(board.clone(), index);
            score += self.eval_impl(board.flip_diag(), index);
            board = board.rot90();
        }
        let raw_score = score + *self.weights[index].last().unwrap() as i32;
        let scale32 = SCALE as i32;
        (if raw_score > 63 * scale32 {
            64 * scale32 - scale32 * scale32 / (raw_score - 62 * scale32)
        } else if raw_score < -63 * scale32 {
            -64 * scale32 - scale32 * scale32 / (raw_score + 62 * scale32)
        } else {
            raw_score
        }) as i16
    }
}
