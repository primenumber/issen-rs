use crate::bits::*;
use crate::board::*;
use std::cmp::max;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::ops::RangeInclusive;
use std::path::Path;
use yaml_rust::yaml;

pub struct Evaluator {
    stones_range: RangeInclusive<usize>,
    weights: Vec<Vec<i16>>,
    offsets: Vec<usize>,
    patterns: Vec<u64>,
    base3: Vec<usize>,
}

fn pow3(x: i8) -> usize {
    if x == 0 {
        1
    } else {
        3 * pow3(x - 1)
    }
}

pub const SCALE: i16 = 128;

impl Evaluator {
    pub fn new(table_dirname: &str) -> Evaluator {
        let table_path = Path::new(table_dirname);
        let mut config_file = File::open(table_path.join("config.yaml")).unwrap();
        let mut config_string = String::new();
        config_file.read_to_string(&mut config_string).unwrap();
        let config_objs = yaml::YamlLoader::load_from_str(&config_string).unwrap();
        let config = &config_objs[0]; // first document of the file
        let mut patterns = Vec::new();
        let mut offsets = Vec::new();
        let mut length: usize = 0;
        let mut max_bits = 0;
        let masks = &config["masks"];
        for pattern_obj in masks.clone() {
            let pattern_str = pattern_obj.as_str().unwrap();
            //let bits = u64::from_str_radix(&pattern_str, 2).unwrap();
            let bits =
                flip_horizontal(flip_vertical(u64::from_str_radix(&pattern_str, 2).unwrap()));
            patterns.push(bits);
            offsets.push(length);
            length += pow3(popcnt(bits));
            max_bits = max(max_bits, popcnt(bits));
        }
        offsets.push(length);
        //length += 4;
        length += 1;

        let from = config["stone_counts"]["from"].as_i64().unwrap() as usize;
        let to = config["stone_counts"]["to"].as_i64().unwrap() as usize;
        let stones_range = from..=to;
        let range_size = to - from + 1;
        let mut weights = vec![vec![0i16; length]; range_size];
        for num in stones_range.clone() {
            let mut value_file = File::open(table_path.join(format!("value{}", num))).unwrap();
            let mut buf = vec![0u8; length * 8];
            value_file.read(&mut buf).unwrap();
            for i in 0usize..length {
                let mut ary: [u8; 8] = Default::default();
                ary.copy_from_slice(&buf[(8 * i)..(8 * (i + 1))]);
                let raw_weight = unsafe { mem::transmute::<[u8; 8], f64>(ary) };
                weights[num - from][i] = (SCALE as f64 * raw_weight)
                    .max(SCALE as f64 * -64.0)
                    .min(SCALE as f64 * 64.0)
                    .round() as i16;
            }
        }

        let mut smoothed_weights = vec![vec![0i16; length]; range_size];
        for count_index in 0..range_size {
            for pattern_index in 0..length {
                let mut w = Vec::new();
                for diff in -2..=2 {
                    let ref_count_index = count_index as isize + diff;
                    if ref_count_index < 0 || ref_count_index >= range_size as isize {
                        continue;
                    }
                    w.push(weights[ref_count_index as usize][pattern_index]);
                }
                w.sort_unstable();
                smoothed_weights[count_index][pattern_index] = w[w.len() / 2];
            }
        }

        let mut base3 = vec![0; 1 << max_bits];
        for i in 0usize..(1usize << max_bits) {
            let mut sum = 0;
            for j in 0..max_bits {
                if ((i >> j) & 1) != 0 {
                    sum += pow3(j);
                }
            }
            base3[i] = sum;
        }
        Evaluator {
            stones_range,
            weights: smoothed_weights,
            offsets,
            patterns,
            base3,
        }
    }

    fn eval_impl(&self, board: Board, index: usize) -> i32 {
        let mut score = 0i32;
        for (i, pattern) in self.patterns.iter().enumerate() {
            let player_pattern = pext(board.player, *pattern) as usize;
            let opponent_pattern = pext(board.opponent, *pattern) as usize;
            score += self.weights[index]
                [self.offsets[i] + self.base3[player_pattern] + 2 * self.base3[opponent_pattern]]
                as i32;
        }
        score
    }

    fn smooth_val(raw_score: i32) -> i16 {
        let scale32 = SCALE as i32;
        (if raw_score > 63 * scale32 {
            64 * scale32 - scale32 * scale32 / (raw_score - 62 * scale32)
        } else if raw_score < -63 * scale32 {
            -64 * scale32 - scale32 * scale32 / (raw_score + 62 * scale32)
        } else {
            raw_score
        }) as i16
    }

    pub fn eval(&self, mut board: Board) -> i16 {
        let mut score = 0i32;
        let rem: usize = popcnt(board.empty()) as usize;
        let stones = (64 - rem)
            .max(*self.stones_range.start())
            .min(*self.stones_range.end());
        let index = stones - self.stones_range.start();
        for _i in 0..4 {
            score += self.eval_impl(board.clone(), index);
            score += self.eval_impl(board.flip_diag(), index);
            board = board.rot90();
        }
        let non_patterns_offset = self.offsets.last().unwrap();
        //score += popcnt(board.mobility_bits()) as i32
        //    * self.weights[index][non_patterns_offset + 0] as i32;
        //score += popcnt(board.pass().mobility_bits()) as i32
        //    * self.weights[index][non_patterns_offset + 1] as i32;
        //if rem % 2 == 1 {
        //    score += self.weights[index][non_patterns_offset + 2] as i32;
        //}
        //score += self.weights[index][non_patterns_offset + 3] as i32;
        score += self.weights[index][non_patterns_offset + 0] as i32;
        Self::smooth_val(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth() {
        for raw in -10000..=10000 {
            let smoothed = Evaluator::smooth_val(raw);
            assert!(smoothed > -64 * SCALE);
            assert!(smoothed < 64 * SCALE);
        }
    }
}
