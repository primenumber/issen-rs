use crate::bits::*;
use crate::board::*;
use std::cmp::max;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::ops::RangeInclusive;
use std::path::Path;
use yaml_rust::yaml;

struct EvaluatorConfig {
    masks: Vec<String>,
    stones_range: RangeInclusive<usize>,
}

impl EvaluatorConfig {
    fn new(config_path: &Path) -> Option<EvaluatorConfig> {
        let mut config_file = File::open(&config_path).ok()?;
        let mut config_string = String::new();
        config_file.read_to_string(&mut config_string).ok()?;
        let config_objs = yaml::YamlLoader::load_from_str(&config_string).ok()?;
        let config = &config_objs[0];
        let masks = config["masks"]
            .as_vec()?
            .iter()
            .map(|e| e.as_str().unwrap().to_string())
            .collect();
        let stones_range_yaml = &config["stone_counts"];
        let from = stones_range_yaml["from"].as_i64()? as usize;
        let to = stones_range_yaml["to"].as_i64()? as usize;
        let stones_range = from..=to;
        Some(EvaluatorConfig {
            masks,
            stones_range,
        })
    }
}

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

pub const SCALE: i16 = 256;

impl Evaluator {
    pub fn new(table_dirname: &str) -> Evaluator {
        let table_path = Path::new(table_dirname);
        let config_path = table_path.join("config.yaml");
        let config = EvaluatorConfig::new(&config_path).unwrap();
        let mut patterns = Vec::new();
        let mut offsets = Vec::new();
        let mut length: usize = 0;
        let mut max_bits = 0;
        for pattern_str in config.masks.iter() {
            let bits = u64::from_str_radix(&pattern_str, 2).unwrap();
            patterns.push(bits);
            offsets.push(length);
            length += pow3(popcnt(bits));
            max_bits = max(max_bits, popcnt(bits));
        }
        offsets.push(length);
        length += 4;

        let from = *config.stones_range.start();
        let to = *config.stones_range.end();
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
                for diff in 0..=0 {
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
            unsafe {
                score += *self.weights.get_unchecked(index).get_unchecked(
                    self.offsets.get_unchecked(i)
                        + self.base3.get_unchecked(player_pattern)
                        + 2 * self.base3.get_unchecked(opponent_pattern),
                ) as i32;
            }
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
            score += self.eval_impl(board, index);
            score += self.eval_impl(board.flip_diag(), index);
            board = board.rot90();
        }
        let non_patterns_offset = self.offsets.last().unwrap();
        score += popcnt(board.mobility_bits()) as i32
            * self.weights[index][non_patterns_offset + 0] as i32;
        score += popcnt(board.pass().mobility_bits()) as i32
            * self.weights[index][non_patterns_offset + 1] as i32;
        if rem % 2 == 1 {
            score += self.weights[index][non_patterns_offset + 2] as i32;
        }
        score += self.weights[index][non_patterns_offset + 3] as i32;
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
