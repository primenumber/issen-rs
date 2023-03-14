use crate::engine::bits::*;
use crate::engine::board::*;
use std::cmp::max;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::ops::RangeInclusive;
use std::path::Path;
use yaml_rust::yaml;

struct EvaluatorConfig {
    masks: Vec<String>,
    stones_range: RangeInclusive<i8>,
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
        let from = stones_range_yaml["from"].as_i64()? as i8;
        let to = stones_range_yaml["to"].as_i64()? as i8;
        let stones_range = from..=to;
        Some(EvaluatorConfig {
            masks,
            stones_range,
        })
    }
}

struct Parameters {
    pattern_weights: Vec<i16>,
    patterns: Vec<u64>,
    offsets: Vec<usize>,
    p_mobility_score: i16,
    o_mobility_score: i16,
    parity_score: i16,
    constant_score: i16,
}

impl Parameters {
    fn permute_indices(pattern: u64) -> Vec<Vec<usize>> {
        let pattern_size = pattern.count_ones();
        let table_size = 1 << pattern_size;
        let mut indices = vec![Vec::new(); 8];
        for i in 0..table_size {
            let mut t_pattern = pattern;
            let mut t_bits = pdep(i as u64, t_pattern);
            for dir in 0..4 {
                indices[dir * 2].push(pext(t_bits, t_pattern) as usize);
                indices[dir * 2 + 1].push(pext(flip_diag(t_bits), flip_diag(t_pattern)) as usize);
                t_bits = rot90(t_bits);
                t_pattern = rot90(t_pattern);
            }
        }
        indices
    }
    fn expand_weights_by_d4(
        weights: &[i16],
        pattern: u64,
        b3conv: &[usize],
    ) -> Vec<(u64, Vec<i16>)> {
        let mut permuted_weights = vec![vec![0; weights.len()]; 8];
        let perm = Self::permute_indices(pattern);
        let pattern_size = pattern.count_ones();
        for pidx in 0..(1 << pattern_size) {
            for oidx in 0..(1 << pattern_size) {
                if (pidx & oidx) != 0 {
                    continue;
                }
                let orig_index = b3conv[pidx] + b3conv[oidx] * 2;
                for i in 0..8 {
                    let t_pidx = perm[i][pidx];
                    let t_oidx = perm[i][oidx];
                    let index = b3conv[t_pidx] + b3conv[t_oidx] * 2;
                    permuted_weights[i][index] = weights[orig_index];
                }
            }
        }
        let mut patterns_by_d4 = Vec::new();
        let mut pattern_bits = pattern;
        for _ in 0..4 {
            patterns_by_d4.push(pattern_bits);
            patterns_by_d4.push(flip_diag(pattern_bits));
            pattern_bits = rot90(pattern_bits);
        }
        patterns_by_d4
            .into_iter()
            .zip(permuted_weights.into_iter())
            .collect()
    }

    fn new(
        orig_weights: &[i16],
        orig_offsets: &[usize],
        orig_patterns: &[u64],
        b3conv: &[usize],
    ) -> Option<Parameters> {
        let mut v: Vec<(u64, Vec<i16>)> = Vec::new();
        for (i, &pattern) in orig_patterns.iter().enumerate() {
            let expanded = Self::expand_weights_by_d4(
                &orig_weights[orig_offsets[i]..orig_offsets[i + 1]],
                pattern,
                b3conv,
            );
            for (t_pattern, t_weights) in expanded {
                if let Some((_, vw)) = v.iter_mut().find(|(p, _)| *p == t_pattern) {
                    for (w, &e) in vw.into_iter().zip(t_weights.iter()) {
                        *w += e;
                    }
                } else {
                    v.push((t_pattern, t_weights));
                }
            }
        }
        let mut pattern_weights = Vec::new();
        let mut patterns = Vec::new();
        let mut offsets = Vec::new();
        let mut offset = 0;
        for (pattern, ex_weights) in v {
            offsets.push(offset);
            offset += ex_weights.len();
            pattern_weights.extend(ex_weights);
            patterns.push(pattern);
        }
        let non_patterns_offset = *orig_offsets.last().unwrap();
        Some(Parameters {
            pattern_weights,
            patterns,
            offsets,
            p_mobility_score: orig_weights[non_patterns_offset],
            o_mobility_score: orig_weights[non_patterns_offset + 1],
            parity_score: orig_weights[non_patterns_offset + 2],
            constant_score: orig_weights[non_patterns_offset + 3],
        })
    }

    fn fold_parity(&mut self, stone_count: i8) {
        if stone_count % 2 == 1 {
            self.constant_score += self.parity_score;
        }
        self.parity_score = 0;
    }

    fn eval(&self, board: Board, b3conv: &[usize]) -> i16 {
        let mut result = self.constant_score;
        for (i, &pattern) in self.patterns.iter().enumerate() {
            let pidx = pext(board.player, pattern) as usize;
            let oidx = pext(board.opponent, pattern) as usize;
            let windex = b3conv[pidx] + b3conv[oidx] * 2;
            result += self.pattern_weights[windex + self.offsets[i]];
        }
        result += self.p_mobility_score * popcnt(board.mobility_bits()) as i16;
        result += self.o_mobility_score * popcnt(board.pass().mobility_bits()) as i16;
        result
    }
}

pub struct Evaluator {
    stones_range: RangeInclusive<i8>,
    params: Vec<Parameters>,
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
pub const EVAL_SCORE_MAX: i16 = BOARD_SIZE as i16 * SCALE;
pub const EVAL_SCORE_MIN: i16 = -EVAL_SCORE_MAX;

impl Evaluator {
    fn load_weight(path: &Path, length: usize) -> Option<Vec<i16>> {
        let mut value_file = File::open(path).ok()?;
        let mut buf = vec![0u8; length * 8];
        value_file.read(&mut buf).ok()?;
        let mut v = Vec::with_capacity(length);
        for i in 0usize..length {
            let mut ary: [u8; 8] = Default::default();
            ary.copy_from_slice(&buf[(8 * i)..(8 * (i + 1))]);
            let raw_weight = unsafe { mem::transmute::<[u8; 8], f64>(ary) };
            v.push(
                (SCALE as f64 * raw_weight)
                    .max(SCALE as f64 * -64.0)
                    .min(SCALE as f64 * 64.0)
                    .round() as i16,
            );
        }
        Some(v)
    }

    fn smooth_weight(weights: &[Vec<i16>], length: usize, window_size: i8) -> Vec<Vec<i16>> {
        assert_eq!(window_size % 2, 1);
        let half_window = window_size / 2;
        let mut result = vec![vec![0; length]; weights.len()];
        for (i, w) in result.iter_mut().enumerate() {
            for (k, e) in w.iter_mut().enumerate() {
                for d in -half_window..=half_window {
                    let j = i as isize + d as isize;
                    let j = if j < 0 { 0 } else { j as usize };
                    let j = if j >= weights.len() {
                        weights.len() - 1
                    } else {
                        j
                    };
                    *e += weights[j][k];
                }
                *e /= window_size as i16;
            }
        }
        result
    }

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

        let mut weights = Vec::new();
        for num in config.stones_range.clone() {
            let path = table_path.join(format!("value{}", num));
            weights.push(Self::load_weight(&path, length).unwrap());
        }

        let smoothed_weights = Self::smooth_weight(&weights, length, 1);

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

        let params = smoothed_weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let mut param = Parameters::new(w, &offsets, &patterns, &base3).unwrap();
                param.fold_parity(i as i8 + config.stones_range.start());
                param
            })
            .collect();

        Evaluator {
            stones_range: config.stones_range,
            params,
            base3,
        }
    }

    fn smooth_val(raw_score: i32) -> i16 {
        let scale32 = SCALE as i32;
        (if raw_score > 63 * scale32 {
            (BOARD_SIZE as i32) * scale32 - scale32 * scale32 / (raw_score - 62 * scale32)
        } else if raw_score < -63 * scale32 {
            -(BOARD_SIZE as i32) * scale32 - scale32 * scale32 / (raw_score + 62 * scale32)
        } else {
            raw_score
        }) as i16
    }

    pub fn eval(&self, board: Board) -> i16 {
        let rem: usize = popcnt(board.empty()) as usize;
        let stones = (BOARD_SIZE - rem)
            .max(*self.stones_range.start() as usize)
            .min(*self.stones_range.end() as usize);
        let index = stones - *self.stones_range.start() as usize;
        let score = self.params[index].eval(board, &self.base3) as i32;
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
            assert!(smoothed > EVAL_SCORE_MIN);
            assert!(smoothed < EVAL_SCORE_MAX);
        }
    }
}
