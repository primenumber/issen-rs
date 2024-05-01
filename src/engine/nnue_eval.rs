use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem;
use std::path::Path;

#[derive(Deserialize, Debug)]
struct NNUEConfig {
    #[serde(with = "pattern_format")]
    patterns: Vec<u64>,
    front: usize,
    middle: usize,
    back: usize,
}

mod pattern_format {
    use serde::{Deserialize, Deserializer};
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Vec::<String>::deserialize(deserializer)?;
        v.into_iter()
            .map(|s| u64::from_str_radix(&s, 16))
            .try_collect()
            .map_err(|e| serde::de::Error::custom(format!("Failed to parse pattern: {:?}", e)))
    }
}

impl NNUEConfig {
    fn from_file(path: &Path) -> Option<NNUEConfig> {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        Some(serde_json::from_reader(reader).unwrap())
    }
}

pub struct NNUEEvaluator {
    config: NNUEConfig,
    embedding: Vec<f32>,
    offsets: Vec<usize>,
    #[allow(dead_code)]
    pos_to_indices: Vec<[usize; 64]>,
    layer1_weight: Vec<f32>,
    layer1_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: Vec<f32>,
    layer3_weight: Vec<f32>,
    layer3_bias: Vec<f32>,
}

impl NNUEEvaluator {
    fn load_param(path: &Path, length: usize) -> Option<Vec<f32>> {
        let mut value_file = File::open(path).ok()?;
        let mut buf = vec![0u8; length * 4];
        value_file.read_exact(&mut buf).ok()?;
        let mut v = Vec::with_capacity(length);
        for ary in buf.as_chunks().0 {
            let raw_weight = unsafe { mem::transmute::<[u8; 4], f32>(*ary) };
            v.push(raw_weight);
        }
        Some(v)
    }

    fn normalize_vec(v: &mut [f32], max_norm: f32) {
        let mut sum = 0.;
        for e in v.iter() {
            sum += *e * *e;
        }
        if sum > max_norm {
            let scale = (max_norm / sum).sqrt();
            for e in v {
                *e *= scale;
            }
        }
    }

    fn generate_pos_to_indices(pattern: u64) -> [usize; 64] {
        let mut count = 0;
        let mut result = [0; 64];
        for pos in 0..64 {
            if (pattern >> pos) & 1 == 1 {
                result[pos] = pow3(count);
                count += 1;
            }
        }
        result
    }

    fn transpose_mat(v: &mut [f32], row: usize, col: usize) {
        let mut tmp = vec![0.; row * col];
        for i in 0..row {
            for j in 0..col {
                tmp[i * col + j] = v[i + j * row];
            }
        }
        v.copy_from_slice(&tmp);
    }

    pub fn load(path: &Path) -> Option<Self> {
        let config = NNUEConfig::from_file(&path.join("config.json"))?;
        let mut offsets = Vec::new();
        let mut offset = 0;
        for pattern_bits in &config.patterns {
            offsets.push(offset);
            offset += pow3(pattern_bits.count_ones() as i8);
        }
        let mut embedding = Self::load_param(&path.join("embedding.weight"), offset * config.front)?;
        for chunk in embedding.chunks_mut(config.front) {
            Self::normalize_vec(chunk, 1.0);
        }
        let embedding = embedding;
        let mut layer1_weight = Self::load_param(
            &path.join("backend_block.0.weight"),
            config.front * config.middle,
        )?;
        Self::transpose_mat(&mut layer1_weight, config.front, config.middle);
        let mut layer2_weight = Self::load_param(
            &path.join("backend_block.2.weight"),
            config.middle * config.back,
        )?;
        Self::transpose_mat(&mut layer2_weight, config.middle, config.back);
        let layer3_weight = Self::load_param(&path.join("backend_block.4.weight"), config.back)?;
        let layer1_bias = Self::load_param(&path.join("backend_block.0.bias"), config.middle)?;
        let layer2_bias = Self::load_param(&path.join("backend_block.2.bias"), config.back)?;
        let layer3_bias = Self::load_param(&path.join("backend_block.4.bias"), 1)?;
        let pos_to_indices = config
            .patterns
            .iter()
            .map(|pattern| Self::generate_pos_to_indices(*pattern))
            .collect();
        Some(Self {
            config,
            embedding,
            offsets,
            pos_to_indices,
            layer1_weight,
            layer1_bias,
            layer2_weight,
            layer2_bias,
            layer3_weight,
            layer3_bias,
        })
    }

    fn lookup_vec(&self, index: usize) -> &[f32] {
        let first = index * self.config.front;
        let last = first + self.config.front;
        &self.embedding[first..last]
    }

    fn score_scale() -> i16 {
        256
    }

    fn score_min() -> i16 {
        -Self::score_scale() * BOARD_SIZE as i16
    }

    fn score_max() -> i16 {
        Self::score_scale() * BOARD_SIZE as i16
    }

    fn compute_index(&self, board: Board, pattern: u64, offset: usize) -> usize {
        let pbits = board.player.pext(pattern) as usize;
        let obits = board.opponent.pext(pattern) as usize;
        BASE_2_TO_3[pbits] + 2 * BASE_2_TO_3[obits] + offset
    }

    fn smooth_val(raw_score: i32) -> i16 {
        let scale32 = Self::score_scale() as i32;
        (if raw_score > 63 * scale32 {
            (BOARD_SIZE as i32) * scale32 - scale32 * scale32 / (raw_score - 62 * scale32)
        } else if raw_score < -63 * scale32 {
            -(BOARD_SIZE as i32) * scale32 - scale32 * scale32 / (raw_score + 62 * scale32)
        } else {
            raw_score
        }) as i16
    }

    fn embedding_bag_impl(&self, index: usize, front_vec: &mut [f32]) {
        for (f, e) in front_vec.iter_mut().zip(self.lookup_vec(index)) {
            *f += *e;
        }
    }
}

impl Evaluator for NNUEEvaluator {
    fn eval(&self, mut board: Board) -> i16 {
        let mut front_vec = vec![0.0; self.config.front];
        for _ in 0..4 {
            for (pattern, offset) in self.config.patterns.iter().zip(self.offsets.iter()) {
                let index = self.compute_index(board, *pattern, *offset);
                self.embedding_bag_impl(index, &mut front_vec);
            }
            let board_flip = board.flip_diag();
            for (pattern, offset) in self.config.patterns.iter().zip(self.offsets.iter()) {
                let index = self.compute_index(board_flip, *pattern, *offset);
                self.embedding_bag_impl(index, &mut front_vec);
            }
            board = board.rot90();
        }
        let mut middle_vec = self.layer1_bias.clone();
        let mut index = 0;
        for fe in front_vec.iter() {
            for me in middle_vec.iter_mut() {
                *me += *fe * unsafe { *self.layer1_weight.get_unchecked(index) };
                index += 1;
            }
        }
        for me in middle_vec.iter_mut() {
            if *me < 0. {
                *me = 0.;
            }
        }
        let mut back_vec = self.layer2_bias.clone();
        let mut index = 0;
        for me in middle_vec.iter() {
            for be in back_vec.iter_mut() {
                *be += *me * unsafe { *self.layer2_weight.get_unchecked(index) };
                index += 1;
            }
        }
        for be in back_vec.iter_mut() {
            if *be < 0. {
                *be = 0.;
            }
        }
        let mut result = self.layer3_bias[0];
        for (be, w) in back_vec.iter().zip(self.layer3_weight.iter()) {
            result += *be * *w;
        }
        Self::smooth_val((Self::score_scale() as f32 * result).round() as i32)
    }

    fn score_scale(&self) -> i16 {
        Self::score_scale()
    }

    fn score_min(&self) -> i16 {
        Self::score_min()
    }

    fn score_max(&self) -> i16 {
        Self::score_max()
    }
}
