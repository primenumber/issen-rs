#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::board::*;
#[cfg(target_feature = "avx2")]
use core::arch::x86_64::*;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::ops::RangeInclusive;
use std::path::Path;
use std::simd::prelude::*;
use yaml_rust::yaml;

struct EvaluatorConfig {
    masks: Vec<String>,
    stones_range: RangeInclusive<i8>,
}

impl EvaluatorConfig {
    fn new(config_path: &Path) -> Option<EvaluatorConfig> {
        let mut config_file = File::open(config_path).ok()?;
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

// interprete base-2 number as base-3 number
// base_2_to_3(x) := radix_parse(radix_fmt(x, 2), 3)
const fn base_2_to_3(mut x: usize) -> usize {
    let mut base3 = 0;
    let mut pow3 = 1;
    while x > 0 {
        base3 += (x % 2) * pow3;
        pow3 *= 3;
        x /= 2;
    }
    base3
}

const NON_PATTERN_SCORES: usize = 4;

const BASE_2_TO_3_TABLE_BITS: usize = 13;
const BASE_2_TO_3: [usize; 1 << BASE_2_TO_3_TABLE_BITS] = {
    let mut table = [0usize; 1 << BASE_2_TO_3_TABLE_BITS];
    let mut i = 0;
    while i < table.len() {
        table[i] = base_2_to_3(i);
        i += 1;
    }
    table
};

struct EvaluatorPattern {
    mask: u64,
    offset: usize,
    pattern_count: usize,
}

struct Parameters {
    pattern_weights: Vec<i16>,
    patterns: Vec<u64>,
    offsets: Vec<u32>,
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
            let mut t_bits = (i as u64).pdep(t_pattern);
            for dir in 0..4 {
                indices[dir * 2].push(t_bits.pext(t_pattern) as usize);
                indices[dir * 2 + 1].push(flip_diag(t_bits).pext(flip_diag(t_pattern)) as usize);
                t_bits = rot90(t_bits);
                t_pattern = rot90(t_pattern);
            }
        }
        indices
    }

    fn expand_weights_by_d4(weights: &[i16], pattern: u64) -> Vec<(u64, Vec<i16>)> {
        let mut permuted_weights = vec![vec![0; weights.len()]; 8];
        let perm = Self::permute_indices(pattern);
        let pattern_size = pattern.count_ones();
        for pidx in 0..(1 << pattern_size) {
            for oidx in 0..(1 << pattern_size) {
                if (pidx & oidx) != 0 {
                    continue;
                }
                let orig_index = BASE_2_TO_3[pidx] + BASE_2_TO_3[oidx] * 2;
                for i in 0..8 {
                    let t_pidx = perm[i][pidx];
                    let t_oidx = perm[i][oidx];
                    let index = BASE_2_TO_3[t_pidx] + BASE_2_TO_3[t_oidx] * 2;
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
        patterns_by_d4.into_iter().zip(permuted_weights).collect()
    }

    fn new(orig_weights: &[i16], orig_patterns: &[EvaluatorPattern]) -> Option<Parameters> {
        let mut v: Vec<(u64, Vec<i16>)> = Vec::new();
        for pattern in orig_patterns.iter() {
            let expanded = if pattern.mask != 0 {
                Self::expand_weights_by_d4(
                    &orig_weights[pattern.offset..(pattern.offset + pattern.pattern_count)],
                    pattern.mask,
                )
            } else {
                // non-pattern scores should not be expanded
                vec![(
                    pattern.mask,
                    orig_weights[pattern.offset..(pattern.offset + pattern.pattern_count)].to_vec(),
                )]
            };
            for (t_pattern, t_weights) in expanded {
                if let Some((_, vw)) = v.iter_mut().find(|(p, _)| *p == t_pattern) {
                    for (w, &e) in vw.iter_mut().zip(t_weights.iter()) {
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
            offsets.push(offset as u32);
            offset += ex_weights.len();
            pattern_weights.extend(ex_weights);
            patterns.push(pattern);
        }
        pattern_weights.push(0);
        while offsets.len() % 16 != 0 {
            offsets.push(offset as u32);
        }
        let non_patterns_offset = orig_patterns.last().unwrap().offset;
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
}

// not configuable
const INDICES_VECTORIZER_VECTOR_ELEMENTS: usize = 16;
const INDICES_VECTORIZER_VECTOR_COUNT: usize = 3;
const INDICES_VECTORIZER_LEN: usize = INDICES_VECTORIZER_VECTOR_COUNT * INDICES_VECTORIZER_VECTOR_ELEMENTS;

struct IndicesVectorizer<const W: usize> {
    chunk_count: usize,
    shifts: Vec<usize>,
    indices: Vec<u16>,
}

impl<const W: usize> IndicesVectorizer<W> {
    fn new(patterns: &[u64]) -> Self {
        let chunk_count = (64 + W - 1) / W;
        let mut shifts = Vec::new();
        let mut indices = Vec::new();
        for chunk_idx in 0..chunk_count {
            let shift = chunk_idx * W;
            shifts.push(shift);
            for bits in 0..(1 << W) {
                let chunk_bits = bits << shift;
                for &pattern in patterns {
                    let index = BASE_2_TO_3[chunk_bits.pext(pattern) as usize];
                    indices.push(index as u16);
                }
                while indices.len() % INDICES_VECTORIZER_VECTOR_ELEMENTS != 0 {
                    indices.push(0);
                }
            }
        }
        Self {
            chunk_count,
            shifts,
            indices,
        }
    }

    fn feature_indices(&self, board: Board) -> [u16x16; INDICES_VECTORIZER_VECTOR_COUNT] {
        let mut idx0 = Simd::splat(0);
        let mut idx1 = Simd::splat(0);
        let mut idx2 = Simd::splat(0);
        let mask: u64 = (1 << W) - 1;
        for chunk_idx in 0..self.chunk_count {
            let pidx = ((board.player >> (chunk_idx * W)) & mask) as usize;
            let oidx = ((board.opponent >> (chunk_idx * W)) & mask) as usize;
            let offset_base_p = INDICES_VECTORIZER_LEN * (pidx + (1 << W) * chunk_idx);
            let offset_base_o = INDICES_VECTORIZER_LEN * (oidx + (1 << W) * chunk_idx);
            let vp0 = Simd::from_slice(unsafe {
                self.indices
                    .get_unchecked(offset_base_p..(offset_base_p + 16))
            });
            let vp1 = Simd::from_slice(unsafe {
                self.indices
                    .get_unchecked((offset_base_p + 16)..(offset_base_p + 32))
            });
            let vp2 = Simd::from_slice(unsafe {
                self.indices
                    .get_unchecked((offset_base_p + 32)..(offset_base_p + 48))
            });
            let vo0 = Simd::from_slice(unsafe {
                self.indices
                    .get_unchecked(offset_base_o..(offset_base_o + 16))
            });
            let vo1 = Simd::from_slice(unsafe {
                self.indices
                    .get_unchecked((offset_base_o + 16)..(offset_base_o + 32))
            });
            let vo2 = Simd::from_slice(unsafe {
                self.indices
                    .get_unchecked((offset_base_o + 32)..(offset_base_o + 48))
            });
            idx0 = idx0 + vp0 + vo0 + vo0;
            idx1 = idx1 + vp1 + vo1 + vo1;
            idx2 = idx2 + vp2 + vo2 + vo2;
        }
        [idx0, idx1, idx2]
    }
}

// configuable
const INDICES_VECTORIZER_PACK_SIZE: usize = 8;

pub struct Evaluator {
    stones_range: RangeInclusive<i8>,
    params: Vec<Parameters>,
    vectorizer: IndicesVectorizer<INDICES_VECTORIZER_PACK_SIZE>,
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
        value_file.read_exact(&mut buf).ok()?;
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

    fn load_all_weights(table_path: &Path, config: &EvaluatorConfig, length: usize) -> Vec<Vec<i16>> {
        let mut weights = Vec::new();
        for num in config.stones_range.clone() {
            let path = table_path.join(format!("value{}", num));
            weights.push(Self::load_weight(&path, length).unwrap());
        }
        weights
    }

    fn load_patterns(config: &EvaluatorConfig) -> (Vec<EvaluatorPattern>, usize) {
        let mut patterns = Vec::new();
        let mut offset = 0;
        for pattern_str in config.masks.iter() {
            let mask = u64::from_str_radix(pattern_str, 2).unwrap();
            let mask_size = popcnt(mask);
            let pattern_count = pow3(mask_size);
            patterns.push(EvaluatorPattern {
                mask,
                offset,
                pattern_count,
            });
            offset += pattern_count;
        }

        patterns.push(EvaluatorPattern {
            mask: 0,
            offset,
            pattern_count: NON_PATTERN_SCORES,
        });
        offset += NON_PATTERN_SCORES;
        (patterns, offset)
    }

    pub fn new(table_dirname: &str) -> Evaluator {
        let table_path = Path::new(table_dirname);
        let config_path = table_path.join("config.yaml");
        let config = EvaluatorConfig::new(&config_path).unwrap();
        let (patterns, length) = Self::load_patterns(&config);
        let weights = Self::load_all_weights(table_path, &config, length);
        let smoothed_weights = Self::smooth_weight(&weights, length, 1);
        let params: Vec<_> = smoothed_weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let mut param = Parameters::new(w, &patterns).unwrap();
                param.fold_parity(i as i8 + config.stones_range.start());
                param
            })
            .collect();

        let vectorizer = IndicesVectorizer::<INDICES_VECTORIZER_PACK_SIZE>::new(&params.first().unwrap().patterns);

        Evaluator {
            stones_range: config.stones_range,
            params,
            vectorizer,
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

    #[cfg(target_feature = "avx2")]
    fn eval_gather(&self, param: &Parameters, vidx: [u16x16; 3]) -> i32 {
        unsafe {
            unsafe fn unpack_idx(idx: __m256i) -> (__m256i, __m256i) {
                const MM_PERM_ACBD: i32 = 0b11011000;
                let permed = _mm256_permute4x64_epi64(idx, MM_PERM_ACBD);
                let lo = _mm256_unpacklo_epi16(permed, _mm256_setzero_si256());
                let hi = _mm256_unpackhi_epi16(permed, _mm256_setzero_si256());
                (lo, hi)
            }
            let (idxh0, idxh1) = unpack_idx(vidx[0].into());
            let (idxh2, idxh3) = unpack_idx(vidx[1].into());
            let (idxh4, idxh5) = unpack_idx(vidx[2].into());
            unsafe fn gather_weight(param: &Parameters, idx: __m256i, start: usize) -> __m256i {
                let offset = _mm256_add_epi32(
                    idx,
                    _mm256_loadu_si256(&param.offsets[start] as *const u32 as *const __m256i),
                );
                _mm256_i32gather_epi32(param.pattern_weights.as_ptr() as *const i32, offset, 2)
            }
            let vw0 = gather_weight(param, idxh0, 0);
            let vw1 = gather_weight(param, idxh1, 8);
            let vw2 = gather_weight(param, idxh2, 16);
            let vw3 = gather_weight(param, idxh3, 24);
            let vw4 = gather_weight(param, idxh4, 32);
            let vw5 = gather_weight(param, idxh5, 40);
            unsafe fn vector_sum(
                v0: __m256i,
                v1: __m256i,
                v2: __m256i,
                v3: __m256i,
                v4: __m256i,
                v5: __m256i,
            ) -> u16x16 {
                let sum0 = _mm256_blend_epi16(v0, _mm256_slli_epi32(v1, 16), 0b10101010);
                let sum1 = _mm256_blend_epi16(v2, _mm256_slli_epi32(v3, 16), 0b10101010);
                let sum2 = _mm256_blend_epi16(v4, _mm256_slli_epi32(v5, 16), 0b10101010);
                _mm256_add_epi16(_mm256_add_epi16(sum0, sum1), sum2).into()
            }
            vector_sum(vw0, vw1, vw2, vw3, vw4, vw5).reduce_sum() as i16 as i32
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    fn eval_gather(&self, param: &Parameters, vidx: [u16x16; 3]) -> i32 {
        let mut offsets = [0u16; 48];
        vidx[0].copy_to_slice(unsafe { offsets.get_unchecked_mut(0..16) });
        vidx[1].copy_to_slice(unsafe { offsets.get_unchecked_mut(16..32) });
        vidx[2].copy_to_slice(unsafe { offsets.get_unchecked_mut(32..48) });
        let mut sum = 0i32;
        for (idx, poffset) in offsets.iter().zip(param.offsets.iter()) {
            let offset = *idx as u32 + *poffset;
            sum += *unsafe { param.pattern_weights.get_unchecked(offset as usize) } as i32;
        }
        sum
    }

    fn eval_impl(&self, board: Board) -> i32 {
        let rem: usize = popcnt(board.empty()) as usize;
        let stones = (BOARD_SIZE - rem)
            .max(*self.stones_range.start() as usize)
            .min(*self.stones_range.end() as usize);
        let param_index = stones - *self.stones_range.start() as usize;
        let param = unsafe { self.params.get_unchecked(param_index) };

        // non-pattern scores
        let mut score = param.constant_score;
        score += param.p_mobility_score * popcnt(board.mobility_bits()) as i16;
        score += param.o_mobility_score * popcnt(board.pass_unchecked().mobility_bits()) as i16;
        let mut score = score as i32;

        // pattern-based scores
        let vidx = self.vectorizer.feature_indices(board);
        score += self.eval_gather(param, vidx);
        score
    }

    pub fn eval(&self, board: Board) -> i16 {
        Self::smooth_val(self.eval_impl(board))
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
