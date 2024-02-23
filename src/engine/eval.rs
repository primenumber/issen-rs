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
    masks: Vec<u64>,
    stones_range: RangeInclusive<i8>,
}

impl EvaluatorConfig {
    fn from_file(config_path: &Path) -> Option<EvaluatorConfig> {
        let mut config_file = File::open(config_path).ok()?;
        let mut config_string = String::new();
        config_file.read_to_string(&mut config_string).ok()?;
        let config_objs = yaml::YamlLoader::load_from_str(&config_string).ok()?;
        let config = &config_objs[0];
        let masks = config["masks"]
            .as_vec()?
            .iter()
            .map(|e| u64::from_str_radix(e.as_str().unwrap(), 2).unwrap())
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

    fn pattern_info(&self) -> (Vec<EvaluatorPattern>, usize, usize) {
        let mut patterns = Vec::new();
        let mut offset = 0;
        for &mask in self.masks.iter() {
            let mask_size = popcnt(mask);
            let pattern_count = pow3(mask_size);
            patterns.push(EvaluatorPattern {
                mask,
                offset,
                pattern_count,
            });
            offset += pattern_count;
        }
        (patterns, offset, NON_PATTERN_SCORES)
    }
}

struct PatternWeightTable {
    mask: u64,
    weights: Vec<i16>,
}

struct EvaluatorPattern {
    mask: u64,
    offset: usize,
    pattern_count: usize,
}

struct WeightTable {
    pattern_tables: Vec<PatternWeightTable>,
    p_mobility_score: i16,
    o_mobility_score: i16,
    parity_score: i16,
    constant_score: i16,
}

struct FoldedEvaluator {
    config: EvaluatorConfig,
    weights: Vec<WeightTable>,
}

impl FoldedEvaluator {
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

    fn unpack_weights(weights: &[i16], patterns: &[EvaluatorPattern], non_patterns_offset: usize) -> WeightTable {
        let mut pattern_tables = Vec::new();
        for pattern in patterns {
            pattern_tables.push(PatternWeightTable {
                mask: pattern.mask,
                weights: weights[pattern.offset..(pattern.offset + pattern.pattern_count)].to_vec(),
            });
        }
        let p_mobility_score = weights[non_patterns_offset];
        let o_mobility_score = weights[non_patterns_offset + 1];
        let parity_score = weights[non_patterns_offset + 2];
        let constant_score = weights[non_patterns_offset + 3];
        WeightTable {
            pattern_tables,
            p_mobility_score,
            o_mobility_score,
            parity_score,
            constant_score,
        }
    }

    fn load(path: &Path) -> Option<FoldedEvaluator> {
        let config_path = path.join("config.yaml");
        let config = EvaluatorConfig::from_file(&config_path)?;
        let (patterns, pettern_weights_length, non_patterns_count) = config.pattern_info();
        let packed_weights = Self::load_all_weights(path, &config, pettern_weights_length + non_patterns_count);
        let smoothed_packed_weights = Self::smooth_weight(
            &packed_weights,
            pettern_weights_length + non_patterns_count,
            1,
        );
        let weights = smoothed_packed_weights
            .iter()
            .map(|packed| Self::unpack_weights(packed, &patterns, pettern_weights_length))
            .collect();
        Some(FoldedEvaluator { config, weights })
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

// x = R^rot M ^ mirror
struct SquareGroup {
    rot: u8,
    mirror: u8,
}

impl SquareGroup {
    #[allow(dead_code)]
    fn compose(&self, rhs: &Self) -> Self {
        let rot = if self.mirror != 0 {
            (4 + self.rot - rhs.rot) % 4
        } else {
            (self.rot + rhs.rot) % 4
        };
        Self {
            rot,
            mirror: (self.mirror + rhs.mirror) % 2,
        }
    }

    fn apply(&self, mut bits: u64) -> u64 {
        if self.mirror != 0 {
            bits = flip_diag(bits);
        }
        for _ in 0..self.rot {
            bits = rot90(bits);
        }
        bits
    }

    fn all_elements() -> Vec<Self> {
        let mut res = Vec::new();
        for rot in 0..4 {
            for mirror in 0..2 {
                res.push(Self { rot, mirror });
            }
        }
        res
    }
}

struct PatternPermutation {
    perms: Vec<(u64, Vec<usize>)>,
}

impl PatternPermutation {
    fn permute_indices_base2(mask: u64) -> PatternPermutation {
        let pattern_size = mask.count_ones();
        let table_size = (1 << pattern_size) as usize;
        PatternPermutation {
            perms: SquareGroup::all_elements()
                .iter()
                .map(|op| {
                    let pattern = op.apply(mask);
                    let permutation = (0..table_size)
                        .map(|pidx| {
                            let bits = (pidx as u64).pdep(mask);
                            op.apply(bits).pext(pattern) as usize
                        })
                        .collect();
                    (pattern, permutation)
                })
                .collect(),
        }
    }

    fn permute_indices(mask: u64) -> PatternPermutation {
        let pattern_size = mask.count_ones();
        PatternPermutation {
            perms: Self::permute_indices_base2(mask)
                .perms
                .iter()
                .map(|(pattern, perm_base2)| {
                    let mut perm_base3 = vec![0; pow3(pattern_size as i8)];
                    for pidx in 0..(1 << pattern_size) {
                        for oidx in 0..(1 << pattern_size) {
                            if (pidx & oidx) != 0 {
                                continue;
                            }
                            let orig_index = BASE_2_TO_3[pidx] + BASE_2_TO_3[oidx] * 2;
                            let t_pidx = perm_base2[pidx];
                            let t_oidx = perm_base2[oidx];
                            let index = BASE_2_TO_3[t_pidx] + BASE_2_TO_3[t_oidx] * 2;
                            perm_base3[index] = orig_index;
                        }
                    }
                    (*pattern, perm_base3)
                })
                .collect(),
        }
    }
}

impl PatternPermutation {
    fn expand_weights_by_d4(&self, weights: &[i16]) -> Vec<PatternWeightTable> {
        self.perms
            .iter()
            .map(|(pattern, perm_base3)| {
                let mut permuted_weights = vec![0; perm_base3.len()];
                for (i, &p) in perm_base3.iter().enumerate() {
                    permuted_weights[i] = weights[p];
                }
                PatternWeightTable {
                    mask: *pattern,
                    weights: permuted_weights,
                }
            })
            .collect()
    }

    fn expand_weights_with_compaction(&self, weights: &[i16]) -> Vec<PatternWeightTable> {
        let mut result: Vec<PatternWeightTable> = Vec::new();
        for table in self.expand_weights_by_d4(weights) {
            if let Some(ref_table) = result
                .iter_mut()
                .find(|ref_table| ref_table.mask == table.mask)
            {
                for (w, &e) in ref_table.weights.iter_mut().zip(table.weights.iter()) {
                    *w += e;
                }
            } else {
                result.push(table);
            }
        }
        result
    }
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
    fn new(table: &WeightTable) -> Option<Parameters> {
        let mut v: Vec<PatternWeightTable> = Vec::new();
        for table in table.pattern_tables.iter() {
            let perm = PatternPermutation::permute_indices(table.mask);
            v.extend(perm.expand_weights_with_compaction(&table.weights));
        }
        let mut pattern_weights = Vec::new();
        let mut patterns = Vec::new();
        let mut offsets = Vec::new();
        let mut offset = 0;
        for table in v {
            offsets.push(offset as u32);
            offset += table.weights.len();
            pattern_weights.extend(table.weights);
            patterns.push(table.mask);
        }
        // padding for vectorization
        pattern_weights.push(0);
        while offsets.len() % 16 != 0 {
            offsets.push(offset as u32);
        }
        Some(Parameters {
            pattern_weights,
            patterns,
            offsets,
            p_mobility_score: table.p_mobility_score,
            o_mobility_score: table.o_mobility_score,
            parity_score: table.parity_score,
            constant_score: table.constant_score,
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
    indices: Vec<u16>,
}

impl<const W: usize> IndicesVectorizer<W> {
    fn new(patterns: &[u64]) -> Self {
        let chunk_count = (64 + W - 1) / W;
        let mut indices = Vec::new();
        for chunk_idx in 0..chunk_count {
            let shift = chunk_idx * W;
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
        Self { indices }
    }

    fn feature_indices(&self, board: Board) -> [u16x16; INDICES_VECTORIZER_VECTOR_COUNT] {
        let mut indices = [Simd::splat(0); INDICES_VECTORIZER_VECTOR_COUNT];
        let mask: u64 = (1 << W) - 1;
        let chunk_count = (64 + W - 1) / W;
        for chunk_idx in 0..chunk_count {
            let pidx = ((board.player >> (chunk_idx * W)) & mask) as usize;
            let oidx = ((board.opponent >> (chunk_idx * W)) & mask) as usize;
            let offset_base_p = INDICES_VECTORIZER_LEN * (pidx + (1 << W) * chunk_idx);
            let offset_base_o = INDICES_VECTORIZER_LEN * (oidx + (1 << W) * chunk_idx);
            for (vidx, idx) in indices.iter_mut().enumerate() {
                let vp = Simd::from_slice(unsafe {
                    self.indices
                        .get_unchecked((offset_base_p + 16 * vidx)..(offset_base_p + 16 * (vidx + 1)))
                });
                let vo = Simd::from_slice(unsafe {
                    self.indices
                        .get_unchecked((offset_base_o + 16 * vidx)..(offset_base_o + 16 * (vidx + 1)))
                });
                *idx += vp + vo + vo;
            }
        }
        indices
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
    pub fn load(path: &Path) -> Option<Evaluator> {
        let folded = FoldedEvaluator::load(path)?;
        Some(Self::from_folded(&folded))
    }

    fn from_folded(folded: &FoldedEvaluator) -> Evaluator {
        let params: Vec<_> = folded
            .weights
            .iter()
            .enumerate()
            .map(|(i, table)| {
                let mut param = Parameters::new(table).unwrap();
                param.fold_parity(i as i8 + folded.config.stones_range.start());
                param
            })
            .collect();

        let vectorizer = IndicesVectorizer::<INDICES_VECTORIZER_PACK_SIZE>::new(&params.first().unwrap().patterns);

        Evaluator {
            stones_range: folded.config.stones_range.clone(),
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
            let (idxh0, idxh1) = unpack_idx((*vidx.get_unchecked(0)).into());
            let (idxh2, idxh3) = unpack_idx((*vidx.get_unchecked(1)).into());
            let (idxh4, idxh5) = unpack_idx((*vidx.get_unchecked(2)).into());
            unsafe fn gather_weight(param: &Parameters, idx: __m256i, start: usize) -> __m256i {
                let offset = _mm256_add_epi32(
                    idx,
                    _mm256_loadu_si256(param.offsets.get_unchecked(start) as *const u32 as *const __m256i),
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
        // Since parity_score s absorbed into constant_score, parity_score can be assumed to be 0
        // here.
        // score += param.parity_score;
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
        for raw in -60000..=60000 {
            let smoothed = Evaluator::smooth_val(raw);
            assert!(smoothed > EVAL_SCORE_MIN);
            assert!(smoothed < EVAL_SCORE_MAX);
        }
    }
}
