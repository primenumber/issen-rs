#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem;
use std::path::Path;
use std::simd::prelude::*;
use std::simd::StdFloat;

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

fn get_pattern_to_indices(pattern: u64) -> [u16; BOARD_SIZE] {
    let mut count = 0;
    let mut result = [0; BOARD_SIZE];
    for (i, e) in result.iter_mut().enumerate() {
        if (pattern >> i) & 1 == 1 {
            *e = pow3(count) as u16;
            count += 1;
        }
    }
    result
}

fn get_pos_to_indices_single(pos: usize, indices: &[u16; 64]) -> Vec<u16> {
    let mut bits: u64 = 1 << pos;
    let mut result = Vec::new();
    for _ in 0..4 {
        let new_pos = bits.trailing_zeros() as usize;
        result.push(indices[new_pos]);
        let new_pos_flip = flip_vertical(bits).trailing_zeros() as usize;
        result.push(indices[new_pos_flip]);
        bits = rot90(bits);
    }
    result
}

fn get_pos_to_indices(patterns: &[u64]) -> Vec<Vec<u16>> {
    let mut result = vec![Vec::new(); BOARD_SIZE];
    for pattern in patterns.into_iter() {
        let indices = get_pattern_to_indices(*pattern);
        for pos in 0..BOARD_SIZE {
            result[pos].extend(get_pos_to_indices_single(pos, &indices));
        }
    }
    result
}

fn get_line_to_indices(patterns: &[u64]) -> Vec<u16> {
    let p2i = get_pos_to_indices(patterns);
    let vec_len = p2i[0].len();
    let mut result = Vec::new();
    for row in 0..8 {
        for col_bits in 0..256 {
            let mut indices = vec![0; vec_len];
            for col in 0..8 {
                if (col_bits >> col) & 1 == 1 {
                    for (d, s) in indices.iter_mut().zip(p2i[row * 8 + col].iter()) {
                        *d += *s;
                    }
                }
            }
            result.extend(indices);
        }
    }
    result
}

#[derive(Clone, Copy, Debug)]
struct Bf16(u16);

impl From<f32> for Bf16 {
    fn from(value: f32) -> Self {
        Self(u16::from_le_bytes(
            *value.to_le_bytes().last_chunk::<2>().unwrap(),
        ))
    }
}

impl From<Bf16> for f32 {
    fn from(value: Bf16) -> Self {
        let lower_ext = (value.0 as u32) << 16;
        Self::from_le_bytes(lower_ext.to_le_bytes())
    }
}

#[derive(Clone, Debug)]
#[repr(C, align(64))]
struct AlignedBF16Array([Bf16; 32]);

pub struct NNUEEvaluator {
    config: NNUEConfig,
    embedding: Vec<AlignedBF16Array>,
    offsets_extended: Vec<u32>,
    line_to_idx_vec: Vec<u16>,
    layer1_weight: Vec<f32>,
    layer1_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: Vec<f32>,
    layer3_weight: Vec<f32>,
    layer3_bias: Vec<f32>,
}

const EXPECTED_FRONT: usize = 32;
const EXPECTED_MIDDLE: usize = 64;
const EXPECTED_BACK: usize = 32;

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

    fn transpose_mat(v: &mut [f32], row: usize, col: usize) {
        let mut tmp = vec![0.; row * col];
        for i in 0..row {
            for j in 0..col {
                tmp[i * col + j] = v[i + j * row];
            }
        }
        v.copy_from_slice(&tmp);
    }

    #[cfg(not(target_feature = "avx512bf16"))]
    fn trunc_to_bf16(v: &[f32]) -> Vec<AlignedBF16Array> {
        let mut result = vec![AlignedBF16Array([Bf16(0); 32]); v.len() / 32];
        unsafe {
            use std::arch::x86_64::*;
            let (head, _tail) = v.as_chunks::<16>();
            for (i, chunk) in head.into_iter().enumerate() {
                let v0 = _mm256_castps_si256(_mm256_loadu_ps(&chunk[0] as *const f32));
                let v1 = _mm256_castps_si256(_mm256_loadu_ps(&chunk[8] as *const f32));
                let packed = _mm256_packus_epi32(_mm256_srli_epi32(v0, 16), _mm256_srli_epi32(v1, 16));
                _mm256_storeu_si256(
                    &mut result[i * 16] as *mut AlignedBF16Array as *mut __m256i,
                    packed,
                );
            }
        }
        result
    }

    #[cfg(target_feature = "avx512bf16")]
    fn trunc_to_bf16(v: &[f32]) -> Vec<AlignedBF16Array> {
        let mut result = vec![AlignedBF16Array([Bf16(0); 32]); v.len() / 32];
        unsafe {
            use std::arch::x86_64::*;
            let (head, _tail) = v.as_chunks::<32>();
            for (i, chunk) in head.into_iter().enumerate() {
                let v0 = _mm512_castps_si512(_mm512_loadu_ps(&chunk[0] as *const f32));
                let v1 = _mm512_castps_si512(_mm512_loadu_ps(&chunk[16] as *const f32));
                let packed = _mm512_packus_epi32(_mm512_srli_epi32(v0, 16), _mm512_srli_epi32(v1, 16));
                _mm512_storeu_si512(&mut result[i] as *mut AlignedBF16Array as *mut i32, packed);
            }
        }
        result
    }

    pub fn load(path: &Path) -> Option<Self> {
        let config = NNUEConfig::from_file(&path.join("config.json"))?;
        assert_eq!(config.front, EXPECTED_FRONT);
        assert_eq!(config.middle, EXPECTED_MIDDLE);
        assert_eq!(config.back, EXPECTED_BACK);
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
        let embedding = Self::trunc_to_bf16(&embedding);
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
        let line_to_idx_vec = get_line_to_indices(&config.patterns);
        let mut offsets_extended = Vec::new();
        for ofs in offsets.iter() {
            for _ in 0..8 {
                offsets_extended.push(*ofs as u32);
            }
        }
        Some(Self {
            config,
            embedding,
            offsets_extended,
            line_to_idx_vec,
            layer1_weight,
            layer1_bias,
            layer2_weight,
            layer2_bias,
            layer3_weight,
            layer3_bias,
        })
    }

    unsafe fn lookup_vec(&self, index: usize) -> &[AlignedBF16Array] {
        let first = (index * EXPECTED_FRONT) / 32;
        let last = first + EXPECTED_FRONT / 32;
        &self.embedding.get_unchecked(first..last)
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

    #[inline(never)]
    fn feature_indices(&self, board: Board) -> [u16; 128] {
        let num_features = self.offsets_extended.len();
        let mut iv0 = u16x64::splat(0);
        let mut iv1 = u16x64::splat(0);
        unsafe {
            for row in 0..8 {
                let pbits = ((board.player >> (row * 8)) & 0xff) as usize;
                let pfrom = (row * 256 + pbits) * num_features;
                iv0 += u16x64::from_slice(
                    &self
                        .line_to_idx_vec
                        .get_unchecked((pfrom + 0)..(pfrom + 64)),
                );
                iv1 += u16x64::from_slice(
                    &self
                        .line_to_idx_vec
                        .get_unchecked((pfrom + 64)..(pfrom + 128)),
                );
                let obits = ((board.opponent >> (row * 8)) & 0xff) as usize;
                let ofrom = (row * 256 + obits) * num_features;
                iv0 += u16x64::splat(2)
                    * u16x64::from_slice(
                        &self
                            .line_to_idx_vec
                            .get_unchecked((ofrom + 0)..(ofrom + 64)),
                    );
                iv1 += u16x64::splat(2)
                    * u16x64::from_slice(
                        &self
                            .line_to_idx_vec
                            .get_unchecked((ofrom + 64)..(ofrom + 128)),
                    );
            }
        }
        let mut idx_vec = [0u16; 128];
        iv0.copy_to_slice(&mut idx_vec[0..64]);
        iv1.copy_to_slice(&mut idx_vec[64..128]);
        idx_vec
    }

    #[allow(dead_code)]
    #[inline(never)]
    fn feature_indices_naive(&self, board: Board) -> [u16; 128] {
        let num_features = self.offsets_extended.len();
        let mut idx_vec = [0u16; 128];
        for row in 0..8 {
            let pbits = ((board.player >> (row * 8)) & 0xff) as usize;
            let pfrom = (row * 256 + pbits) * num_features;
            for (d, s) in idx_vec
                .iter_mut()
                .zip(self.line_to_idx_vec[pfrom..(pfrom + 128)].iter())
            {
                *d += s;
            }
            let obits = ((board.opponent >> (row * 8)) & 0xff) as usize;
            let ofrom = (row * 256 + obits) * num_features;
            for (d, s) in idx_vec
                .iter_mut()
                .zip(self.line_to_idx_vec[ofrom..(ofrom + 128)].iter())
            {
                *d += 2 * s;
            }
        }
        idx_vec
    }

    #[cfg(target_feature = "avx512bf16")]
    #[inline(never)]
    fn embedding_bag(&self, board: Board, front_vec: &mut [f32]) {
        let idx_vec = self.feature_indices(board);
        unsafe {
            use std::arch::x86_64::*;
            let mut v0 = _mm512_setzero_ps();
            let mut v1 = _mm512_setzero_ps();
            for (index, offset) in idx_vec.iter().zip(self.offsets_extended.iter()) {
                let emb_vec = self.lookup_vec(*index as usize + *offset as usize);
                let ve01_int = _mm512_load_si512(&emb_vec[0] as *const AlignedBF16Array as *const i32);
                let ve0 = _mm512_unpacklo_epi16(_mm512_setzero_si512(), ve01_int);
                let ve1 = _mm512_unpackhi_epi16(_mm512_setzero_si512(), ve01_int);
                v0 = _mm512_add_ps(v0, _mm512_castsi512_ps(ve0));
                v1 = _mm512_add_ps(v1, _mm512_castsi512_ps(ve1));
            }
            _mm512_storeu_ps(&mut front_vec[0] as *mut f32, v0);
            _mm512_storeu_ps(&mut front_vec[16] as *mut f32, v1);
        }
    }

    #[cfg(all(not(target_feature = "avx512bf16"), target_feature = "avx2"))]
    #[inline(never)]
    fn embedding_bag(&self, board: Board, front_vec: &mut [f32]) {
        let idx_vec = self.feature_indices(board);
        unsafe {
            use std::arch::x86_64::*;
            let mut v0 = _mm256_setzero_ps();
            let mut v1 = _mm256_setzero_ps();
            let mut v2 = _mm256_setzero_ps();
            let mut v3 = _mm256_setzero_ps();
            for (index, offset) in idx_vec.iter().zip(self.offsets_extended.iter()) {
                let emb_vec = self.lookup_vec(*index as usize + *offset as usize);
                let ve01 = _mm256_loadu_si256(&emb_vec[0] as *const Bf16 as *const __m256i);
                let ve0 = _mm256_unpacklo_epi16(_mm256_setzero_si256(), ve01);
                let ve1 = _mm256_unpackhi_epi16(_mm256_setzero_si256(), ve01);
                let ve23 = _mm256_loadu_si256(&emb_vec[16] as *const Bf16 as *const __m256i);
                let ve2 = _mm256_unpacklo_epi16(_mm256_setzero_si256(), ve23);
                let ve3 = _mm256_unpackhi_epi16(_mm256_setzero_si256(), ve23);
                v0 = _mm256_add_ps(v0, _mm256_castsi256_ps(ve0));
                v1 = _mm256_add_ps(v1, _mm256_castsi256_ps(ve1));
                v2 = _mm256_add_ps(v2, _mm256_castsi256_ps(ve2));
                v3 = _mm256_add_ps(v3, _mm256_castsi256_ps(ve3));
            }
            _mm256_storeu_ps(&mut front_vec[0] as *mut f32, v0);
            _mm256_storeu_ps(&mut front_vec[8] as *mut f32, v1);
            _mm256_storeu_ps(&mut front_vec[16] as *mut f32, v2);
            _mm256_storeu_ps(&mut front_vec[24] as *mut f32, v3);
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    fn unpack_u16(v: [u16; 16]) -> [f32; 16] {
        let mut result = [0.; 16];
        for (i, e) in v.iter().enumerate() {
            let hi_lo = (i / 4) % 2;
            let xmm_lane = (i / 8) % 2;
            result[hi_lo * 8 + xmm_lane * 4 + i % 4] = f32::from(*e);
        }
        result
    }

    #[cfg(not(target_feature = "avx2"))]
    #[inline(never)]
    fn embedding_bag(&self, board: Board, front_vec: &mut [f32]) {
        let idx_vec = self.feature_indices(board);
        let mut v = vec![0.; self.config.front];
        for (index, offset) in idx_vec.iter().zip(self.offsets_extended.iter()) {
            let emb_vec = unsafe { self.lookup_vec(*index as usize + *offset as usize) };
            let (schunks, _tail) = emb_vec.as_chunks::<16>();
            for (dchunk, schunk) in v.iter_mut().array_chunks::<16>().zip(schunks.into_iter()) {
                let s_unpacked = Self::unpack_u16(*schunk);
                for (d, s) in dchunk.into_iter().zip(s_unpacked) {
                    *d += s;
                }
            }
        }
        front_vec.copy_from_slice(&v);
    }

    #[inline(never)]
    fn layer_1(&self, front_vec: &[f32], middle_vec: &mut [f32]) {
        let mut b0 = f32x64::from_slice(&self.layer1_bias[0..64]);
        for (xelem, weight_chunk) in front_vec
            .iter()
            .zip(self.layer1_weight.array_chunks::<64>())
        {
            let xev = f32x64::splat(*xelem);
            b0 = xev.mul_add(f32x64::from_array(*weight_chunk), b0);
        }
        b0 = b0.simd_max(f32x64::splat(0.));
        b0.copy_to_slice(&mut middle_vec[0..64]);
    }

    #[inline(never)]
    fn layer_2(&self, middle_vec: &[f32], back_vec: &mut [f32]) {
        let mut b0 = f32x32::from_slice(&self.layer2_bias[0..32]);
        for (xelem, weight_chunk) in middle_vec
            .iter()
            .zip(self.layer2_weight.array_chunks::<32>())
        {
            let xev = f32x32::splat(*xelem);
            b0 = xev.mul_add(f32x32::from_array(*weight_chunk), b0);
        }
        b0 = b0.simd_max(f32x32::splat(0.));
        b0.copy_to_slice(&mut back_vec[0..32]);
    }

    #[inline(never)]
    fn layer_3(&self, back_vec: &[f32]) -> f32 {
        let mut result = self.layer3_bias[0];
        for (be, w) in back_vec.iter().zip(self.layer3_weight.iter()) {
            result += *be * *w;
        }
        result
    }
}

impl Evaluator for NNUEEvaluator {
    fn eval(&self, board: Board) -> i16 {
        let mut front_vec = vec![0.0; self.config.front];
        self.embedding_bag(board, &mut front_vec);
        let mut middle_vec = vec![0.0; self.config.middle];
        self.layer_1(&front_vec, &mut middle_vec);
        let mut back_vec = vec![0.0; self.config.back];
        self.layer_2(&middle_vec, &mut back_vec);
        let result = self.layer_3(&back_vec);
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
