#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::hand::*;
use clap::ArgMatches;
use core::arch::x86_64::*;
use lazy_static::lazy_static;
use std::fmt;
use std::io::{BufWriter, Write};
use std::str::FromStr;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Board {
    pub player: u64,
    pub opponent: u64,
    pub is_black: bool,
}

#[derive(Debug)]
pub struct UnmovableError;

#[derive(Debug)]
pub struct BoardParseError;

pub struct PlayIterator {
    board: Board,
    remain: u64,
}

pub const BOARD_SIZE: usize = 64;

unsafe fn upper_bit(mut x: __m256i) -> __m256i {
    x = _mm256_or_si256(x, _mm256_srli_epi64(x, 1));
    x = _mm256_or_si256(x, _mm256_srli_epi64(x, 2));
    x = _mm256_or_si256(x, _mm256_srli_epi64(x, 4));
    x = _mm256_or_si256(x, _mm256_srli_epi64(x, 8));
    x = _mm256_or_si256(x, _mm256_srli_epi64(x, 16));
    x = _mm256_or_si256(x, _mm256_srli_epi64(x, 32));
    let lowers = _mm256_srli_epi64(x, 1);
    _mm256_andnot_si256(lowers, x)
}

unsafe fn iszero(x: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    _mm256_cmpeq_epi64(x, zero)
}

unsafe fn reduce_or(x: __m256i) -> u64 {
    let xh = _mm_or_si128(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    (_mm_cvtsi128_si64(xh) | _mm_extract_epi64(xh, 1)) as u64
}

impl Board {
    pub fn initial_state() -> Board {
        Board {
            player: 0x00_00_00_08_10_00_00_00,
            opponent: 0x00_00_00_10_08_00_00_00,
            is_black: true,
        }
    }

    pub fn reverse_vertical(&self) -> Board {
        Board {
            player: flip_vertical(self.player),
            opponent: flip_vertical(self.opponent),
            is_black: self.is_black,
        }
    }

    unsafe fn flip_simd(&self, pos: usize) -> u64 {
        let p = _mm256_set1_epi64x(self.player as i64);
        let o = _mm256_set1_epi64x(self.opponent as i64);
        let omask = _mm256_setr_epi64x(
            0xFFFFFFFFFFFFFFFFu64 as i64,
            0x7E7E7E7E7E7E7E7Eu64 as i64,
            0x7E7E7E7E7E7E7E7Eu64 as i64,
            0x7E7E7E7E7E7E7E7Eu64 as i64,
        );
        let om = _mm256_and_si256(o, omask);
        let mask1 = _mm256_setr_epi64x(
            0x0080808080808080u64 as i64,
            0x7f00000000000000u64 as i64,
            0x0102040810204000u64 as i64,
            0x0040201008040201u64 as i64,
        );
        let mut mask = _mm256_srlv_epi64(mask1, _mm256_set1_epi64x((63 - pos) as i64));
        let mut outflank = _mm256_and_si256(upper_bit(_mm256_andnot_si256(om, mask)), p);
        let mut flipped = _mm256_and_si256(
            _mm256_slli_epi64(_mm256_sub_epi64(_mm256_setzero_si256(), outflank), 1),
            mask,
        );
        let mask2 = _mm256_setr_epi64x(
            0x0101010101010100u64 as i64,
            0x00000000000000feu64 as i64,
            0x0002040810204080u64 as i64,
            0x8040201008040200u64 as i64,
        );
        mask = _mm256_sllv_epi64(mask2, _mm256_set1_epi64x(pos as i64));
        outflank = _mm256_andnot_si256(
            _mm256_sub_epi64(_mm256_andnot_si256(om, mask), _mm256_set1_epi64x(1)),
            _mm256_and_si256(mask, p),
        );
        flipped = _mm256_or_si256(
            flipped,
            _mm256_andnot_si256(_mm256_sub_epi64(iszero(outflank), outflank), mask),
        );
        reduce_or(flipped)
    }

    pub fn flip_unchecked(&self, pos: usize) -> u64 {
        unsafe { self.flip_simd(pos) }
    }

    pub fn flip(&self, pos: usize) -> u64 {
        if ((self.empty() >> pos) & 1) == 0 {
            0
        } else {
            self.flip_unchecked(pos)
        }
    }

    pub fn is_movable(&self, pos: usize) -> bool {
        if pos >= BOARD_SIZE {
            return false;
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return false;
        }
        self.flip(pos) != 0
    }

    pub fn play(&self, pos: usize) -> Result<Board, UnmovableError> {
        if pos >= BOARD_SIZE {
            return Err(UnmovableError {});
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return Err(UnmovableError {});
        }
        let flip_bits = self.flip(pos);
        if flip_bits == 0 {
            return Err(UnmovableError {});
        }
        Ok(Board {
            player: self.opponent ^ flip_bits,
            opponent: (self.player ^ flip_bits) | (1u64 << pos),
            is_black: !self.is_black,
        })
    }

    pub fn pass(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
            is_black: !self.is_black,
        }
    }

    pub fn empty(&self) -> u64 {
        !(self.player | self.opponent)
    }

    unsafe fn mobility_bits_simd(&self) -> u64 {
        let shift1 = _mm256_setr_epi64x(1, 7, 9, 8);
        let mask = _mm256_setr_epi64x(
            0x7e7e7e7e7e7e7e7eu64 as i64,
            0x7e7e7e7e7e7e7e7eu64 as i64,
            0x7e7e7e7e7e7e7e7eu64 as i64,
            0xffffffffffffffffu64 as i64,
        );
        let v_player = _mm256_set1_epi64x(self.player as i64);
        let masked_op = _mm256_and_si256(_mm256_set1_epi64x(self.opponent as i64), mask);
        let mut flip_l = _mm256_and_si256(masked_op, _mm256_sllv_epi64(v_player, shift1));
        let mut flip_r = _mm256_and_si256(masked_op, _mm256_srlv_epi64(v_player, shift1));
        flip_l = _mm256_or_si256(
            flip_l,
            _mm256_and_si256(masked_op, _mm256_sllv_epi64(flip_l, shift1)),
        );
        flip_r = _mm256_or_si256(
            flip_r,
            _mm256_and_si256(masked_op, _mm256_srlv_epi64(flip_r, shift1)),
        );
        let pre_l = _mm256_and_si256(masked_op, _mm256_sllv_epi64(masked_op, shift1));
        let pre_r = _mm256_srlv_epi64(pre_l, shift1);
        let shift2 = _mm256_add_epi64(shift1, shift1);
        flip_l = _mm256_or_si256(
            flip_l,
            _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)),
        );
        flip_r = _mm256_or_si256(
            flip_r,
            _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)),
        );
        flip_l = _mm256_or_si256(
            flip_l,
            _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)),
        );
        flip_r = _mm256_or_si256(
            flip_r,
            _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)),
        );
        let mut res = _mm256_sllv_epi64(flip_l, shift1);
        res = _mm256_or_si256(res, _mm256_srlv_epi64(flip_r, shift1));
        res = _mm256_and_si256(res, _mm256_set1_epi64x(self.empty() as i64));
        reduce_or(res)
    }

    pub fn mobility_bits(&self) -> u64 {
        unsafe { self.mobility_bits_simd() }
    }

    pub fn mobility(&self) -> Vec<usize> {
        let mut result = Vec::new();
        let mut empties = self.empty();
        while empties != 0 {
            let bit = empties & empties.wrapping_neg();
            empties = empties & (empties - 1);
            let pos = popcnt(bit - 1) as usize;
            if self.is_movable(pos) {
                result.push(pos);
            }
        }
        result
    }

    pub fn next_iter(&self) -> PlayIterator {
        let e = self.empty();
        let remain = if popcnt(e) > 16 {
            self.mobility_bits()
        } else {
            e
        };
        PlayIterator {
            board: *self,
            remain,
        }
    }

    pub fn is_gameover(&self) -> bool {
        if !self.mobility().is_empty() {
            return false;
        }
        self.pass().mobility().is_empty()
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        let mut writer = BufWriter::new(std::io::stdout());
        for i in 0..BOARD_SIZE {
            if ((self.player >> i) & 1) != 0 {
                if self.is_black {
                    write!(writer, "X").unwrap();
                } else {
                    write!(writer, "O").unwrap();
                }
            } else if ((self.opponent >> i) & 1) != 0 {
                if self.is_black {
                    write!(writer, "O").unwrap();
                } else {
                    write!(writer, "X").unwrap();
                }
            } else {
                write!(writer, ".").unwrap();
            }
            if i % 8 == 7 {
                writeln!(writer).unwrap();
            }
        }
    }

    pub fn print_with_sides(&self) {
        let mut writer = BufWriter::new(std::io::stdout());
        write!(writer, " |abcdefgh\n-+--------\n").unwrap();
        for r in 0..8 {
            write!(writer, "{}|", r + 1).unwrap();
            for c in 0..8 {
                let i = r * 8 + c;
                if ((self.player >> i) & 1) != 0 {
                    if self.is_black {
                        write!(writer, "X").unwrap();
                    } else {
                        write!(writer, "O").unwrap();
                    }
                } else if ((self.opponent >> i) & 1) != 0 {
                    if self.is_black {
                        write!(writer, "O").unwrap();
                    } else {
                        write!(writer, "X").unwrap();
                    }
                } else {
                    write!(writer, ".").unwrap();
                }
            }
            writeln!(writer).unwrap();
        }
    }

    pub fn score(&self) -> i8 {
        let pcnt = popcnt(self.player);
        let ocnt = popcnt(self.opponent);
        if pcnt == ocnt {
            0
        } else if pcnt > ocnt {
            BOARD_SIZE as i8 - 2 * ocnt
        } else {
            -(BOARD_SIZE as i8) + 2 * pcnt
        }
    }

    pub fn flip_diag(&self) -> Board {
        Board {
            player: flip_diag(self.player),
            opponent: flip_diag(self.opponent),
            is_black: self.is_black,
        }
    }

    pub fn rot90(&self) -> Board {
        Board {
            player: rot90(self.player),
            opponent: rot90(self.opponent),
            is_black: self.is_black,
        }
    }

    pub fn stable_partial(&self) -> (u64, u64) {
        const MASK_TOP: u64 = 0x0000_0000_0000_00FF;
        const MASK_BOTTOM: u64 = 0xFF00_0000_0000_0000;
        const MASK_LEFT: u64 = 0x0101_0101_0101_0101;
        const MASK_RIGHT: u64 = 0x8080_8080_8080_8080;
        const MASKS: [u64; 4] = [MASK_TOP, MASK_BOTTOM, MASK_LEFT, MASK_RIGHT];
        let mut res = 0;
        for mask in &MASKS {
            let me = pext(self.player, *mask) as usize;
            let op = pext(self.opponent, *mask) as usize;
            let base3 = BASE3[me] + 2 * BASE3[op];
            res |= pdep(STABLE[base3], *mask);
        }
        let filled = !self.empty();
        let mut filled_v = filled;
        let mut filled_h = filled;
        let mut filled_a1h8 = filled;
        let mut filled_a8h1 = filled;
        // reduce
        filled_v &= filled_v >> 32;
        filled_h &= (filled_h >> 4) & 0x0F0F_0F0F_0F0F_0F0F;
        filled_a1h8 &= (filled_a1h8 >> 36) | 0x0F0F_0F0F_F0F0_F0F0;
        filled_a8h1 &= (filled_a8h1 >> 28) | 0xF0F0_F0F0_0F0F_0F0F;
        filled_v &= filled_v >> 16;
        filled_h &= (filled_h >> 2) & 0x3333_3333_3333_3333;
        filled_a1h8 &= (filled_a1h8 >> 18) | 0x0303_0000_0000_C0C0;
        filled_a8h1 &= (filled_a8h1 >> 14) | 0xC0C0_0000_0000_0303;
        filled_v &= filled_v >> 8;
        filled_h &= (filled_h >> 1) & 0x5555_5555_5555_5555;
        filled_a1h8 &= (filled_a1h8 >> 9) | 0x0100_0000_0000_0080;
        filled_a8h1 &= (filled_a8h1 >> 7) | 0x8000_0000_0000_0001;
        filled_a1h8 &= 0x0101_0101_0101_01FF;
        filled_a8h1 &= 0x8080_8080_8080_80FF;
        // broadcast
        filled_v |= filled_v << 8;
        filled_h |= filled_h << 1;
        filled_a1h8 |= (filled_a1h8 << 9) & 0x0202_0202_0202_FE00;
        filled_a8h1 |= (filled_a8h1 << 7) & 0x4040_4040_4040_7F00;
        filled_v |= filled_v << 16;
        filled_h |= filled_h << 2;
        filled_a1h8 |= (filled_a1h8 << 18) & 0x0C0C_0C0C_FCFC_0000;
        filled_a8h1 |= (filled_a8h1 << 14) & 0x3030_3030_3F3F_0000;
        filled_v |= filled_v << 32;
        filled_h |= filled_h << 4;
        filled_a1h8 |= (filled_a1h8 << 36) & 0xF0F0_F0F0_0000_0000;
        filled_a8h1 |= (filled_a8h1 << 28) & 0x0F0F_0F0F_0000_0000;
        // lines that size <= 2 are treated as filled
        filled_a1h8 |= 0x0301_0000_0000_80C0;
        filled_a8h1 |= 0xC080_0000_0000_0103;
        res |= filled_v & filled_h & filled_a1h8 & filled_a8h1;
        let res_me = res & self.player;
        let res_op = res & self.opponent;
        (res_me, res_op)
    }

    pub fn from_base81(s: &str) -> Result<Board, BoardParseError> {
        if s.len() != 16 {
            return Err(BoardParseError {});
        }
        let mut player = 0;
        let mut opponent = 0;
        for (i, b) in s.as_bytes().iter().enumerate() {
            let ofs = b - 33;
            let mut a = [0; 4];
            a[3] = ofs / 32;
            let mut rem = ofs % 32;
            a[2] = rem / 9;
            rem %= 9;
            a[1] = rem / 3;
            a[0] = rem % 3;
            for (j, t) in a.iter().enumerate() {
                match t {
                    1 => {
                        player |= 1 << (i * 4 + j);
                    }
                    2 => {
                        opponent |= 1 << (i * 4 + j);
                    }
                    0 => (),
                    _ => return Err(BoardParseError {}),
                }
            }
        }
        Ok(Board {
            player,
            opponent,
            is_black: true,
        })
    }

    pub fn to_base81(&self) -> String {
        let mut result = Vec::with_capacity(16);
        let coeff = [1, 3, 9, 32];
        for i in 0..16 {
            let mut val = 33;
            let bp = (self.player >> (i * 4)) & 0xf;
            let bo = (self.opponent >> (i * 4)) & 0xf;
            for (j, &c) in coeff.iter().enumerate() {
                if (bp >> j) & 1 > 0 {
                    val += c;
                }
                if (bo >> j) & 1 > 0 {
                    val += c * 2;
                }
            }
            result.push(val);
        }
        String::from_utf8(result).unwrap()
    }

    pub fn transform(&self, rotate: usize, mirror: bool) -> Board {
        let mut tmp = *self;
        for _ in 0..rotate {
            tmp = tmp.rot90();
        }
        if mirror {
            tmp = tmp.flip_diag();
        }
        tmp
    }

    pub fn sym_boards(&self) -> Vec<Board> {
        let mut tmp = *self;
        let mut boards = Vec::new();
        for _ in 0..4 {
            tmp = tmp.rot90();
            boards.push(tmp);
            boards.push(tmp.flip_diag());
        }
        boards
    }
}

impl FromStr for Board {
    type Err = BoardParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() <= 66 {
            return Err(BoardParseError {});
        }
        let mut black = 0u64;
        let mut white = 0u64;
        for (i, c) in s.chars().take(BOARD_SIZE).enumerate() {
            match c {
                'X' => black |= 1u64 << i,
                'O' => white |= 1u64 << i,
                _ => (),
            }
        }
        if s.chars().nth(65) == Some('X') {
            Ok(Board {
                player: black,
                opponent: white,
                is_black: true,
            })
        } else if s.chars().nth(65) == Some('O') {
            Ok(Board {
                player: white,
                opponent: black,
                is_black: false,
            })
        } else {
            Err(BoardParseError {})
        }
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..BOARD_SIZE {
            if ((self.player >> i) & 1) != 0 {
                if self.is_black {
                    write!(f, "X")?;
                } else {
                    write!(f, "O")?;
                }
            } else if ((self.opponent >> i) & 1) != 0 {
                if self.is_black {
                    write!(f, "O")?;
                } else {
                    write!(f, "X")?;
                }
            } else {
                write!(f, ".")?;
            }
            if i % 8 == 7 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl Iterator for PlayIterator {
    type Item = (Board, Hand);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remain != 0 {
            let pos = self.remain.trailing_zeros() as usize;
            self.remain &= self.remain - 1;
            if let Ok(next) = self.board.play(pos) {
                return Some((next, Hand::Play(pos)));
            }
        }
        None
    }
}

pub fn weighted_mobility(board: &Board) -> i8 {
    let b = board.mobility_bits();
    let corner = 0x8100000000000081u64;
    popcnt(b) + popcnt(b & corner)
}

fn stable_bits_8(board: Board, passed: bool, memo: &mut [Option<u64>]) -> u64 {
    let index = BASE3[board.player as usize] + 2 * BASE3[board.opponent as usize];
    if let Some(res) = memo[index] {
        return res;
    }
    let mut res = 0xFF;
    for pos in 0..8 {
        if ((board.empty() >> pos) & 1) != 1 {
            continue;
        }
        let flip = board.flip(pos);
        let pos_bit = 1 << pos;
        let next = Board {
            player: board.opponent ^ flip,
            opponent: (board.player ^ flip) | pos_bit,
            is_black: !board.is_black,
        };
        res &= !flip;
        res &= !pos_bit;
        res &= stable_bits_8(next, false, memo);
    }
    if !passed {
        let next = board.pass();
        res &= stable_bits_8(next, true, memo);
        memo[index] = Some(res);
    }
    res
}

pub fn parse_board(matches: &ArgMatches) {
    let s = matches.get_one::<String>("str").unwrap();

    let data: Vec<&str> = s.split(' ').collect();
    let player = u64::from_str_radix(data[0], 16).unwrap();
    let opponent = u64::from_str_radix(data[1], 16).unwrap();
    let board = Board {
        player,
        opponent,
        is_black: true,
    };
    println!("{}", board);
}

lazy_static! {
    static ref STABLE: [u64; 6561] = {
        let mut memo = [None; 6561];
        for i in 0..6561 {
            let mut me = 0;
            let mut op = 0;
            let mut tmp = i;
            for j in 0..8 {
                let state = tmp % 3;
                match state {
                    1 => me |= 1 << j,
                    2 => op |= 1 << j,
                    _ => (),
                }
                tmp /= 3;
            }
            let board = Board {
                player: me,
                opponent: op,
                is_black: true,
            };
            stable_bits_8(board, false, &mut memo);
        }
        let mut res = [0; 6561];
        for i in 0..6561 {
            res[i] = memo[i].unwrap() & 0xFF;
        }
        res
    };
}
