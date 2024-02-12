#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::hand::*;
use anyhow::Result;
use clap::ArgMatches;
use std::cmp::min;
use std::fmt;
use std::io::{BufWriter, Write};
use std::simd::prelude::*;
use std::str::FromStr;
use thiserror::Error;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Board {
    pub player: u64,
    pub opponent: u64,
}

#[derive(Error, Debug)]
#[error("Unmovable hand")]
pub struct UnmovableError;

#[derive(Error, Debug)]
#[error("Parse error")]
pub struct BoardParseError;

pub struct PlayIterator {
    board: Board,
    remain: u64,
}

pub const BOARD_SIZE: usize = 64;

#[cfg(all(target_feature = "avx512cd", target_feature = "avx512vl"))]
fn smart_upper_bit(x: u64x4) -> u64x4 {
    let y = x.leading_zeros();
    0x8000_0000_0000_0000u64 >> y
}

#[cfg(all(
    target_feature = "avx2",
    not(all(target_feature = "avx512cd", target_feature = "avx512vl"))
))]
fn smart_upper_bit(mut x: u64x4) -> u64x4 {
    x |= x >> u64x4::from_array([8, 1, 7, 9]);
    x |= x >> u64x4::from_array([16, 2, 14, 18]);
    x |= x >> u64x4::from_array([32, 4, 28, 36]);
    let lowers = x >> Simd::from_array([8, 1, 7, 9]);
    !lowers & x
}

const fn smart_upper_bit_scalar(mut x: u64, lane: usize) -> u64 {
    x |= x >> [8, 1, 7, 9][lane];
    x |= x >> [16, 2, 14, 18][lane];
    x |= x >> [32, 4, 28, 36][lane];
    let lowers = x >> [8, 1, 7, 9][lane];
    !lowers & x
}

#[allow(dead_code)]
fn upper_bit(mut x: u64x4) -> u64x4 {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    let lowers = x >> 1;
    !lowers & x
}

fn iszero(x: u64x4) -> u64x4 {
    x.simd_eq(Simd::splat(0u64)).to_int().cast()
}

impl Board {
    pub fn initial_state() -> Board {
        Board {
            player: 0x00_00_00_08_10_00_00_00,
            opponent: 0x00_00_00_10_08_00_00_00,
        }
    }

    pub fn reverse_vertical(&self) -> Board {
        Board {
            player: flip_vertical(self.player),
            opponent: flip_vertical(self.opponent),
        }
    }

    fn flip_simd(&self, pos: usize) -> u64 {
        let p = Simd::splat(self.player);
        let o = Simd::splat(self.opponent);
        let omask = Simd::from_array([
            0xFFFFFFFFFFFFFFFFu64,
            0x7E7E7E7E7E7E7E7Eu64,
            0x7E7E7E7E7E7E7E7Eu64,
            0x7E7E7E7E7E7E7E7Eu64,
        ]);
        let om = o & omask;
        let mask1 = Simd::from_array([
            0x0080808080808080u64,
            0x7f00000000000000u64,
            0x0102040810204000u64,
            0x0040201008040201u64,
        ]);
        let mut mask = mask1 >> ((63 - pos) as u64);
        let mut outflank = smart_upper_bit(!om & mask) & p;
        let mut flipped = ((Simd::splat(0) - outflank) << 1) & mask;
        let mask2 = Simd::from_array([
            0x0101010101010100u64,
            0x00000000000000feu64,
            0x0002040810204080u64,
            0x8040201008040200u64,
        ]);
        mask = mask2 << (pos as u64);
        outflank = !((!om & mask) - Simd::splat(1)) & mask & p;
        flipped |= !(iszero(outflank) - outflank) & mask;
        flipped.reduce_or()
    }

    pub const fn flip_scalar(&self, pos: usize) -> u64 {
        let o_mask = 0x7E7E_7E7E_7E7E_7E7Eu64;
        let om = [
            self.opponent,
            self.opponent & o_mask,
            self.opponent & o_mask,
            self.opponent & o_mask,
        ];
        let mask1 = [
            0x0080808080808080u64,
            0x7f00000000000000u64,
            0x0102040810204000u64,
            0x0040201008040201u64,
        ];
        let mask2 = [
            0x0101010101010100u64,
            0x00000000000000feu64,
            0x0002040810204080u64,
            0x8040201008040200u64,
        ];
        let mut flipped = 0;
        let mut i = 0;
        while i < 4 {
            let mask = mask1[i] >> (63 - pos);
            let outflank = smart_upper_bit_scalar(!om[i] & mask, i) & self.player;
            flipped |= (outflank.wrapping_neg() << 1) & mask;
            let mask = mask2[i] << pos;
            let outflank = !((!om[i] & mask).wrapping_sub(1)) & mask & self.player;
            flipped |= !((if outflank == 0 {
                0xFFFF_FFFF_FFFF_FFFFu64
            } else {
                0
            })
            .wrapping_sub(outflank))
                & mask;
            i += 1;
        }
        flipped
    }

    pub fn flip_unchecked(&self, pos: usize) -> u64 {
        self.flip_simd(pos)
    }

    pub fn flip(&self, pos: usize) -> u64 {
        if ((self.empty() >> pos) & 1) == 0 {
            0
        } else {
            self.flip_unchecked(pos)
        }
    }

    pub const fn flip_const(&self, pos: usize) -> u64 {
        if ((self.empty() >> pos) & 1) == 0 {
            0
        } else {
            self.flip_scalar(pos)
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

    pub fn play(&self, pos: usize) -> Option<Board> {
        if pos >= BOARD_SIZE {
            return None;
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return None;
        }
        let flip_bits = self.flip(pos);
        if flip_bits == 0 {
            return None;
        }
        Some(Board {
            player: self.opponent ^ flip_bits,
            opponent: (self.player ^ flip_bits) | (1u64 << pos),
        })
    }

    pub fn play_hand(&self, hand: Hand) -> Option<Board> {
        match hand {
            Hand::Play(pos) => self.play(pos),
            Hand::Pass => self.pass(),
        }
    }

    pub const fn pass_unchecked(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
        }
    }

    pub fn pass(&self) -> Option<Board> {
        if self.mobility_bits() == 0 {
            Some(Board {
                player: self.opponent,
                opponent: self.player,
            })
        } else {
            None
        }
    }

    pub const fn empty(&self) -> u64 {
        !(self.player | self.opponent)
    }

    fn mobility_bits_simd(&self) -> u64 {
        let shift1 = Simd::from_array([1, 7, 9, 8]);
        let mask = Simd::from_array([
            0x7e7e7e7e7e7e7e7eu64,
            0x7e7e7e7e7e7e7e7eu64,
            0x7e7e7e7e7e7e7e7eu64,
            0xffffffffffffffffu64,
        ]);
        let v_player = Simd::splat(self.player);
        let masked_op = Simd::splat(self.opponent) & mask;
        let mut flip_l = masked_op & (v_player << shift1);
        let mut flip_r = masked_op & (v_player >> shift1);
        flip_l |= masked_op & (flip_l << shift1);
        flip_r |= masked_op & (flip_r >> shift1);
        let pre_l = masked_op & (masked_op << shift1);
        let pre_r = pre_l >> shift1;
        let shift2 = shift1 + shift1;
        flip_l |= pre_l & (flip_l << shift2);
        flip_r |= pre_r & (flip_r >> shift2);
        flip_l |= pre_l & (flip_l << shift2);
        flip_r |= pre_r & (flip_r >> shift2);
        let mut res = flip_l << shift1;
        res |= flip_r >> shift1;
        let res = res.reduce_or();
        res & self.empty()
    }

    pub fn mobility_bits(&self) -> u64 {
        self.mobility_bits_simd()
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
        let remain = if popcnt(e) > 4 {
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
        self.pass_unchecked().mobility().is_empty()
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
        }
    }

    pub fn rot90(&self) -> Board {
        Board {
            player: rot90(self.player),
            opponent: rot90(self.opponent),
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
        Ok(Board { player, opponent })
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

    #[allow(dead_code)]
    pub fn normalize(&self) -> (Board, usize, bool) {
        let mut res = (*self, 0, false);
        let mut tmp = *self;
        for i in 0..4 {
            res = min(res, (tmp, i, false));
            res = min(res, (tmp.flip_diag(), i, true));
            tmp = tmp.rot90();
        }
        res
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..BOARD_SIZE {
            if ((self.player >> i) & 1) != 0 {
                write!(f, "X")?;
            } else if ((self.opponent >> i) & 1) != 0 {
                write!(f, "O")?;
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BoardWithColor {
    pub board: Board,
    pub is_black: bool,
}

impl BoardWithColor {
    pub fn initial_state() -> BoardWithColor {
        BoardWithColor {
            board: Board::initial_state(),
            is_black: true,
        }
    }

    pub fn play(&self, pos: usize) -> Option<BoardWithColor> {
        let board = self.board.play(pos)?;
        Some(BoardWithColor {
            board,
            is_black: !self.is_black,
        })
    }

    pub fn pass_unchecked(&self) -> BoardWithColor {
        BoardWithColor {
            board: self.board.pass_unchecked(),
            is_black: !self.is_black,
        }
    }

    #[allow(dead_code)]
    pub fn pass(&self) -> Option<BoardWithColor> {
        Some(BoardWithColor {
            board: self.board.pass()?,
            is_black: !self.is_black,
        })
    }

    pub fn empty(&self) -> u64 {
        self.board.empty()
    }

    pub fn is_gameover(&self) -> bool {
        self.board.is_gameover()
    }

    pub fn score(&self) -> i8 {
        let raw_score = self.board.score();
        if self.is_black {
            raw_score
        } else {
            -raw_score
        }
    }

    pub fn print_with_sides(&self) {
        let mut writer = BufWriter::new(std::io::stdout());
        write!(writer, " |abcdefgh\n-+--------\n").unwrap();
        for r in 0..8 {
            write!(writer, "{}|", r + 1).unwrap();
            for c in 0..8 {
                let i = r * 8 + c;
                if ((self.board.player >> i) & 1) != 0 {
                    if self.is_black {
                        write!(writer, "X").unwrap();
                    } else {
                        write!(writer, "O").unwrap();
                    }
                } else if ((self.board.opponent >> i) & 1) != 0 {
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
}

impl FromStr for BoardWithColor {
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
            Ok(BoardWithColor {
                board: Board {
                    player: black,
                    opponent: white,
                },
                is_black: true,
            })
        } else if s.chars().nth(65) == Some('O') {
            Ok(BoardWithColor {
                board: Board {
                    player: white,
                    opponent: black,
                },
                is_black: false,
            })
        } else {
            Err(BoardParseError {})
        }
    }
}

impl Iterator for PlayIterator {
    type Item = (Board, Hand);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remain != 0 {
            let pos = self.remain.trailing_zeros() as usize;
            self.remain &= self.remain - 1;
            if let Some(next) = self.board.play(pos) {
                return Some((next, Hand::Play(pos)));
            }
        }
        None
    }
}

impl fmt::Display for BoardWithColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..BOARD_SIZE {
            if ((self.board.player >> i) & 1) != 0 {
                if self.is_black {
                    write!(f, "X")?;
                } else {
                    write!(f, "O")?;
                }
            } else if ((self.board.opponent >> i) & 1) != 0 {
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

pub fn weighted_mobility(board: &Board) -> i8 {
    let b = board.mobility_bits();
    let corner = 0x8100000000000081u64;
    popcnt(b) + popcnt(b & corner)
}

const fn stable_bits_8(board: Board, passed: bool, memo: &[Option<u64>]) -> u64 {
    let index = BASE3[board.player as usize] + 2 * BASE3[board.opponent as usize];
    if let Some(res) = memo[index] {
        return res;
    }
    let mut res = 0xFF;
    let mut pos = 0;
    while pos < 8 {
        if ((board.empty() >> pos) & 1) != 1 {
            pos += 1;
            continue;
        }
        let flip = board.flip_const(pos);
        let pos_bit = 1 << pos;
        let next = Board {
            player: board.opponent ^ flip,
            opponent: (board.player ^ flip) | pos_bit,
        };
        res &= !flip;
        res &= !pos_bit;
        res &= stable_bits_8(next, false, memo);
        pos += 1;
    }
    if !passed {
        let next = board.pass_unchecked();
        res &= stable_bits_8(next, true, memo);
        //memo[index] = Some(res);
    }
    res
}

pub fn parse_board(matches: &ArgMatches) {
    let s = matches.get_one::<String>("str").unwrap();

    let data: Vec<&str> = s.split(' ').collect();
    let player = u64::from_str_radix(data[0], 16).unwrap();
    let opponent = u64::from_str_radix(data[1], 16).unwrap();
    let board = Board { player, opponent };
    println!("{}", board);
}

const STABLE: [u64; 6561] = {
    let mut memo = [None; 6561];
    let mut ri = 0;
    while ri < 6561 {
        let i = 6561 - ri - 1;
        let mut me = 0;
        let mut op = 0;
        let mut tmp = i;
        let mut j = 0;
        while j < 8 {
            let state = tmp % 3;
            match state {
                1 => me |= 1 << j,
                2 => op |= 1 << j,
                _ => (),
            }
            tmp /= 3;
            j += 1;
        }
        let board = Board {
            player: me,
            opponent: op,
        };
        let res = stable_bits_8(board, false, &memo);
        memo[i] = Some(res);
        ri += 1;
    }
    let mut res = [0; 6561];
    let mut i = 0;
    while i < 6561 {
        res[i] = memo[i].unwrap() & 0xFF;
        i += 1;
    }
    res
};
