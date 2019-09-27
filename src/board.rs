#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Board {
    pub player: u64,
    pub opponent: u64,
    pub is_black: bool
}

pub struct UnmovableError;

use std::io::Write;
use std::str::FromStr;
use packed_simd::*;
use crate::bits::*;

impl Board {
    fn flip_simd(&self, pos: usize) -> u64x4 {
        let p = u64x4::new(self.player, self.player, self.player, self.player);
        let o = u64x4::new(self.opponent, self.opponent, self.opponent, self.opponent);
        let omask = u64x4::new(0xFFFFFFFFFFFFFFFFu64,
                               0x7E7E7E7E7E7E7E7Eu64,
                               0x7E7E7E7E7E7E7E7Eu64,
                               0x7E7E7E7E7E7E7E7Eu64);
        let om = o & omask;
        let mask1 = u64x4::new(
            0x0080808080808080u64,
            0x7f00000000000000u64,
            0x0102040810204000u64,
            0x0040201008040201u64
        );
        let mut mask = mask1 >> (63 - pos) as u32;
        let mut outflank = upper_bit(!om & mask) & p;
        let mut flipped = u64x4::from_cast(-i64x4::from_cast(outflank) << 1) & mask;
        let mask2 = u64x4::new(
            0x0101010101010100u64,
            0x00000000000000feu64,
            0x0002040810204080u64,
            0x8040201008040200u64
        );
        mask = mask2 << pos as u32;
        outflank = mask & ((om | !mask) + 1) & p;
        flipped |= (outflank - nonzero(outflank)) & mask;
        return flipped;
    }

    pub fn flip(&self, pos: usize) -> u64 {
        let flips = self.flip_simd(pos);
        flips.or()
    }

    pub fn is_movable(&self, pos: usize) -> bool {
        if pos >= 64 {
            return false;
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return false;
        }
        self.flip(pos) != 0
    }

    pub fn play(&self, pos: usize) -> Result<Board, UnmovableError> {
        if pos >= 64 {
            return Err(UnmovableError{});
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return Err(UnmovableError{});
        }
        let flip_bits = self.flip(pos);
        if flip_bits == 0 {
            return Err(UnmovableError{});
        }
        Ok(Board {
            player: self.opponent ^ flip_bits,
            opponent: (self.player ^ flip_bits) | (1u64 << pos),
            is_black: !self.is_black
        })
    }

    pub fn pass(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
            is_black: !self.is_black
        }
    }

    pub fn empty(&self) -> u64 {
        !(self.player | self.opponent)
    }

    pub fn mobility_bits(&self) -> u64 {
        let shift1 = u64x4::new(1, 7, 9, 8);
        let mask = u64x4::new(
            0x7e7e7e7e7e7e7e7eu64,
            0x7e7e7e7e7e7e7e7eu64,
            0x7e7e7e7e7e7e7e7eu64,
            0xffffffffffffffffu64
        );
        let v_player = u64x4::splat(self.player);
        let masked_op = u64x4::splat(self.opponent) & mask;
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
        res &= u64x4::splat(self.empty());
        return res.or();
    }

    pub fn mobility(&self) -> Vec<usize> {
        let mut result = Vec::new();
        let mut empties = self.empty();
        while empties != 0 {
            let bit = empties  & empties.wrapping_neg();
            empties = empties & (empties - 1);
            let pos = popcnt(bit - 1) as usize;
            if self.is_movable(pos) {
                result.push(pos);
            }
        }
        result
    }

    pub fn is_gameover(&self) -> bool {
        if !self.mobility().is_empty() {
            return false;
        }
        self.pass().mobility().is_empty()
    }

    pub fn print(&self) -> () {
        for i in 0..64 {
            if ((self.player >> i) & 1) != 0 {
                if self.is_black {
                    write!(std::io::stdout(), "X").unwrap();
                } else {
                    write!(std::io::stdout(), "O").unwrap();
                }
            } else if ((self.opponent >> i) & 1) != 0 {
                if self.is_black {
                    write!(std::io::stdout(), "O").unwrap();
                } else {
                    write!(std::io::stdout(), "X").unwrap();
                }
            } else {
                write!(std::io::stdout(), "-").unwrap();
            }
            if i % 8 == 7 {
                write!(std::io::stdout(), "\n").unwrap();
            }
        }
    }

    pub fn score(&self) -> i8 {
        let pcnt = popcnt(self.player);
        let ocnt = popcnt(self.opponent);
        if pcnt == ocnt {
            0
        } else if pcnt > ocnt {
            64 - 2*ocnt
        } else {
            -64 + 2*pcnt
        }
    }

    pub fn flip_diag(&self) -> Board {
        Board {
            player: flip_diag(self.player),
            opponent: flip_diag(self.opponent),
            is_black: self.is_black
        }
    }

    pub fn rot90(&self) -> Board {
        Board {
            player: rot90(self.player),
            opponent: rot90(self.opponent),
            is_black: self.is_black
        }
    }
}

#[derive(Debug)]
pub struct BoardParseError;

impl FromStr for Board {
    type Err = BoardParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() <= 66 {
            return Err(BoardParseError{});
        }
        let mut black = 0u64;
        let mut white = 0u64;
        for (i, c) in s.chars().take(64).enumerate() {
            match c {
                'X' => black |= 1u64 << i,
                'O' => white |= 1u64 << i,
                _ => ()
            }
        }
        if s.chars().nth(65) == Some('X') {
            Ok(Board{ player: black, opponent: white, is_black: true })
        } else if s.chars().nth(65) == Some('O') {
            Ok(Board{ player: white, opponent: black, is_black: false })
        } else {
            Err(BoardParseError{})
        }
    }
}

pub fn weighted_mobility(board: & Board) -> i8 {
    let b = board.mobility_bits();
    let corner = 0x8100000000000081u64;
    popcnt(b) + popcnt(b & corner)
}
