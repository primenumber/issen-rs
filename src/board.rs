use std::io::{BufWriter, Write};
use std::str::FromStr;
use packed_simd::*;
use crate::bits::*;
use lazy_static::lazy_static;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Board {
    pub player: u64,
    pub opponent: u64,
    pub is_black: bool
}

#[derive(Debug)]
pub struct UnmovableError;

#[derive(Debug)]
pub struct BoardParseError;

pub const PASS: usize = 64;

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

    pub fn flip_unchecked(&self, pos: usize) -> u64 {
        let flips = self.flip_simd(pos);
        flips.or()
    }

    pub fn flip(&self, pos: usize) -> u64 {
        if ((self.empty() >> pos) & 1) == 0 {
            0
        } else {
            self.flip_unchecked(pos)
        }
    }

    pub fn is_movable(&self, pos: usize) -> bool {
        if pos >= PASS {
            return false;
        }
        if ((self.player >> pos) & 1) != 0 || ((self.opponent >> pos) & 1) != 0 {
            return false;
        }
        self.flip(pos) != 0
    }

    pub fn play(&self, pos: usize) -> Result<Board, UnmovableError> {
        if pos >= PASS {
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
        let mut writer = BufWriter::new(std::io::stdout());
        for i in 0..64 {
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
                write!(writer, "-").unwrap();
            }
            if i % 8 == 7 {
                write!(writer, "\n").unwrap();
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

    pub fn stable_partial(&self) -> (u64, u64) {
        const MASK_TOP: u64 = 0x0000_0000_0000_00FF;
        const MASK_BOTTOM: u64 = 0xFF00_0000_0000_0000;
        const MASK_LEFT: u64 = 0x0101_0101_0101_0101;
        const MASK_RIGHT: u64 = 0x8080_8080_8080_8080;
        const MASKS: [u64; 4] = [
            MASK_TOP, MASK_BOTTOM, MASK_LEFT, MASK_RIGHT
        ];
        let mut res = 0;
        for mask in &MASKS {
            let me = pext(self.player, *mask) as usize;
            let op = pext(self.opponent, *mask) as usize;
            let base3 = BASE3[me] + 2 * BASE3[op];
            res |= pdep(STABLE[base3], *mask);
        }
        for r in 0..8 {
            let mask_h = MASK_TOP << (r * 8);
            for c in 0..8 {
                let mask_v = MASK_LEFT << c;
                const MASK_D_A1H8: u64 = 0x8040201008040201;
                let mask_d_a1h8 = if (r - c) >= 0 {
                    MASK_D_A1H8 << ((r - c) * 8)
                } else {
                    MASK_D_A1H8 >> ((c - r) * 8)
                };
                const MASK_D_A8H1: u64 = 0x0102040810204080;
                let mask_d_a8h1 = if (r + c - 7) >= 0 {
                    MASK_D_A8H1 << ((r + c - 7) * 8)
                } else {
                    MASK_D_A8H1 >> ((7 - r - c) * 8)
                };
                let mask = mask_h | mask_v | mask_d_a1h8 | mask_d_a8h1;
                let pos = r * 8 + c;
                if (self.empty() & mask) == 0 {
                    res |= 1 << pos;
                }
            }
        }
        let res_me = res & self.player;
        let res_op = res & self.opponent;
        (res_me, res_op)
    }

    pub fn from_base81(s: &str) -> Result<Board, BoardParseError> {
        if s.len() != 16 {
            return Err(BoardParseError{});
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
                    },
                    2 => {
                        opponent |= 1 << (i * 4 + j);
                    },
                    0 => (),
                    _ => return Err(BoardParseError{})
                }
            }
        }
        Ok(Board{ player, opponent, is_black: true })
    }
}

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

fn stable_bits_8(board: Board, passed: bool, memo: &mut [Option<u64>]) -> u64 {
    let index = BASE3[board.player as usize] + 2 * BASE3[board.opponent as usize];
    match memo[index] {
        Some(res) => return res,
        None => ()
    }
    let mut res = 0xFFFF_FFFF_FFFF_FFFF;
    let mut pass = true;
    for next_pos in board.mobility() {
        pass = false;
        let next = board.play(next_pos).unwrap();
        let flip = board.flip(next_pos);
        res &= !flip;
        res &= !(1 << next_pos);
        res &= stable_bits_8(next, false, memo);
    }
    if pass {
        if !passed {
            let next = board.pass();
            res &= stable_bits_8(next, true, memo);
        }
    }
    for pos in 0..8 {
        if ((board.empty() >> pos) & 1) != 1 {
            continue;
        }
        let flip = board.flip(pos);
        if flip != 0 {
            continue;
        }
        let next = Board {
            player: board.opponent,
            opponent: board.player | (1 << pos),
            is_black: !board.is_black
        };
        res &= !(1 << pos);
        res &= stable_bits_8(next, false, memo);
    }
    memo[index] = Some(res);
    return res;
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
                    1 => {me |= 1 << j},
                    2 => {op |= 1 << j},
                    _ => ()
                }
                tmp /= 3;
            }
            let board = Board{
                player: me,
                opponent: op,
                is_black: true
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Empty,
        Player,
        Opponent
    }

    #[derive(Clone)]
    struct NaiveBoard {
        data: [State; 64],
        is_black: bool
    }

    impl From<Board> for NaiveBoard {
        fn from(board: Board) -> Self {
            let mut res = NaiveBoard {
                data: [State::Empty; 64],
                is_black: board.is_black
            };
            for i in 0..64 {
                if ((board.player >> i) & 1) == 1 {
                    res.data[i] = State::Player;
                } else if ((board.opponent >> i) & 1) == 1 {
                    res.data[i] = State::Opponent;
                }
            }
            res
        }
    }

    impl From<NaiveBoard> for Board {
        fn from(naive_board: NaiveBoard) -> Self {
            let mut player = 0;
            let mut opponent = 0;
            for i in 0..64 {
                match naive_board.data[i] {
                    State::Player => {
                        player |= 1 << i;
                    },
                    State::Opponent => {
                        opponent |= 1 << i;
                    },
                    _ => ()
                }
            }
            Board {
                player,
                opponent,
                is_black: naive_board.is_black
            }
        }
    }

    impl NaiveBoard {
        fn flip(&self, pos: usize) -> u64 {
            const DELTA: [(isize, isize); 8] = [
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1)
            ];
            if self.data[pos] != State::Empty {
                return 0;
            }
            let r = (pos / 8) as isize;
            let c = (pos % 8) as isize;
            let mut res = 0;
            for (dr, dc) in DELTA.iter() {
                for i in 1..8 {
                    let nr = r + i * dr;
                    let nc = c + i * dc;
                    if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 {
                        break;
                    }
                    let ni = (nr * 8 + nc) as usize;
                    match self.data[ni] {
                        State::Player => {
                            for j in 1..i {
                                let mr = r + j * dr;
                                let mc = c + j * dc;
                                let mi = mr * 8 + mc;
                                res |= 1 << mi;
                            }
                            break;
                        },
                        State::Empty => {
                            break;
                        },
                        State::Opponent => ()
                    }
                }
            }
            res
        }

        fn is_movable(&self, pos: usize) -> bool {
            if pos >= PASS {
                return false;
            }
            if self.data[pos] != State::Empty {
                return false;
            }
            self.flip(pos) != 0
        }

        fn play(&self, pos: usize) -> Result<NaiveBoard, UnmovableError> {
            if pos >= PASS {
                return Err(UnmovableError{});
            }
            if self.data[pos] != State::Empty {
                return Err(UnmovableError{});
            }
            let flip_bits = self.flip(pos);
            if flip_bits == 0 {
                return Err(UnmovableError{});
            }
            let mut res = NaiveBoard {
                data: [State::Empty; 64],
                is_black: !self.is_black
            };
            for i in 0..64 {
                if ((flip_bits >> i) & 1) == 1 {
                    res.data[i] = State::Opponent;
                } else if self.data[i] == State::Player {
                    res.data[i] = State::Opponent;
                } else if self.data[i] == State::Opponent {
                    res.data[i] = State::Player;
                } else if i == pos {
                    res.data[i] = State::Opponent;
                }
            }
            Ok(res)
        }

        fn empty(&self) -> u64 {
            let mut res = 0;
            for i in 0..64 {
                if self.data[i] == State::Empty {
                    res |= 1 << i;
                }
            }
            res
        }

        fn mobility_bits(&self) -> u64 {
            let mut res = 0;
            for i in 0..64 {
                if self.is_movable(i) {
                    res |= 1 << i;
                }
            }
            res
        }

        fn mobility(&self) -> Vec<usize> {
            let mut res = Vec::new();
            for i in 0..64 {
                if self.is_movable(i) {
                    res.push(i);
                }
            }
            res
        }

        fn score(&self) -> i8 {
            let mut pcnt = 0;
            let mut ocnt = 0;
            for i in 0..64 {
                match self.data[i] {
                    State::Player => {
                        pcnt += 1;
                    },
                    State::Opponent => {
                        ocnt += 1;
                    },
                    _ => ()
                }
            }
            if pcnt == ocnt {
                0
            } else if pcnt > ocnt {
                64 - 2*ocnt
            } else {
                -64 + 2*pcnt
            }
        }
    }

    #[test]
    fn test_ops() {
        const TEST_BASE81: &str = "!#jiR;rO[ORNM2MN";
        let board = Board::from_base81(TEST_BASE81).unwrap();
        let naive_board = NaiveBoard::from(board.clone());
        assert_eq!(board.clone(), Board::from(naive_board.clone()));
        for i in 0..64 {
            assert_eq!(board.flip(i), naive_board.flip(i));
            assert_eq!(board.is_movable(i), naive_board.is_movable(i));
            if board.is_movable(i) {
                assert_eq!(board.play(i).unwrap(), Board::from(naive_board.play(i).unwrap()));
            }
        }
        assert_eq!(board.empty(), naive_board.empty());
        assert_eq!(board.mobility_bits(), naive_board.mobility_bits());
        assert_eq!(board.mobility(), naive_board.mobility());
        assert_eq!(board.score(), naive_board.score());
    }
}
