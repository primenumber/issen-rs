use crate::bits::*;
use crate::serialize::*;
use clap::ArgMatches;
use lazy_static::lazy_static;
use packed_simd::*;
use std::fmt;
use std::fs::File;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hand {
    Play(usize),
    Pass,
}

impl Hand {
    fn rot90(&self) -> Hand {
        match self {
            Self::Pass => *self,
            Self::Play(pos) => {
                let row = pos / 8;
                let col = pos % 8;
                Self::Play((7 - col) * 8 + row)
            }
        }
    }

    fn flip_diag(&self) -> Hand {
        match self {
            Self::Pass => *self,
            Self::Play(pos) => {
                let row = pos / 8;
                let col = pos % 8;
                Self::Play(col * 8 + row)
            }
        }
    }

    pub fn transform(&self, rotate: usize, mirror: bool) -> Hand {
        let mut tmp = *self;
        for _ in 0..rotate {
            tmp = tmp.rot90();
        }
        if mirror {
            tmp = tmp.flip_diag();
        }
        tmp
    }
}

pub const BOARD_SIZE: usize = 64;

pub const PASS: usize = 64;

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

    fn flip_simd(&self, pos: usize) -> u64x4 {
        let p = u64x4::splat(self.player);
        let o = u64x4::splat(self.opponent);
        let omask = u64x4::new(
            0xFFFFFFFFFFFFFFFFu64,
            0x7E7E7E7E7E7E7E7Eu64,
            0x7E7E7E7E7E7E7E7Eu64,
            0x7E7E7E7E7E7E7E7Eu64,
        );
        let om = o & omask;
        let mask1 = u64x4::new(
            0x0080808080808080u64,
            0x7f00000000000000u64,
            0x0102040810204000u64,
            0x0040201008040201u64,
        );
        let mut mask = mask1 >> (63 - pos) as u32;
        let mut outflank = upper_bit(!om & mask) & p;
        let mut flipped = u64x4::from_cast(-i64x4::from_cast(outflank) << 1) & mask;
        let mask2 = u64x4::new(
            0x0101010101010100u64,
            0x00000000000000feu64,
            0x0002040810204080u64,
            0x8040201008040200u64,
        );
        mask = mask2 << pos as u32;
        outflank = !((!om & mask) - 1) & (mask & p);
        flipped |= !(iszero(outflank) - outflank) & mask;
        flipped
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

    pub fn mobility_bits(&self) -> u64 {
        let shift1 = u64x4::new(1, 7, 9, 8);
        let mask = u64x4::new(
            0x7e7e7e7e7e7e7e7eu64,
            0x7e7e7e7e7e7e7e7eu64,
            0x7e7e7e7e7e7e7e7eu64,
            0xffffffffffffffffu64,
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
        res.or()
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
        PlayIterator {
            board: *self,
            remain: self.empty(),
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
            64 - 2 * ocnt
        } else {
            -64 + 2 * pcnt
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
        //const MASK_BOTTOM: u64 = 0xFF00_0000_0000_0000;
        const MASK_LEFT: u64 = 0x0101_0101_0101_0101;
        //const MASK_RIGHT: u64 = 0x8080_8080_8080_8080;
        //const MASKS: [u64; 4] = [MASK_TOP, MASK_BOTTOM, MASK_LEFT, MASK_RIGHT];
        let mut res = 0;
        // FIXME: edge stability is buggy
        //for mask in &MASKS {
        //    let me = pext(self.player, *mask) as usize;
        //    let op = pext(self.opponent, *mask) as usize;
        //    let base3 = BASE3[me] + 2 * BASE3[op];
        //    res |= pdep(STABLE[base3], *mask);
        //}
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

    #[allow(dead_code)]
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
        for (i, c) in s.chars().take(64).enumerate() {
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
        for i in 0..64 {
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
    if pass && !passed {
        let next = board.pass();
        res &= stable_bits_8(next, true, memo);
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
            is_black: !board.is_black,
        };
        res &= !(1 << pos);
        res &= stable_bits_8(next, false, memo);
    }
    memo[index] = Some(res);
    res
}

pub fn gen_last_table(matches: &ArgMatches) {
    let output_path = matches.value_of("OUTPUT").unwrap();

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let mut v = Vec::new();
    for pos in 0..8 {
        for bits in 0..256 {
            let board = Board {
                player: bits,
                opponent: !(bits | (1 << pos)),
                is_black: true,
            };
            let fcnt = popcnt(board.flip(pos));
            v.push(fcnt);
        }
    }
    for chunk in v.chunks(2) {
        let val = chunk[0] | (chunk[1] << 3);
        write!(writer, "{}", encode_base64_impl(val as u8).unwrap() as char).unwrap();
    }
    writeln!(writer).unwrap();
}

pub fn gen_last_mask(matches: &ArgMatches) {
    let output_path = matches.value_of("OUTPUT").unwrap();

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    for pos in 0..64i8 {
        let mut masks: [u64; 3] = [0, 0, 0];
        let pr = pos / 8;
        let pc = pos % 8;
        for row in 0..8 {
            for col in 0..8 {
                let index = row * 8 + col;
                if col == pc {
                    masks[0] |= 1u64 << index;
                }
                if row + pc == pr + col {
                    masks[1] |= 1u64 << index;
                }
                if row + col == pr + pc {
                    masks[2] |= 1u64 << index;
                }
            }
        }
        for mask in masks {
            for row in 0..8 {
                let mask_in_the_row = (mask >> (8 * row)) & 0xff;
                let col = if mask_in_the_row == 0 {
                    8
                } else {
                    mask_in_the_row.trailing_zeros()
                };
                write!(writer, "{}", col).unwrap();
            }
        }
    }
    writeln!(writer).unwrap();
}

pub fn parse_board(matches: &ArgMatches) {
    let s = matches.value_of("str").unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Empty,
        Player,
        Opponent,
    }

    #[derive(Clone)]
    struct NaiveBoard {
        data: [State; 64],
        is_black: bool,
    }

    impl From<Board> for NaiveBoard {
        fn from(board: Board) -> Self {
            let mut res = NaiveBoard {
                data: [State::Empty; 64],
                is_black: board.is_black,
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
                    }
                    State::Opponent => {
                        opponent |= 1 << i;
                    }
                    _ => (),
                }
            }
            Board {
                player,
                opponent,
                is_black: naive_board.is_black,
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
                (1, -1),
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
                        }
                        State::Empty => {
                            break;
                        }
                        State::Opponent => (),
                    }
                }
            }
            res
        }

        fn is_movable(&self, pos: usize) -> bool {
            if pos >= BOARD_SIZE {
                return false;
            }
            if self.data[pos] != State::Empty {
                return false;
            }
            self.flip(pos) != 0
        }

        fn play(&self, pos: usize) -> Result<NaiveBoard, UnmovableError> {
            if pos >= BOARD_SIZE {
                return Err(UnmovableError {});
            }
            if self.data[pos] != State::Empty {
                return Err(UnmovableError {});
            }
            let flip_bits = self.flip(pos);
            if flip_bits == 0 {
                return Err(UnmovableError {});
            }
            let mut res = NaiveBoard {
                data: [State::Empty; 64],
                is_black: !self.is_black,
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
                    }
                    State::Opponent => {
                        ocnt += 1;
                    }
                    _ => (),
                }
            }
            if pcnt == ocnt {
                0
            } else if pcnt > ocnt {
                64 - 2 * ocnt
            } else {
                -64 + 2 * pcnt
            }
        }
    }

    #[test]
    fn test_ops() {
        const TEST_BASE81: &str = "!#jiR;rO[ORNM2MN";
        let board = Board::from_base81(TEST_BASE81).unwrap();
        let naive_board = NaiveBoard::from(board);
        assert_eq!(board, Board::from(naive_board.clone()));
        for i in 0..64 {
            assert_eq!(board.flip(i), naive_board.flip(i));
            assert_eq!(board.is_movable(i), naive_board.is_movable(i));
            if board.is_movable(i) {
                assert_eq!(
                    board.play(i).unwrap(),
                    Board::from(naive_board.play(i).unwrap())
                );
            }
        }
        assert_eq!(board.empty(), naive_board.empty());
        assert_eq!(board.mobility_bits(), naive_board.mobility_bits());
        assert_eq!(board.mobility(), naive_board.mobility());
        assert_eq!(board.score(), naive_board.score());
    }
}
