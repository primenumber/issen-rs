#[derive(Clone)]
pub struct HalfBoard {
    data: [bool; 64]
}

#[derive(Clone)]
pub struct Board {
    player: HalfBoard,
    opponent: HalfBoard,
    is_black: bool
}

use std::io::Write;

pub struct UnmovableError;

impl Board {
    fn is_movable_impl(&self, i: usize, j: usize, k: usize) -> bool {
        let di: [isize; 8] = [1, 1, 1, 0, -1, -1, -1, 0];
        let dj: [isize; 8] = [-1, 0, 1, 1, 1, 0, -1, -1];
        for l in 1..8 {
            let ni: isize = i as isize + l * di[k];
            let nj: isize = j as isize + l * dj[k];
            if ni < 0 || nj < 0 || ni >= 8 || nj >= 8 {
                return false;
            }
            let index: usize = (ni * 8 + nj) as usize;
            if self.player.data[index] {
                return l >= 2;
            }
            if !self.opponent.data[index] {
                return false;
            }
        }
        false
    }

    pub fn is_movable(&self, pos: usize) -> bool {
        if pos >= 64 {
            return false;
        }
        let i = pos / 8;
        let j = pos % 8;
        if self.player.data[pos] || self.opponent.data[pos] {
            return false;
        }
        for k in 0..8 {
            if self.is_movable_impl(i, j, k) {
                return true;
            }
        }
        false
    }

    fn play_impl(&self, result: &mut Board, i: usize, j: usize, k: usize) -> () {
        let di: [isize; 8] = [1, 1, 1, 0, -1, -1, -1, 0];
        let dj: [isize; 8] = [-1, 0, 1, 1, 1, 0, -1, -1];
        for l in 1..8 {
            let ni: isize = i as isize + l * di[k];
            let nj: isize = j as isize + l * dj[k];
            if ni < 0 || nj < 0 || ni >= 8 || nj >= 8 {
                return;
            }
            let index: usize = (ni * 8 + nj) as usize;
            if self.player.data[index] {
                for p in 1..l {
                    let oi: isize = i as isize + p * di[k];
                    let oj: isize = j as isize + p * dj[k];
                    let flip_index: usize = (oi * 8 + oj) as usize;
                    result.player.data[flip_index] = false;
                    result.opponent.data[flip_index] = true;
                }
                return;
            }
            if !self.opponent.data[index] {
                return;
            }
        }
    }

    pub fn play(&self, pos: usize) -> Result<Board, UnmovableError> {
        if !self.is_movable(pos) {
            return Err(UnmovableError{});
        }
        let mut result = Board {
            player: self.opponent.clone(),
            opponent: self.player.clone(),
            is_black: !self.is_black
        };
        let i = pos / 8;
        let j = pos % 8;
        for k in 0..8 {
            self.play_impl(&mut result, i, j, k);
        }
        result.opponent.data[pos] = true;
        Ok(result)
    }

    pub fn pass(&self) -> Board {
        Board {
            player: self.opponent.clone(),
            opponent: self.player.clone(),
            is_black: !self.is_black
        }
    }

    pub fn mobility(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0usize..64usize {
            if self.is_movable(i) {
                result.push(i);
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
            if self.player.data[i] {
                if self.is_black {
                    write!(std::io::stdout(), "X").unwrap();
                } else {
                    write!(std::io::stdout(), "O").unwrap();
                }
            } else if self.opponent.data[i] {
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
}

use std::str::FromStr;

pub struct BoardParseError;

impl FromStr for Board {
    type Err = BoardParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() <= 66 {
            return Err(BoardParseError{});
        }
        let mut black = HalfBoard { data: [false; 64] };
        let mut white = HalfBoard { data: [false; 64] };
        for (i, c) in s.chars().take(64).enumerate() {
            match c {
                'X' => black.data[i] = true,
                'O' => white.data[i] = true,
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

use std::io::prelude::*;

pub struct HandParseError;

fn read_hand() -> Option<usize> {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();
    if s.len() < 2 {
        return None;
    }
    if s == "ps" {
        return Some(64);
    }
    let column_code = s.chars().nth(0).unwrap() as usize;
    if column_code < 'a' as usize || ('h' as usize) < column_code {
        return None;
    }
    let row_code = s.chars().nth(1).unwrap() as usize;
    if row_code < '1' as usize || ('8' as usize) < row_code {
        return None;
    }
    Some((row_code - '1' as usize) * 8 + (column_code - 'a' as usize))
}

fn play(mut board: Board) -> Board {
    while !board.is_gameover() {
        board.print();
        println!("Input move");
        let hand: usize;
        loop {
            match read_hand() {
                Some(h) => {
                    hand = h;
                    break;
                },
                None => ()
            }
        }
        if hand == 64 {
            board = board.pass();
        } else {
            match board.play(hand) {
                Ok(next) => board = next,
                Err(_) => println!("Invalid move")
            }
        }
    }
    board
}

use std::io::BufReader;
use std::fs::File;

fn main() {
    let file = File::open("fforum-1-19.obf").unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines() {
        match Board::from_str(&line.unwrap()) {
            Ok(board) => play(board).print(),
            Err(_) => println!("Parse error")
        }
    }
}
