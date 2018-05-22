pub struct HalfBoard {
    data: [bool; 64]
}

pub struct Board {
    player: HalfBoard,
    opponent: HalfBoard,
    is_black: bool
}

use std::io::Write;

impl Board {
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
        }
        write!(std::io::stdout(), "\n").unwrap();
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

use std::io::BufReader;
use std::io::prelude::*;
use std::fs::File;

fn main() {
    let file = File::open("fforum-1-19.obf").unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines() {
        match Board::from_str(&line.unwrap()) {
            Ok(board) => board.print(),
            Err(_) => println!("Parse error")
        }
    }
}
