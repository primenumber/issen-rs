#[cfg(test)]
mod test;
use crate::engine::board::*;
use crate::engine::hand::*;
use anyhow::Result;
use std::fmt::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Record {
    initial_board: Board,
    hands: Vec<Hand>,
    final_score: Option<i16>,
}

#[derive(Error, Debug)]
#[error("Score is not registered")]
pub struct ScoreIsNotRegistered {}

impl Record {
    pub fn new(initial_board: Board, hands: &[Hand], final_score: Option<i16>) -> Record {
        Record {
            initial_board,
            hands: hands.to_vec(),
            final_score,
        }
    }

    pub fn parse(record_str: &str) -> Result<Record> {
        let mut hands = Vec::new();
        let mut board = Board::initial_state();
        let splitted = record_str.split_ascii_whitespace().collect::<Vec<_>>();
        let l = splitted[0].len();
        for i in 0..(l / 2) {
            let h = splitted[0][(2 * i)..(2 * i + 2)].parse::<Hand>()?;
            board = match board.play_hand(h) {
                Some(next) => next,
                None => {
                    let passed = board.pass().ok_or(UnmovableError {})?;
                    match passed.play_hand(h) {
                        Some(next) => {
                            hands.push(Hand::Pass);
                            next
                        }
                        None => return Err(UnmovableError {}.into()),
                    }
                }
            };
            hands.push(h);
        }
        let score = if let Some(score) = splitted.get(1) {
            score.parse().ok()
        } else if board.is_gameover() {
            Some(board.score() as i16)
        } else {
            None
        };
        Ok(Record::new(Board::initial_state(), &hands, score))
    }

    pub fn get_initial(&self) -> Board {
        self.initial_board
    }

    pub fn timeline(&self) -> Result<Vec<(Board, Hand, i16)>> {
        let mut board = self.initial_board;
        let mut res = Vec::new();
        let final_score = self.final_score.ok_or(ScoreIsNotRegistered {})?;
        let mut score = if self.hands.len() % 2 == 0 {
            final_score
        } else {
            -final_score
        };
        for &h in &self.hands {
            res.push((board, h, score));
            board = board.play_hand(h).ok_or(UnmovableError {})?;
            score = -score;
        }
        res.push((board, Hand::Pass, score));
        Ok(res)
    }
}

impl Display for Record {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for h in &self.hands {
            write!(f, "{}", h)?;
        }
        if let Some(final_score) = self.final_score {
            write!(f, " {}", final_score)?;
        }
        Ok(())
    }
}

pub fn load_records(path: &Path) -> Result<Vec<Record>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut input_line = String::new();

    reader.read_line(&mut input_line)?;
    let num_records = input_line.trim().parse()?;

    let mut records = Vec::new();
    for _i in 0..num_records {
        let mut input_line = String::new();
        reader.read_line(&mut input_line)?;
        records.push(Record::parse(&input_line)?);
    }
    Ok(records)
}
