#[cfg(test)]
mod test;
use crate::engine::board::*;
use crate::engine::hand::*;
use anyhow::Result;
use std::fmt::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
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

#[derive(Error, Debug)]
pub enum ParseRecordError {
    #[error("Failed to parse hand :{0}")]
    FailedToParseHand(String),
    #[error("invalid hand")]
    InvalidHand,
}

impl Record {
    pub fn new(initial_board: Board, hands: &[Hand], final_score: Option<i16>) -> Record {
        Record {
            initial_board,
            hands: hands.to_vec(),
            final_score,
        }
    }

    pub fn get_initial(&self) -> Board {
        self.initial_board
    }

    pub fn timeline(&self) -> Result<Vec<(Board, Hand, i16)>> {
        let mut board = self.initial_board;
        let mut res = Vec::new();
        let final_score = self.final_score.ok_or(ScoreIsNotRegistered {})?;
        let mut score = final_score;
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

impl FromStr for Record {
    type Err = ParseRecordError;

    fn from_str(record_str: &str) -> Result<Self, Self::Err> {
        let mut hands = Vec::new();
        let mut board = Board::initial_state();
        let splitted = record_str.split_ascii_whitespace().collect::<Vec<_>>();
        let l = splitted[0].len();
        for i in 0..(l / 2) {
            let hand_s = &splitted[0][(2 * i)..(2 * i + 2)];
            let h = hand_s
                .parse::<Hand>()
                .or(Err(ParseRecordError::FailedToParseHand(hand_s.to_string())))?;
            board = match board.play_hand(h) {
                Some(next) => next,
                None => {
                    let passed = board.pass().ok_or(ParseRecordError::InvalidHand)?;
                    match passed.play_hand(h) {
                        Some(next) => {
                            hands.push(Hand::Pass);
                            next
                        }
                        None => return Err(ParseRecordError::InvalidHand.into()),
                    }
                }
            };
            hands.push(h);
        }
        let score = if let Some(score) = splitted.get(1) {
            score.parse().ok()
        } else if board.is_gameover() {
            let absolute_score = if l % 2 == 0 {
                board.score()
            } else {
                -board.score()
            };
            Some(absolute_score as i16)
        } else {
            None
        };
        Ok(Record::new(Board::initial_state(), &hands, score))
    }
}

pub fn load_records(path: &Path) -> Result<impl Iterator<Item = Result<Record, ParseRecordError>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    Ok(reader.lines().map(|line| line.unwrap().parse::<Record>()))
}
