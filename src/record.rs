use crate::engine::board::*;
use std::str::FromStr;
use crate::engine::hand::*;
use anyhow::Result;
use std::fmt::*;

#[derive(Clone, Debug)]
pub struct Record {
    initial_board: Board,
    hands: Vec<Hand>,
    final_score: i16,
}

impl Record {
    pub fn new(initial_board: Board, hands: &[Hand], final_score: i16) -> Record {
        Record {
            initial_board,
            hands: hands.to_vec(),
            final_score,
        }
    }

    pub fn parse(record_str: &str) -> Result<Record> {
        let mut hands = Vec::new();
        let mut board = Board::initial_state();
        let l = record_str.len();
        for i in 0..(l / 2) {
            let h = Hand::from_str(&record_str[(2*i)..(2*i+2)])?;
            hands.push(h);
            board = board.play_hand(h)?;
        }
        Ok(Record::new(Board::initial_state(), &hands, board.score() as i16))
    }

    pub fn timeline(&self) -> Result<Vec<(Board, Hand, i16)>> {
        let mut board = self.initial_board;
        let mut res = Vec::new();
        let mut score = if self.hands.len() % 2 == 0 {
            self.final_score
        } else {
            -self.final_score
        };
        for &h in &self.hands {
            res.push((board, h, score));
            board = board.play_hand(h)?;
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
        Ok(())
    }
}
