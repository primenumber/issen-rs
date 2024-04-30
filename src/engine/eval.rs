#[cfg(test)]
mod test;
use crate::engine::board::*;

pub trait Evaluator: Send + Sync {
    fn eval(&self, board: Board) -> i16;
    fn score_scale(&self) -> i16;
    fn score_max(&self) -> i16;
    fn score_min(&self) -> i16;
}
