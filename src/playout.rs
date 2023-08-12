use crate::engine::bits::*;
use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::hand::*;
use crate::engine::search::*;
use crate::engine::think::*;
use crate::train::{create_record_by_solve, pos_to_str, step_by_pos_with_color};
use std::sync::Arc;
use std::time::Instant;

pub async fn playout(
    record: &[usize],
    solve_obj: &mut SolveObj,
    sub_solver: &Arc<SubSolver>,
    think_time: u128,
    search_depth: usize,
) -> Option<(String, i8)> {
    let mut board = BoardWithColor::initial_state();
    let mut updated_record = String::new();
    for &pos in record {
        if popcnt(board.empty()) as usize <= search_depth {
            let (s, b) = create_record_by_solve(board, solve_obj, sub_solver).await;
            board = b;
            updated_record += &s;
            break;
        } else {
            board = match step_by_pos_with_color(&board, pos) {
                Some(next) => next,
                None => {
                    return None;
                }
            };
            updated_record += &pos_to_str(pos);
        }
    }
    let start = Instant::now();
    let timer = Timer {
        period: start,
        time_limit: think_time,
    };
    let mut searcher = Searcher {
        evaluator: solve_obj.evaluator.clone(),
        cache: solve_obj.eval_cache.clone(),
        timer: Some(timer),
        node_count: 0,
    };
    while !board.is_gameover() {
        if popcnt(board.empty()) as usize <= search_depth {
            let (s, b) = create_record_by_solve(board, solve_obj, sub_solver).await;
            board = b;
            updated_record += &s;
            break;
        } else {
            if board.board.mobility_bits() == 0 {
                board = board.pass_unchecked();
            }
            let (_score, hand, _depth) = searcher.iterative_think(board.board, EVAL_SCORE_MIN, EVAL_SCORE_MAX, false);
            if let Hand::Play(pos) = hand {
                board = board.play(pos).unwrap();
                updated_record += &pos_to_str(pos);
            } else {
                panic!();
            }
        }
    }
    let score = board.score();
    Some((updated_record, score))
}
