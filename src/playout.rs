use crate::bits::*;
use crate::board::*;
use crate::eval::*;
use crate::search::*;
use crate::think::*;
use crate::train::{create_record_by_solve, pos_to_str, step_by_pos};
use std::time::Instant;

pub async fn playout(
    record: &[usize],
    solve_obj: &mut SolveObj,
    think_time: u128,
    search_depth: usize,
) -> Option<(String, i8)> {
    let mut board = Board {
        player: 0x0000_0008_1000_0000,
        opponent: 0x0000_0010_0800_0000,
        is_black: true,
    };
    let mut updated_record = String::new();
    for &pos in record {
        if popcnt(board.empty()) as usize <= search_depth {
            let (s, b) = create_record_by_solve(board, solve_obj).await;
            board = b;
            updated_record += &s;
            break;
        } else {
            board = match step_by_pos(&board, pos) {
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
            let (s, b) = create_record_by_solve(board, solve_obj).await;
            board = b;
            updated_record += &s;
            break;
        } else {
            if board.mobility_bits() == 0 {
                board = board.pass();
            }
            let (_score, hand, _depth) =
                searcher.iterative_think(board, -64 * SCALE, 64 * SCALE, false);
            if let Hand::Play(pos) = hand {
                board = board.play(pos).unwrap();
                updated_record += &pos_to_str(pos);
            } else {
                panic!();
            }
        }
    }
    let score = if board.is_black {
        board.score()
    } else {
        -board.score()
    };
    Some((updated_record, score))
}
