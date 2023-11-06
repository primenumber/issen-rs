use super::*;

use rand::{Rng, SeedableRng};

fn solve_last_naive(board: Board) -> (i8, usize) {
    let pos = board.empty().tzcnt() as usize;
    match board.play(pos) {
        Ok(next) => (-next.score(), 1),
        Err(_) => match board.pass_unchecked().play(pos) {
            Ok(next) => (next.score(), 2),
            Err(_) => (board.score(), 0),
        },
    }
}

#[test]
fn test_last_cache() {
    // gen data
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0xDEADBEAF);
    const LENGTH: usize = 256;
    let last_cache = LastCache::new();
    // last_cache
    for _ in 0..LENGTH {
        let bit = rng.gen::<u64>();
        let pos = rng.gen_range(0..BOARD_SIZE);
        let pos_mask = !(1 << pos);
        let player = bit & pos_mask;
        let opponent = !bit & pos_mask;
        let board = Board { player, opponent };
        assert_eq!(last_cache.solve_last(board), solve_last_naive(board));
    }
}
