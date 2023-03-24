use crate::engine::bits::*;
use crate::engine::board::*;
use bitintr::Tzcnt;
use std::cmp::min;

pub struct LastFlipCache([i8; 2048], [u64; 15], [u64; 15]);

impl LastFlipCache {
    pub fn new() -> LastFlipCache {
        let mut table = [0; 2048];
        for bits in 0..256 {
            for pos in 0..8 {
                if (bits >> pos) & 1 != 0 {
                    continue;
                }
                let b = Board {
                    player: bits,
                    opponent: !bits ^ (1 << pos),
                    is_black: true,
                };
                let index = (bits * 8 + pos) as usize;
                table[index] = popcnt(b.flip(pos as usize));
            }
        }
        let mut a1h8 = [0; 15];
        let mut a8h1 = [0; 15];
        for i in 0..15 {
            if i < 8 {
                a1h8[i] = 0x0102040810204080 >> (8 * (7 - i));
                a8h1[i] = 0x8040201008040201 >> (8 * (7 - i));
            } else {
                a1h8[i] = 0x0102040810204080 << (8 * (i - 7));
                a8h1[i] = 0x8040201008040201 << (8 * (i - 7));
            }
        }
        LastFlipCache(table, a1h8, a8h1)
    }

    pub fn flip_count(&self, board: Board) -> i8 {
        let empty = board.empty();
        let pos = empty.tzcnt() as usize;
        let row = pos / 8;
        let col = pos % 8;
        let diag_a1h8 = row + col;
        let diag_a8h1 = row + 7 - col;
        let mut ans = 0;
        ans += self.0[8 * ((board.player as usize >> (row * 8)) & 0xff) + col];
        ans += self.0[8 * (pext(board.player, 0x0101_0101_0101_0101 << col) as usize) + row];
        ans += self.0[8 * (pext(board.player, self.1[diag_a1h8]) as usize) + min(row, 7 - col)];
        ans += self.0[8 * (pext(board.player, self.2[diag_a8h1]) as usize) + min(row, col)];
        ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test() {
        let cache = LastFlipCache::new();
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0xDEADBEAF);
        const LENGTH: usize = 256;
        for _ in 0..LENGTH {
            let mut player = rng.gen::<u64>();
            let opponent = !player;
            let mut pcnt = popcnt(player);
            let empty_bit = pdep(1 << rng.gen_range(0..pcnt), player);
            let pos = empty_bit.tzcnt();
            player ^= empty_bit;
            pcnt -= 1;
            let ocnt = popcnt(opponent);
            let board = Board {
                player,
                opponent,
                is_black: true,
            };
            assert_eq!(popcnt(board.flip(pos as usize)), cache.flip_count(board));
        }
    }
}
