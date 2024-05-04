#[cfg(test)]
mod test;
use crate::engine::bits::*;
use crate::engine::board::*;

pub struct LastCache {
    table: [(i8, i8); 4096],
    masks: [(u64, u64, u64); BOARD_SIZE],
    indices: [(u8, u8); BOARD_SIZE],
}

impl LastCache {
    fn pre_compute(bits: u64, pos: usize) -> (i8, i8) {
        let opp_mask = 0xff ^ (1 << pos);
        let opp = !bits & opp_mask;
        let board_first = Board {
            player: bits,
            opponent: opp,
        };
        let board_second = Board {
            player: opp,
            opponent: bits,
        };
        (
            popcnt(board_first.flip_unchecked(pos)),
            popcnt(board_second.flip_unchecked(pos)),
        )
    }

    pub fn new() -> LastCache {
        let mut table = [(0, 0); 4096];
        for bits in 0..256 {
            for pos in 0..8 {
                let idx = bits as usize * 8 + pos;
                table[idx] = Self::pre_compute(bits, pos);
            }
        }
        let mut masks = [(0, 0, 0); BOARD_SIZE];
        let mut indices = [(0, 0); BOARD_SIZE];
        for pos in 0..BOARD_SIZE {
            let row = pos / 8;
            let col = pos % 8;
            let col_mask = 0x0101010101010101 << col;
            let diag1_mask = if row > col {
                0x8040201008040201 << ((row - col) * 8)
            } else {
                0x8040201008040201 >> ((col - row) * 8)
            };
            let diag1_idx = if row > col { col } else { row };
            let diag2_mask = if row + col > 7 {
                0x0102040810204080 << ((row + col - 7) * 8)
            } else {
                0x0102040810204080 >> ((7 - row - col) * 8)
            };
            let diag2_idx = if row + col > 7 { 7 - col } else { row };
            masks[pos] = (col_mask, diag1_mask, diag2_mask);
            indices[pos] = (diag1_idx as u8, diag2_idx as u8);
        }
        LastCache {
            table,
            masks,
            indices,
        }
    }

    #[cfg(all(target_feature = "bmi2", not(slow_pext)))]
    fn get_col_bits(bits: u64, mask: u64, _col: usize) -> u64 {
        bits.pext(mask)
    }

    #[cfg(not(all(target_feature = "bmi2", not(slow_pext))))]
    fn get_col_bits(mut bits: u64, mask: u64, col: usize) -> u64 {
        bits &= mask;
        ((bits >> col).wrapping_mul(0x0002_0408_1020_4081) >> 49) & 0xff
    }

    #[cfg(all(target_feature = "bmi2", not(slow_pext)))]
    fn get_diag1_bits(bits: u64, mask: u64, _row: usize, _col: usize) -> u64 {
        bits.pext(mask)
    }

    #[cfg(not(all(target_feature = "bmi2", not(slow_pext))))]
    fn get_diag1_bits(mut bits: u64, mask: u64, row: usize, col: usize) -> u64 {
        bits &= mask;
        let width = if row >= col {
            bits = bits >> (8 * (row - col));
            8 - (row - col)
        } else {
            bits = bits >> (col - row);
            8 - (col - row)
        };
        (bits.wrapping_mul(0x0101_0101_0101_0101) >> 56) & ((1 << width) - 1)
    }

    #[cfg(all(target_feature = "bmi2", not(slow_pext)))]
    fn get_diag2_bits(bits: u64, mask: u64, _row: usize, _col: usize) -> u64 {
        bits.pext(mask)
    }

    #[cfg(not(all(target_feature = "bmi2", not(slow_pext))))]
    fn get_diag2_bits(mut bits: u64, mask: u64, row: usize, col: usize) -> u64 {
        bits &= mask;
        let width = if row + col >= 7 {
            bits = bits >> (8 * (row + col - 7));
            15 - (row + col)
        } else {
            bits = bits << (7 - row - col);
            row + col + 1
        };
        bits |= (bits >> 56) << 63;
        bits = bits.wrapping_mul(0x7f) & 0x8080_8080_8080_8080;
        (bits.wrapping_mul(0x0002_0408_1020_4081) >> 56) & ((1 << width) - 1)
    }

    fn solve_last_impl(&self, board: Board) -> (i8, usize) {
        unsafe {
            let pos = board.empty().trailing_zeros() as usize;
            let row = pos >> 3;
            let col = pos & 0b111;
            let row_bits = (board.player >> (row * 8)) & 0xff;
            let &(col_mask, diag1_mask, diag2_mask) = self.masks.get_unchecked(pos);
            let &(diag1_idx, diag2_idx) = self.indices.get_unchecked(pos);
            let &row_score = self.table.get_unchecked((row_bits as usize) * 8 + col);
            let col_bits = Self::get_col_bits(board.player, col_mask, col);
            let &col_score = self.table.get_unchecked((col_bits as usize) * 8 + row);
            let diag1_bits = Self::get_diag1_bits(board.player, diag1_mask, row, col);
            let &diag1_score = self
                .table
                .get_unchecked((diag1_bits as usize) * 8 + diag1_idx as usize);
            let diag2_bits = Self::get_diag2_bits(board.player, diag2_mask, row, col);
            let &diag2_score = self
                .table
                .get_unchecked((diag2_bits as usize) * 8 + diag2_idx as usize);
            let pcnt = popcnt(board.player);
            let ocnt = 63 - pcnt;
            let diff_first = row_score.0 + col_score.0 + diag1_score.0 + diag2_score.0;
            if diff_first > 0 {
                (pcnt - ocnt + 2 * diff_first + 1, 1)
            } else {
                let diag1_bits_second = Self::get_diag1_bits(board.opponent, diag1_mask, row, col);
                let &diag1_score_second = self
                    .table
                    .get_unchecked((diag1_bits_second as usize) * 8 + diag1_idx as usize);
                let diag2_bits_second = Self::get_diag2_bits(board.opponent, diag2_mask, row, col);
                let &diag2_score_second = self
                    .table
                    .get_unchecked((diag2_bits_second as usize) * 8 + diag2_idx as usize);
                let diff_second = row_score.1 + col_score.1 + diag1_score_second.0 + diag2_score_second.0;
                if diff_second > 0 {
                    (pcnt - ocnt - 2 * diff_second - 1, 2)
                } else if pcnt > ocnt {
                    (64 - 2 * ocnt, 0)
                } else {
                    (2 * pcnt - 64, 0)
                }
            }
        }
    }

    pub fn solve_last(&self, board: Board) -> (i8, usize) {
        self.solve_last_impl(board)
    }
}
