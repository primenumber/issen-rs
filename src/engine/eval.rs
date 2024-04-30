#[cfg(test)]
mod test;
use crate::engine::board::*;

pub fn pow3(x: i8) -> usize {
    if x == 0 {
        1
    } else {
        3 * pow3(x - 1)
    }
}

// interprete base-2 number as base-3 number
// base_2_to_3(x) := radix_parse(radix_fmt(x, 2), 3)
const fn base_2_to_3(mut x: usize) -> usize {
    let mut base3 = 0;
    let mut pow3 = 1;
    while x > 0 {
        base3 += (x % 2) * pow3;
        pow3 *= 3;
        x /= 2;
    }
    base3
}

const BASE_2_TO_3_TABLE_BITS: usize = 13;
pub const BASE_2_TO_3: [usize; 1 << BASE_2_TO_3_TABLE_BITS] = {
    let mut table = [0usize; 1 << BASE_2_TO_3_TABLE_BITS];
    let mut i = 0;
    while i < table.len() {
        table[i] = base_2_to_3(i);
        i += 1;
    }
    table
};

pub trait Evaluator: Send + Sync {
    fn eval(&self, board: Board) -> i16;
    fn score_scale(&self) -> i16;
    fn score_max(&self) -> i16;
    fn score_min(&self) -> i16;
}
