use packed_simd::*;

pub fn upper_bit(mut x: u64x4) -> u64x4 {
    x = x | (x >>  1);
    x = x | (x >>  2);
    x = x | (x >>  4);
    x = x | (x >>  8);
    x = x | (x >> 16);
    x = x | (x >> 32);
    let lowers: u64x4 = x >> 1;
    x & !lowers
}

pub fn nonzero(x: u64x4) -> u64x4 {
    let zero = u64x4::new(0, 0, 0, 0);
    let mask = x.ne(zero);
    let one = u64x4::new(1, 1, 1, 1);
    one & u64x4::from_cast(mask)
}

pub fn popcnt(x: u64) -> i8 {
    x.count_ones() as i8
}

pub fn flip_vertical(mut x: u64) -> u64 {
    x = (x >> 32) | (x << 32);
    x = ((x >> 16) & 0x0000FFFF0000FFFFu64) | ((x << 16) & 0xFFFF0000FFFF0000u64);
    x = ((x >>  8) & 0x00FF00FF00FF00FFu64) | ((x <<  8) & 0xFF00FF00FF00FF00u64);
    x
}

pub fn delta_swap(x: u64, mask: u64, delta: isize) -> u64 {
    let tmp = mask & (x ^ (x << delta));
    x ^ tmp ^ (tmp >> delta)
}

pub fn flip_diag(mut x: u64) -> u64 {
    x = delta_swap(x, 0x0f0f0f0f00000000u64, 28);
    x = delta_swap(x, 0x3333000033330000u64, 14);
    x = delta_swap(x, 0x5500550055005500u64, 7);
    x
}

pub fn rot90(x: u64) -> u64 {
    flip_vertical(flip_diag(x))
}

pub fn mirror_under_8(mut x: u64) -> u64 {
    x = (x >> 4) | ((x << 4) & 0xF0);
    x = ((x >> 2) & 0x33) | ((x << 2) & 0xCC);
    x = ((x >> 1) & 0x55) | ((x << 1) & 0xAA);
    x
}

use bitintr::{Pext, Pdep};

pub fn pext(x: u64, mask: u64) -> u64 {
    x.pext(mask)
}

pub fn pdep(x: u64, mask: u64) -> u64 {
    x.pdep(mask)
}

lazy_static! {
    pub static ref BASE3: [usize; 256] = {
        let mut res = [0usize; 256];
        for x in 0..256 {
            let mut pow3 = 1;
            let mut sum = 0;
            for i in 0..8 {
                if ((x >> i) & 1) == 1 {
                    sum += pow3;
                }
                pow3 *= 3;
            }
            res[x] = sum;
        }
        res
    };
}
