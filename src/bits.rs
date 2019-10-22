use packed_simd::*;
use lazy_static::lazy_static;
use bitintr::{Pext, Pdep};

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn upper_bit_naive(x: u64x4) -> u64x4 {
        let mut res = u64x4::splat(0);
        for i in 0..4 {
            let y = x.extract(i);
            let mut ans = 0;
            for j in 0..64 {
                let bit = 1 << j;
                if (y & bit) != 0 {
                    ans = bit;
                }
            }
            res = res.replace(i, ans);
        }
        res
    }

    fn nonzero_naive(x: u64x4) -> u64x4 {
        let mut res = u64x4::splat(0);
        for i in 0..4 {
            let y = x.extract(i);
            let ans = if y != 0 {
                1
            } else {
                0
            };
            res = res.replace(i, ans);
        }
        res
    }

    fn flip_vertical_naive(x: u64) -> u64 {
        let mut res = 0;
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let flip_i = (7 - r) * 8 + c;
                if ((x >> i) & 1) == 1 {
                    res |= 1 << flip_i;
                }
            }
        }
        res
    }

    fn delta_swap_naive(x: u64, mask: u64, delta: isize) -> u64 {
        let mut res = 0;
        for i in 0..64 {
            if i >= delta && ((mask >> i) & 1) == 1 {
                res |= ((x >> (i - delta)) & 1) << i;
            } else if i + delta < 64 && ((mask >> (i + delta)) & 1) == 1 {
                res |= ((x >> (i + delta)) & 1) << i;
            } else {
                res |= x & (1 << i);
            }
        }
        res
    }

    fn flip_diag_naive(x: u64) -> u64 {
        let mut res = 0;
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let flip_i = c * 8 + r;
                res |= ((x >> i) & 1) << flip_i;
            }
        }
        res
    }

    fn rot90_naive(x: u64) -> u64 {
        let mut res = 0;
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let rot_i = (7 - c) * 8 + r;
                res |= ((x >> i) & 1) << rot_i;
            }
        }
        res
    }

    fn mirror_under_8_naive(x: u64) -> u64 {
        let mut res = 0;
        for i in 0..8 {
            res |= ((x >> i) & 1) << (7 - i);
        }
        res
    }

    #[test]
    fn test_ops() {
        // gen data
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0xDEADBEAF);
        const LENGTH: usize = 256;
        let mut ary = [0u64; LENGTH];
        for i in 0..LENGTH {
            ary[i] = rng.gen::<u64>();
        }
        // upper_bit
        for i in 0..=(LENGTH-4) {
            let a = u64x4::from_slice_unaligned(&ary[i..(i+4)]);
            assert_eq!(upper_bit(a), upper_bit_naive(a));
        }
        // nonzero
        for i in 0..=(LENGTH-4) {
            let a = u64x4::from_slice_unaligned(&ary[i..(i+4)]);
            assert_eq!(nonzero(a), nonzero_naive(a));
        }
        // flip_vertical
        for a in ary.iter() {
            assert_eq!(flip_vertical(*a), flip_vertical_naive(*a));
        }
        const DELTA_SWAP_MASKS: [(u64, isize); 8] = [
            (0xAAAA_AAAA_AAAA_AAAA, 1),
            (0xCCCC_CCCC_CCCC_CCCC, 2),
            (0xF0F0_F0F0_F0F0_F0F0, 4),
            (0xFF00_FF00_FF00_FF00, 8),
            (0xFFFF_0000_FFFF_0000, 16),
            (0xFFFF_FFFF_0000_0000, 32),
            (0x3333_0000_3333_0000, 16),
            (0x00E3_8E38_00E3_8000, 9),
        ];
        // delta_swap
        for a in ary.iter() {
            for (mask, delta) in DELTA_SWAP_MASKS.iter() {
                assert_eq!(
                    delta_swap(*a, *mask, *delta),
                    delta_swap_naive(*a, *mask, *delta));
            }
        }
        // flip_diag
        for a in ary.iter() {
            assert_eq!(flip_diag(*a), flip_diag_naive(*a));
        }
        // rot90
        for a in ary.iter() {
            assert_eq!(rot90(*a), rot90_naive(*a));
        }
        // mirror_under_8
        for a in ary.iter() {
            let masked = *a & 0xFF;
            assert_eq!(mirror_under_8(masked), mirror_under_8_naive(masked));
        }
    }
}
