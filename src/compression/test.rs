extern crate test;
use super::*;

fn decompress_bits(bits: &[bool]) -> u16 {
    let mut data = 0;
    for &bit in bits {
        data <<= 1;
        if bit {
            data |= 1;
        }
    }
    data
}

fn decompress_word(data: &[bool]) -> (i32, usize) {
    if data[0] {
        let sign = data[3];
        let (bits, length) = if data[1] {
            if data[2] {
                let bits = decompress_bits(&data[4..19]) as i32;
                (bits + 249, 19)
            } else {
                let bits = decompress_bits(&data[4..11]) as i32;
                (bits + 121, 11)
            }
        } else if data[2] {
            let bits = decompress_bits(&data[4..10]) as i32;
            (bits + 57, 10)
        } else {
            let bits = decompress_bits(&data[4..9]) as i32;
            (bits + 25, 9)
        };
        let data = if sign { -bits } else { bits };
        (data, length)
    } else if data[1] {
        let sign = data[3];
        let (bits, length) = if data[2] {
            let bits = decompress_bits(&data[4..8]) as i32;
            (bits + 9, 8)
        } else {
            let bits = decompress_bits(&data[4..7]) as i32;
            (bits + 1, 7)
        };
        let data = if sign { -bits } else { bits };
        (data, length)
    } else {
        (0, 2)
    }
}

fn decompress(data_bits: &[bool], length: usize) -> Vec<i16> {
    let mut result = vec![0; length];
    let mut offset = 0;
    for r in result.iter_mut().take(length) {
        let (word, consume) = decompress_word(&data_bits[offset..]);
        *r = word as i16;
        offset += consume;
    }
    result
}

#[test]
fn test_compress_word() {
    for val in i16::MIN..=i16::MAX {
        let compressed = compress_word(val);
        let (decompressed, consume) = decompress_word(&compressed);
        assert_eq!(consume, compressed.len());
        assert_eq!(val, decompressed as i16);
    }
}
