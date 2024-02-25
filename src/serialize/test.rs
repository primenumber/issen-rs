extern crate test;
use super::*;

const LAST_TABLE: &str = "AABACABADABACABAEABACABADABACABAFABACABADABACABAEABACABADABACABAGABACABADABACABAEABACABADABACABAFABACABADABACABAEABACABADABACABAAAAAJAAASAAAJAAAbAAAJAAASAAAJAAAkAAAJAAASAAAJAAAbAAAJAAASAAAJAAAtAAAJAAASAAAJAAAbAAAJAAASAAAJAAAkAAAJAAASAAAJAAAbAAAJAAASAAAJAAAIAAAIAAARJAAIAAAaSAAIAAARJAAIAAAjbAAIAAARJAAIAAAaSAAIAAARJAAIAAAskAAIAAARJAAIAAAaSAAIAAARJAAIAAAjbAAIAAARJAAIAAAaSAAIAAARJAAIAAAQJAAAAAAQJAAAAAAZSJJAAAAQJAAAAAAibSSAAAAQJAAAAAAZSJJAAAAQJAAAAAArkbbAAAAQJAAAAAAZSJJAAAAQJAAAAAAibSSAAAAQJAAAAAAZSJJAAAAQJAAAAAAYSJJAAAAAAAAAAAAYSJJAAAAAAAAAAAAhbSSJJJJAAAAAAAAYSJJAAAAAAAAAAAAqkbbSSSSAAAAAAAAYSJJAAAAAAAAAAAAhbSSJJJJAAAAAAAAYSJJAAAAAAAAAAAAgbSSJJJJAAAAAAAAAAAAAAAAAAAAAAAAgbSSJJJJAAAAAAAAAAAAAAAAAAAAAAAApkbbSSSSJJJJJJJJAAAAAAAAAAAAAAAAgbSSJJJJAAAAAAAAAAAAAAAAAAAAAAAAokbbSSSSJJJJJJJJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAokbbSSSSJJJJJJJJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwtkkbbbbSSSSSSSSJJJJJJJJJJJJJJJJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

// Base64

pub fn decode_base64_impl(input: u8) -> Option<u8> {
    if input == 43 {
        // +
        Some(62)
    } else if input == 47 {
        // /
        Some(63)
    } else if (48..=57).contains(&input) {
        // 0-9
        Some(input + 4)
    } else if (65..=90).contains(&input) {
        // A-Z
        Some(input - 65)
    } else if (97..=122).contains(&input) {
        // a-z
        Some(input - 71)
    } else {
        None
    }
}

pub fn decode_last_table(index: usize) -> i8 {
    let bit = decode_base64_impl(*LAST_TABLE.as_bytes().get(index / 2).unwrap()).unwrap() as i8;
    (bit >> ((index % 2) * 3)) & 0b000111
}

// Decode UTF16-encoded data

pub fn decode_base_utf16(input: &[char]) -> Vec<bool> {
    let mut output = Vec::with_capacity(input.len() * 15);
    for &c in input {
        let bits = c as u32;
        let bits = if bits > 0x2069 { bits - 4 } else { bits };
        let bits = if bits > 0x202e { bits - 7 } else { bits };
        let bits = bits - 0x800;
        for idx in 0..15 {
            output.push((bits >> idx) & 1 == 1);
        }
    }
    output
}

// decompress
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

pub fn decompress(data_bits: &[bool], length: usize) -> Vec<i16> {
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
fn test_encode_utf16() {
    const ENCODE_MAX: u32 = (1 << 15) - 1;
    for val in 0..=ENCODE_MAX {
        let encoded = encode_utf16(val).expect("Failed to encode");
        let decoded = decode_base_utf16(&[encoded]);
        let mut bits = 0u32;
        for (idx, &bit) in decoded.iter().enumerate() {
            if bit {
                bits |= 1 << idx;
            }
        }
        assert_eq!(val, bits);
    }
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
