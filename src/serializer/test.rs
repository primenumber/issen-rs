extern crate test;
use super::*;

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
