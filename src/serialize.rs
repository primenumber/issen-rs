use crate::engine::bits::*;
use crate::engine::board::*;
use clap::ArgMatches;
use std::fs::File;
use std::io::{BufWriter, Write};

pub fn encode_base64_impl(input: u8) -> Option<u8> {
    if input < 26 {
        // A-Z
        Some(input + 65)
    } else if input < 52 {
        // a-z
        Some(input + 71)
    } else if input < 62 {
        // 0-9
        Some(input - 4)
    } else if input == 62 {
        // +
        Some(43)
    } else if input == 63 {
        // /
        Some(47)
    } else {
        None
    }
}

#[derive(Debug)]
pub enum EncodeError {
    OutOfRange,
}

pub fn encode_utf16(mut input: u32) -> Result<char, EncodeError> {
    if input > 0xF7FF {
        return Err(EncodeError::OutOfRange);
    }
    input += 0x800;
    if input >= 0x202A {
        // skip 0x202A - 0x202E
        input += 5;
    }
    if input >= 0x2066 {
        // skip 0x2066 - 0x2069
        input += 4;
    }
    let u32data = input; // 3-byte UTF-8
    Ok(char::from_u32(u32data).unwrap())
}

fn encode_bits(bits: i16, length: usize) -> Vec<bool> {
    let mut result = Vec::new();
    for i in 0..length {
        let bit = (bits >> (length - 1 - i)) & 1;
        result.push(bit == 1);
    }
    result
}

fn compress_word(data: i16) -> Vec<bool> {
    if data == 0 {
        vec![false, false]
    } else {
        let sign = data < 0;
        let data_abs = data.abs();
        if data_abs <= 8 {
            let mut result = vec![false, true, false, sign];
            let bits = data_abs - 1;
            result.append(&mut encode_bits(bits, 3));
            result
        } else if data_abs <= 24 {
            let mut result = vec![false, true, true, sign];
            let bits = data_abs - 9;
            result.append(&mut encode_bits(bits, 4));
            result
        } else if data_abs <= 56 {
            let mut result = vec![true, false, false, sign];
            let bits = data_abs - 25;
            result.append(&mut encode_bits(bits, 5));
            result
        } else if data_abs <= 120 {
            let mut result = vec![true, false, true, sign];
            let bits = data_abs - 57;
            result.append(&mut encode_bits(bits, 6));
            result
        } else if data_abs <= 248 {
            let mut result = vec![true, true, false, sign];
            let bits = data_abs - 121;
            result.append(&mut encode_bits(bits, 7));
            result
        } else {
            let mut result = vec![true, true, true, sign];
            let bits = data_abs - 249;
            result.append(&mut encode_bits(bits, 15));
            result
        }
    }
}

pub fn compress(data: &[i16]) -> Vec<bool> {
    let mut result_bits = Vec::new();
    for &word in data {
        result_bits.append(&mut compress_word(word));
    }
    result_bits
}

pub fn gen_last_table(matches: &ArgMatches) {
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let mut v = Vec::new();
    for pos in 0..8 {
        for bits in 0..256 {
            let board = Board {
                player: bits,
                opponent: !(bits | (1 << pos)),
                is_black: true,
            };
            let fcnt = popcnt(board.flip(pos));
            v.push(fcnt);
        }
    }
    for chunk in v.chunks(2) {
        let val = chunk[0] | (chunk[1] << 3);
        write!(writer, "{}", encode_base64_impl(val as u8).unwrap() as char).unwrap();
    }
    writeln!(writer).unwrap();
}

pub fn gen_last_mask(matches: &ArgMatches) {
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    for pos in 0..(BOARD_SIZE as i8) {
        let mut masks: [u64; 3] = [0, 0, 0];
        let pr = pos / 8;
        let pc = pos % 8;
        for row in 0..8 {
            for col in 0..8 {
                let index = row * 8 + col;
                if col == pc {
                    masks[0] |= 1u64 << index;
                }
                if row + pc == pr + col {
                    masks[1] |= 1u64 << index;
                }
                if row + col == pr + pc {
                    masks[2] |= 1u64 << index;
                }
            }
        }
        for mask in masks {
            for row in 0..8 {
                let mask_in_the_row = (mask >> (8 * row)) & 0xff;
                let col = if mask_in_the_row == 0 {
                    8
                } else {
                    mask_in_the_row.trailing_zeros()
                };
                write!(writer, "{}", col).unwrap();
            }
        }
    }
    writeln!(writer).unwrap();
}
