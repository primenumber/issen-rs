#[cfg(test)]
mod test;
use crate::serializer::*;
use clap::ArgMatches;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

const PACKED_SCALE: i32 = 256;

fn compress_word(data: i16) -> Vec<bool> {
    if data == 0 {
        vec![false, false]
    } else {
        let sign = data < 0;
        let data_abs = (data as i32).abs();
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

fn compress(data: &[i16]) -> Vec<bool> {
    let mut result_bits = Vec::new();
    for &word in data {
        result_bits.append(&mut compress_word(word));
    }
    result_bits
}

pub fn pack_weights(matches: &ArgMatches) {
    let input_path = matches.get_one::<String>("INPUT").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT").unwrap();

    let in_f = File::open(input_path).unwrap();
    let mut reader = BufReader::new(in_f);

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_weight = input_line.trim().parse().unwrap();
    let scale = PACKED_SCALE as f64;

    let mut weights = Vec::new();
    for _i in 0..num_weight {
        input_line.clear();
        reader.read_line(&mut input_line).unwrap();
        let weight = input_line.trim().parse::<f64>().unwrap();
        let weight_scaled = (weight * scale).round() as i16;
        weights.push(weight_scaled);
    }

    let orig_len = weights.len();
    let mut compressed = compress(&weights);

    writeln!(&mut writer, "{}", orig_len).unwrap();

    while compressed.len() % 15 != 0 {
        compressed.push(false);
    }
    for chunk in compressed.chunks(15) {
        let mut bits = 0u32;
        for (idx, &bit) in chunk.iter().enumerate() {
            if bit {
                bits |= 1 << idx;
            }
        }
        let c = encode_utf16(bits).unwrap();
        write!(&mut writer, "{}", c).unwrap();
    }
}
