use crate::bits::*;
use crate::board::*;
use crate::serialize::*;
use clap::ArgMatches;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

pub fn gen_book(matches: &ArgMatches) -> Option<()> {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();
    let max_count = matches.value_of("MAX_COUNT").unwrap().parse().unwrap();

    let in_f = File::open(input_path).ok()?;
    let mut reader = BufReader::new(in_f);

    let mut input_line = String::new();
    reader.read_line(&mut input_line).unwrap();
    let num_boards = input_line.trim().parse().unwrap();
    let mut records = Vec::new();
    for _i in 0..num_boards {
        input_line.clear();
        reader.read_line(&mut input_line).unwrap();
        let data: Vec<&str> = input_line.split(' ').collect();
        let player = u64::from_str_radix(&data[0], 16).ok()?;
        let opponent = u64::from_str_radix(&data[1], 16).ok()?;
        let board = Board {
            player,
            opponent,
            is_black: true, // dummy
        };
        if 64 - popcnt(board.empty()) > max_count {
            continue;
        }
        records.push((
            board,
            data[2].trim().parse::<i8>().unwrap(),
            data[3].trim().parse::<usize>().unwrap(),
        ));
    }

    records.sort_unstable_by_key(|k| popcnt(k.0.empty()));
    //let book = HashMap::new();

    for (board, _score, pos) in records {
        let _next = match board.play(pos) {
            Ok(n) => n,
            Err(_) => continue,
        };
    }

    let out_f = File::create(output_path).ok()?;
    let mut _writer = BufWriter::new(out_f);

    Some(())
}

pub fn pack_book(matches: &ArgMatches) {
    let input_path = matches.value_of("INPUT").unwrap();
    let output_path = matches.value_of("OUTPUT").unwrap();

    let in_f = File::open(input_path).unwrap();
    let reader = BufReader::new(in_f);

    let out_f = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_f);

    for line in reader.lines() {
        write!(writer, ">").unwrap();
        for pos_bytes in line.unwrap().trim().as_bytes().chunks(2) {
            let pos = if pos_bytes[0] < 0x60 {
                (pos_bytes[0] - 0x41) + (pos_bytes[1] - 0x31) * 8
            } else {
                (pos_bytes[0] - 0x61) + (pos_bytes[1] - 0x31) * 8
            };
            write!(writer, "{}", encode_base64_impl(pos).unwrap() as char).unwrap();
        }
    }
    write!(writer, "\n").unwrap();
}
