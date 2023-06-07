extern crate test;
use super::*;
use crate::setup::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use test::Bencher;

fn load_stress_test_set() -> Vec<(Board, i8)> {
    let name = "problem/stress_test_54_1k.b81r";
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();
    for (_idx, line) in reader.lines().enumerate() {
        let line_str = line.unwrap();
        let desired: i8 = line_str[17..].parse().unwrap();
        match Board::from_base81(&line_str[..16]) {
            Ok(board) => {
                dataset.push((board, desired));
            }
            Err(_) => {
                panic!();
            }
        }
    }
    dataset
}

#[test]
fn test_solve_inner() {
    let solve_obj = setup_default();
    let dataset = load_stress_test_set();

    for &(board, desired) in dataset.iter() {
        let mut obj = solve_obj.clone();
        let (res, _stat) = solve_inner(
            &mut obj,
            board,
            -(BOARD_SIZE as i8),
            BOARD_SIZE as i8,
            false,
        );
        if res != desired {
            board.print();
        }
        assert_eq!(res, desired);
    }
}

#[bench]
fn bench_solve_inner(b: &mut Bencher) {
    let solve_obj = setup_default();
    let dataset = load_stress_test_set();

    b.iter(|| {
        for &(board, _desired) in dataset.iter() {
            let mut obj = solve_obj.clone();
            let (_res, _stat) = solve_inner(
                &mut obj,
                board,
                -(BOARD_SIZE as i8),
                BOARD_SIZE as i8,
                false,
            );
        }
    });
}
