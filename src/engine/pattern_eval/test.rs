extern crate test;
use super::*;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
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

#[bench]
fn bench_eval(b: &mut Bencher) {
    let evaluator = Arc::new(PatternLinearEvaluator::load(Path::new("table-220710")).unwrap());
    let dataset = load_stress_test_set();

    b.iter(|| {
        dataset
            .iter()
            .map(|(board, _)| evaluator.eval(*board) as i32)
            .sum::<i32>()
    });
}
