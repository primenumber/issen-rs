use super::*;
use crate::setup::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[test]
fn test_lookup_result() {
    const TEST_BASE81: &str = "!#jiR;rO[ORNM2MN";
    let board = Board::from_base81(TEST_BASE81).unwrap();
    let res_cache = ResCache {
        board,
        lower: -24,
        upper: 16,
        gen: 3,
        best: Some(Hand::Play(0)),
    };
    // [alpha, beta] is contained in [lower, upper]
    {
        let mut alpha = -12;
        let mut beta = 4;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, -12);
        assert_eq!(beta, 4);
    }
    // [alpha, beta] contains [lower, upper]
    {
        let mut alpha = -30;
        let mut beta = 20;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, res_cache.upper);
    }
    // alpha < lower < beta < upper
    {
        let mut alpha = -32;
        let mut beta = 8;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, 8);
    }
    // lower < alpha < upper < beta
    {
        let mut alpha = -6;
        let mut beta = 26;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(res_cache.lower, res_cache.upper, res_cache.best)
        );
        assert_eq!(alpha, -6);
        assert_eq!(beta, res_cache.upper);
    }
    // lower < upper < alpha < beta
    {
        let mut alpha = 22;
        let mut beta = 46;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.upper));
        assert_eq!(alpha, 22);
        assert_eq!(beta, res_cache.upper);
    }
    // alpha < beta < lower < upper
    {
        let mut alpha = -42;
        let mut beta = -24;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.lower));
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, -24);
    }
    // res_cache is None
    {
        let mut alpha = -6;
        let mut beta = 26;
        let result = make_lookup_result(None, &mut alpha, &mut beta);
        assert_eq!(
            result,
            CacheLookupResult::NoCut(-(BOARD_SIZE as i8), BOARD_SIZE as i8, None)
        );
        assert_eq!(alpha, -6);
        assert_eq!(beta, 26);
    }
    // lower < alpha = upper < beta
    {
        let mut alpha = 16;
        let mut beta = 26;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.upper));
        assert_eq!(alpha, 16);
        assert_eq!(beta, 16);
    }
    // alpha < beta = lower < upper
    {
        let mut alpha = -38;
        let mut beta = -24;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.lower));
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, -24);
    }
    // alpha < lower = upper < beta
    {
        let res_cache = ResCache {
            board,
            lower: 16,
            upper: 16,
            gen: 3,
            best: Some(Hand::Play(0)),
        };
        let mut alpha = -38;
        let mut beta = 30;
        let result = make_lookup_result(Some(res_cache.clone()), &mut alpha, &mut beta);
        assert_eq!(result, CacheLookupResult::Cut(res_cache.lower));
        assert_eq!(alpha, res_cache.lower);
        assert_eq!(beta, res_cache.lower);
    }
}

#[test]
fn test_solve_inner() {
    let name = "problem/stress_test_54_1k.b81r";
    let file = File::open(name).unwrap();
    let reader = BufReader::new(file);
    let solve_obj = setup_default();
    for (_idx, line) in reader.lines().enumerate() {
        let line_str = line.unwrap();
        let desired: i8 = line_str[17..].parse().unwrap();
        match Board::from_base81(&line_str[..16]) {
            Ok(board) => {
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
            Err(_) => {
                panic!();
            }
        }
    }
}
