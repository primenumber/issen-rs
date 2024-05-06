extern crate test;
use super::*;

#[test]
fn test_parse_record() {
    let record = Record::parse("f5d6c3d3 0").unwrap();
    assert_eq!(record.initial_board, Board::initial_state());
    let timeline = record.timeline().unwrap();
    assert_eq!(timeline.len(), 5);
    assert_eq!(timeline[0].1, Hand::from_str("f5").unwrap());
    assert_eq!(timeline[1].1, Hand::from_str("d6").unwrap());
    assert_eq!(timeline[2].1, Hand::from_str("c3").unwrap());
    assert_eq!(timeline[3].1, Hand::from_str("d3").unwrap());
    assert_eq!(timeline[4].1, Hand::Pass);
}

#[test]
fn test_parse_record_with_pass() {
    let record_with_pass = Record::parse("f5f6d3g5h5h4f7h6psc5 0").unwrap();
    let record_without_pass = Record::parse("f5f6d3g5h5h4f7h6c5 0").unwrap();
    assert_eq!(record_with_pass, record_without_pass);
}
