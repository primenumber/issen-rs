use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Empty,
    Player,
    Opponent,
}

#[derive(Clone)]
struct NaiveBoard {
    data: [State; BOARD_SIZE],
    is_black: bool,
}

impl From<Board> for NaiveBoard {
    fn from(board: Board) -> Self {
        let mut res = NaiveBoard {
            data: [State::Empty; BOARD_SIZE],
            is_black: board.is_black,
        };
        for i in 0..BOARD_SIZE {
            if ((board.player >> i) & 1) == 1 {
                res.data[i] = State::Player;
            } else if ((board.opponent >> i) & 1) == 1 {
                res.data[i] = State::Opponent;
            }
        }
        res
    }
}

impl From<NaiveBoard> for Board {
    fn from(naive_board: NaiveBoard) -> Self {
        let mut player = 0;
        let mut opponent = 0;
        for i in 0..BOARD_SIZE {
            match naive_board.data[i] {
                State::Player => {
                    player |= 1 << i;
                }
                State::Opponent => {
                    opponent |= 1 << i;
                }
                _ => (),
            }
        }
        Board {
            player,
            opponent,
            is_black: naive_board.is_black,
        }
    }
}

impl NaiveBoard {
    fn flip(&self, pos: usize) -> u64 {
        const DELTA: [(isize, isize); 8] = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ];
        if self.data[pos] != State::Empty {
            return 0;
        }
        let r = (pos / 8) as isize;
        let c = (pos % 8) as isize;
        let mut res = 0;
        for (dr, dc) in DELTA.iter() {
            for i in 1..8 {
                let nr = r + i * dr;
                let nc = c + i * dc;
                if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 {
                    break;
                }
                let ni = (nr * 8 + nc) as usize;
                match self.data[ni] {
                    State::Player => {
                        for j in 1..i {
                            let mr = r + j * dr;
                            let mc = c + j * dc;
                            let mi = mr * 8 + mc;
                            res |= 1 << mi;
                        }
                        break;
                    }
                    State::Empty => {
                        break;
                    }
                    State::Opponent => (),
                }
            }
        }
        res
    }

    fn is_movable(&self, pos: usize) -> bool {
        if pos >= BOARD_SIZE {
            return false;
        }
        if self.data[pos] != State::Empty {
            return false;
        }
        self.flip(pos) != 0
    }

    fn play(&self, pos: usize) -> Result<NaiveBoard, UnmovableError> {
        if pos >= BOARD_SIZE {
            return Err(UnmovableError {});
        }
        if self.data[pos] != State::Empty {
            return Err(UnmovableError {});
        }
        let flip_bits = self.flip(pos);
        if flip_bits == 0 {
            return Err(UnmovableError {});
        }
        let mut res = NaiveBoard {
            data: [State::Empty; BOARD_SIZE],
            is_black: !self.is_black,
        };
        for i in 0..BOARD_SIZE {
            if ((flip_bits >> i) & 1) == 1 {
                res.data[i] = State::Opponent;
            } else if self.data[i] == State::Player {
                res.data[i] = State::Opponent;
            } else if self.data[i] == State::Opponent {
                res.data[i] = State::Player;
            } else if i == pos {
                res.data[i] = State::Opponent;
            }
        }
        Ok(res)
    }

    fn empty(&self) -> u64 {
        let mut res = 0;
        for i in 0..BOARD_SIZE {
            if self.data[i] == State::Empty {
                res |= 1 << i;
            }
        }
        res
    }

    fn mobility_bits(&self) -> u64 {
        let mut res = 0;
        for i in 0..BOARD_SIZE {
            if self.is_movable(i) {
                res |= 1 << i;
            }
        }
        res
    }

    fn mobility(&self) -> Vec<usize> {
        let mut res = Vec::new();
        for i in 0..BOARD_SIZE {
            if self.is_movable(i) {
                res.push(i);
            }
        }
        res
    }

    fn score(&self) -> i8 {
        let mut pcnt = 0;
        let mut ocnt = 0;
        for i in 0..BOARD_SIZE {
            match self.data[i] {
                State::Player => {
                    pcnt += 1;
                }
                State::Opponent => {
                    ocnt += 1;
                }
                _ => (),
            }
        }
        if pcnt == ocnt {
            0
        } else if pcnt > ocnt {
            BOARD_SIZE as i8 - 2 * ocnt
        } else {
            -(BOARD_SIZE as i8) + 2 * pcnt
        }
    }
}

#[test]
fn test_ops() {
    const TEST_BASE81: &str = "!#jiR;rO[ORNM2MN";
    let board = Board::from_base81(TEST_BASE81).unwrap();
    let naive_board = NaiveBoard::from(board);
    assert_eq!(board, Board::from(naive_board.clone()));
    for i in 0..BOARD_SIZE {
        assert_eq!(board.flip(i), naive_board.flip(i));
        assert_eq!(board.is_movable(i), naive_board.is_movable(i));
        if board.is_movable(i) {
            assert_eq!(
                board.play(i).unwrap(),
                Board::from(naive_board.play(i).unwrap())
            );
        }
    }
    assert_eq!(board.empty(), naive_board.empty());
    assert_eq!(board.mobility_bits(), naive_board.mobility_bits());
    assert_eq!(board.mobility(), naive_board.mobility());
    assert_eq!(board.score(), naive_board.score());
}

#[test]
fn test_base81() {
    const TEST_BASE81: &str = "!#jiR;rO[ORNM2MN";
    let board = Board::from_base81(TEST_BASE81).unwrap();
    let encoded = board.to_base81();
    assert_eq!(TEST_BASE81, encoded)
}
