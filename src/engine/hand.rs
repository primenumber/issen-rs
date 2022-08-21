use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hand {
    Play(usize),
    Pass,
}

impl Hand {
    fn rot90(&self) -> Hand {
        match self {
            Self::Pass => *self,
            Self::Play(pos) => {
                let row = pos / 8;
                let col = pos % 8;
                Self::Play((7 - col) * 8 + row)
            }
        }
    }

    fn flip_diag(&self) -> Hand {
        match self {
            Self::Pass => *self,
            Self::Play(pos) => {
                let row = pos / 8;
                let col = pos % 8;
                Self::Play(col * 8 + row)
            }
        }
    }

    pub fn transform(&self, rotate: usize, mirror: bool) -> Hand {
        let mut tmp = *self;
        for _ in 0..rotate {
            tmp = tmp.rot90();
        }
        if mirror {
            tmp = tmp.flip_diag();
        }
        tmp
    }
}

impl fmt::Display for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Play(pos) => {
                let row = *pos as u8 / 8;
                let col = *pos as u8 % 8;
                let row_char = b'1' + row;
                let col_char = b'A' + col;
                write!(f, "{}{}", col_char as char, row_char as char)
            }
            Self::Pass => write!(f, "ps"),
        }
    }
}
