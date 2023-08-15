use std::fmt;
use std::str::FromStr;
use thiserror::Error;

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

#[derive(Error, Debug)]
#[error("Failed to parse hand")]
pub struct ParseHandError {}

impl FromStr for Hand {
    type Err = ParseHandError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 2 {
            return Err(Self::Err {});
        }
        if s == "ps" {
            return Ok(Hand::Pass);
        }
        let s = s.as_bytes();
        const CODE_1: u8 = '1' as u32 as u8;
        const CODE_8: u8 = '8' as u32 as u8;
        const CODE_UPPER_A: u8 = 'A' as u32 as u8;
        const CODE_UPPER_H: u8 = 'H' as u32 as u8;
        const CODE_LOWER_A: u8 = 'a' as u32 as u8;
        const CODE_LOWER_H: u8 = 'h' as u32 as u8;
        let col = if s[0] >= CODE_UPPER_A && s[0] <= CODE_UPPER_H {
            s[0] - CODE_UPPER_A
        } else if s[0] >= CODE_LOWER_A && s[0] <= CODE_LOWER_H {
            s[0] - CODE_LOWER_A
        } else {
            return Err(Self::Err {});
        };
        let row = if s[1] >= CODE_1 && s[1] <= CODE_8 {
            s[1] - CODE_1
        } else {
            return Err(Self::Err {});
        };
        Ok(Hand::Play((row * 8 + col) as usize))
    }
}
