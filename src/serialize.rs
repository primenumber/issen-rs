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

pub fn encode_base64(input: &[u8; 3], output: &mut [u8; 4]) -> Option<()> {
    let mut data = 0;
    for i in 0..3 {
        data |= (input[i] as u32) << (16 - i * 8);
    }
    for i in 0..4 {
        output[i] = encode_base64_impl(((data >> (18 - i * 6)) & 0x3f) as u8)?;
    }
    Some(())
}

fn compress_word(data: u8) -> Vec<bool> {
    match data {
        0x00 => vec![false],
        0xff => vec![true, false, false],
        0x01 => vec![true, false, true, false, false],
        0xfe => vec![true, false, true, false, true],
        0x02 => vec![true, false, true, true, false],
        0xfd => vec![true, false, true, true, true],
        data => {
            let mut res = Vec::with_capacity(10);
            res.push(true);
            res.push(true);
            for i in 0..8 {
                res.push((data >> i) & 1 == 1);
            }
            res
        }
    }
}

pub fn compress(data: &[u8]) -> (Vec<u8>, usize) {
    let mut result_bits = Vec::new();
    for &byte in data {
        result_bits.append(&mut compress_word(byte));
    }
    while result_bits.len() % 8 != 0 {
        result_bits.push(true);
    }
    let mut result = Vec::new();
    for octet in result_bits.chunks(8) {
        let mut data = 0;
        for (idx, &bit) in octet.iter().enumerate() {
            if bit {
                data |= 1 << idx;
            }
        }
        result.push(data);
    }
    (result, data.len())
}
