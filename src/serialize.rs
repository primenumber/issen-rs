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

#[derive(Debug)]
pub enum EncodeError {
    OutOfRange,
}

pub fn encode_utf16(input: u32) -> Result<char, EncodeError> {
    if input > 0xF7FF {
        return Err(EncodeError::OutOfRange);
    }
    let u32data = input + 0x0800; // 3-byte UTF-8
    Ok(char::from_u32(u32data).unwrap())
}

fn encode_bits(bits: i16, length: usize) -> Vec<bool> {
    let mut result = Vec::new();
    for i in 0..length {
        let bit = (bits >> (length - 1 - i)) & 1;
        result.push(bit == 1);
    }
    result
}

fn compress_word(data: i16) -> Vec<bool> {
    if data == 0 {
        vec![false, false]
    } else {
        let sign = data < 0;
        let data_abs = data.abs();
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

pub fn compress(data: &[i16]) -> Vec<bool> {
    let mut result_bits = Vec::new();
    for &word in data {
        result_bits.append(&mut compress_word(word));
    }
    result_bits
}
