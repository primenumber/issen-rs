use rayon::prelude::*;
pub struct SparseMat {
    weight: Vec<f64>,
    col_size: usize,
    row_starts: Vec<usize>,
    cols: Vec<u32>,
}

impl SparseMat {
    pub fn new(weight: Vec<f64>, col_size: usize, row_starts: Vec<usize>, cols: Vec<u32>) -> SparseMat {
        SparseMat {
            weight,
            col_size,
            row_starts,
            cols,
        }
    }
    fn row_size(&self) -> usize {
        self.row_starts.len() - 1
    }

    fn transpose(&self) -> SparseMat {
        let mut weight_t = vec![Vec::new(); self.col_size];
        let mut cols_t = vec![Vec::new(); self.col_size];

        for row in 0..self.row_size() {
            let row_start = self.row_starts[row];
            let row_end = self.row_starts[row + 1];
            for (&col, &w) in self.cols[row_start..row_end]
                .iter()
                .zip(&self.weight[row_start..row_end])
            {
                cols_t[col as usize].push(row as u32);
                weight_t[col as usize].push(w);
            }
        }
        let mut row_starts_t = Vec::new();
        let mut offset = 0;
        for col_t in &cols_t {
            row_starts_t.push(offset);
            offset += col_t.len();
        }
        row_starts_t.push(offset);
        SparseMat {
            weight: weight_t.into_iter().flatten().collect(),
            col_size: self.row_size(),
            row_starts: row_starts_t,
            cols: cols_t.into_iter().flatten().collect(),
        }
    }

    // y = A*x
    fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        y.par_iter_mut().enumerate().for_each(|(row, elem)| {
            *elem = unsafe {
                let row_start = *self.row_starts.get_unchecked(row);
                let row_end = *self.row_starts.get_unchecked(row + 1);
                let mut ans = 0.;
                for (col, w) in self.cols[row_start..row_end]
                    .iter()
                    .zip(&self.weight[row_start..row_end])
                {
                    ans += w * x.get_unchecked(*col as usize);
                }
                ans
            };
        });
    }
}

fn norm(x: &[f64]) -> f64 {
    x.par_iter().map(|x| x * x).sum()
}

fn l1_norm(x: &[f64]) -> f64 {
    x.par_iter().map(|x| x.abs()).sum()
}

// solve min_a ||spm * a - b|| by CGLS method
pub fn cgls(spm: &SparseMat, a: &mut [f64], b: &[f64], iter_num: usize) {
    let mut pa = vec![0.; spm.row_size()];
    spm.mul_vec(a, &mut pa);
    let mut dsc = vec![0.; spm.row_size()];
    for i in 0..spm.row_size() {
        dsc[i] = b[i] - pa[i];
    }
    let mut p = vec![0.; spm.col_size];
    let spm_t = spm.transpose();
    spm_t.mul_vec(&dsc, &mut p);
    let mut re = p.clone();
    let mut old_re_norm = norm(&re);
    let mut q = vec![0.; spm.row_size()];
    let mut diff = vec![0.; spm.row_size()];
    for i in 0..iter_num {
        spm.mul_vec(&p, &mut q);
        let alpha = old_re_norm / norm(&q);
        a.par_iter_mut().enumerate().for_each(|(idx, elem)| {
            *elem += alpha * p[idx];
        });
        dsc.par_iter_mut().enumerate().for_each(|(idx, elem)| {
            *elem -= alpha * q[idx];
        });
        spm_t.mul_vec(&dsc, &mut re);
        let new_re_norm = norm(&re);
        if i % 10 == 0 {
            spm.mul_vec(a, &mut pa);
            let except_l2_norm_len = spm.row_size() - spm.col_size;
            for j in 0..except_l2_norm_len {
                diff[j] = b[j] - pa[j];
            }
            eprintln!(
                "Step: {}, CGLS Diff: {}, L1 err: {}",
                i,
                (norm(&diff[0..except_l2_norm_len]) / except_l2_norm_len as f64).sqrt(),
                l1_norm(&diff[0..except_l2_norm_len]) / except_l2_norm_len as f64,
            );
        }
        if new_re_norm < 1.0 {
            break;
        }
        let beta = new_re_norm / old_re_norm;
        p.par_iter_mut().enumerate().for_each(|(idx, elem)| {
            *elem = re[idx] + beta * *elem;
        });
        old_re_norm = new_re_norm;
    }
}
