use std::sync::{Arc, OnceLock};

#[allow(unused)]
fn fwht_raw(input: &mut [i32; 16]) {
    let mut mid = [0i32; 16];
    for i in 0..4 {
        let (a0, b0, c0, d0) = (input[i], input[4 + i], input[8 + i], input[12 + i]);
        let mut a1 = a0 + b0;
        let mut d1 = d0 - c0;
        let e1 = (a1 - d1) >> 1;
        let b1 = e1 - b0;
        let c1 = e1 - c0;
        a1 -= c1;
        d1 += b1;
        mid[i] = a1;
        mid[4 + i] = c1;
        mid[8 + i] = d1;
        mid[12 + i] = b1;
    }
    for i in 0..4 {
        let base = i * 4;
        let (a0, b0, c0, d0) = (mid[base], mid[base + 1], mid[base + 2], mid[base + 3]);
        let mut a1 = a0 + b0;
        let mut d1 = d0 - c0;
        let e1 = (a1 - d1) >> 1;
        let b1 = e1 - b0;
        let c1 = e1 - c0;
        a1 -= c1;
        d1 += b1;
        input[base] = a1;
        input[base + 1] = c1;
        input[base + 2] = d1;
        input[base + 3] = b1;
    }
}

#[allow(unused)]
fn transpose(m: &[i32; 16]) -> [i32; 16] {
    let mut o = [0i32; 16];
    for r in 0..4 {
        for c in 0..4 {
            o[c * 4 + r] = m[r * 4 + c];
        }
    }
    o
}

#[allow(unused)]
fn wht_scalar(resid: &mut [i32; 16]) {
    fwht_raw(resid);
    let t = transpose(resid);
    resid.copy_from_slice(&t)
}

// signed level (raster) such that dav1d inverse WHT reproduces resid exactly
pub fn levels_from_resid(resid: &mut [i32; 16]) {
    pub(crate) type WHT = dyn Fn(&mut [i32; 16]) + Send + Sync;
    static WHT: OnceLock<Arc<WHT>> = OnceLock::new();
    let f = WHT.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::fwht_raw_neon;
            Arc::new(fwht_raw_neon)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(wht_scalar)
        }
    });
    f(resid);
}
