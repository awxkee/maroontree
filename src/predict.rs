//! Prediction, shared between encoder and decoder.
//!
//! Both sides MUST compute predictors from the *reconstructed* buffer with
//! identical code, or they desync. That's why this lives in one place and is
//! called by `encoder` and `decoder` alike.

/// Simple left-neighbor DC predictor on reconstructed samples for the 4x4
/// block at block-coords `(bx, by)`. Leftmost column falls back to mid-grey.
pub(crate) fn left_predictor(
    recon: &[i32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    bit_depth: u8,
) -> [i32; 16] {
    let mut pred = [1i32 << (bit_depth - 1); 16];
    if bx > 0 {
        let lx = bx * 4 - 1;
        let mut sum = 0i32;
        let mut cnt = 0i32;
        for yy in 0..4 {
            let gy = by * 4 + yy;
            if gy < height && lx < width {
                sum += recon[gy * width + lx];
                cnt += 1;
            }
        }
        if cnt > 0 {
            pred = [sum / cnt; 16];
        }
    }
    pred
}
