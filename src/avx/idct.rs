/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::idct::IdctDequant;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
struct I32x8(__m256i);

#[inline]
#[target_feature(enable = "avx2")]
fn splat(v: i32) -> __m256i {
    _mm256_set1_epi32(v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x8(src: *const i32) -> I32x8 {
    unsafe { I32x8(_mm256_loadu_si256(src.cast::<__m256i>())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8(dst: *mut i32, v: I32x8) {
    unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), v.0) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn i32x8_to_array(v: I32x8) -> [i32; 8] {
    let mut out = [0i32; 8];
    store_i32x8(out.as_mut_ptr(), v);
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn i32x8_from_array(v: [i32; 8]) -> I32x8 {
    load_i32x8(v.as_ptr())
}

impl I32x8 {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn add(self, rhs: I32x8) -> I32x8 {
        I32x8(_mm256_add_epi32(self.0, rhs.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn sub(self, rhs: I32x8) -> I32x8 {
        I32x8(_mm256_sub_epi32(self.0, rhs.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn muli(self, k: i32) -> I32x8 {
        I32x8(_mm256_mullo_epi32(self.0, splat(k)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn rsh<const SH: i32>(self, add: i32) -> I32x8 {
        I32x8(_mm256_srai_epi32::<SH>(_mm256_add_epi32(
            self.0,
            splat(add),
        )))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn clip(self, min: __m256i, max: __m256i) -> I32x8 {
        I32x8(_mm256_min_epi32(_mm256_max_epi32(self.0, min), max))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn neg(self) -> I32x8 {
        I32x8(_mm256_sub_epi32(_mm256_setzero_si256(), self.0))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8(c: &mut [I32x8; 8]) {
    let mut rows = [[0i32; 8]; 8];
    for (i, v) in c.iter().copied().enumerate() {
        rows[i] = i32x8_to_array(v);
    }
    let mut transposed = [[0i32; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            transposed[y][x] = rows[x][y];
        }
    }
    for (i, row) in transposed.iter().copied().enumerate() {
        c[i] = i32x8_from_array(row);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequant8(lvl: __m256i, q: __m256i, cf_max: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let absl = _mm256_abs_epi32(lvl);
    let masked = _mm256_and_si256(_mm256_mullo_epi32(absl, q), splat(0x00ff_ffff));
    let neg_mask = _mm256_cmpgt_epi32(zero, lvl);
    let neg_one = _mm256_and_si256(neg_mask, splat(1));
    let cap = _mm256_add_epi32(cf_max, neg_one);
    let mag = _mm256_min_epi32(masked, cap);
    let neg = _mm256_sub_epi32(zero, mag);
    _mm256_blendv_epi8(mag, neg, neg_mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequant8_dq1(lvl: __m256i, q: __m256i, cf_max: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let absl = _mm256_abs_epi32(lvl);
    let masked = _mm256_srai_epi32::<1>(_mm256_and_si256(
        _mm256_mullo_epi32(absl, q),
        splat(0x00ff_ffff),
    ));
    let neg_mask = _mm256_cmpgt_epi32(zero, lvl);
    let neg_one = _mm256_and_si256(neg_mask, splat(1));
    let cap = _mm256_add_epi32(cf_max, neg_one);
    let mag = _mm256_min_epi32(masked, cap);
    let neg = _mm256_sub_epi32(zero, mag);
    _mm256_blendv_epi8(mag, neg, neg_mask)
}

#[target_feature(enable = "avx2")]
fn dequant_levels<const N: usize, const DQ1: bool>(
    levels: &[i32; N],
    dequant: &IdctDequant,
) -> [i32; N] {
    let mut coeff_u = MaybeUninit::<[i32; N]>::uninit();
    let coeff_ptr = coeff_u.as_mut_ptr() as *mut i32;
    let q_dc = _mm256_setr_epi32(
        dequant.dc_q,
        dequant.ac_q,
        dequant.ac_q,
        dequant.ac_q,
        dequant.ac_q,
        dequant.ac_q,
        dequant.ac_q,
        dequant.ac_q,
    );
    let q_ac = splat(dequant.ac_q);
    let cfm = splat(dequant.cf_max);
    unsafe {
        let first = _mm256_loadu_si256(levels.as_ptr().cast::<__m256i>());
        let first = if DQ1 {
            dequant8_dq1(first, q_dc, cfm)
        } else {
            dequant8(first, q_dc, cfm)
        };
        _mm256_storeu_si256(coeff_ptr.cast::<__m256i>(), first);
        let mut i = 8;
        while i < N {
            let l = _mm256_loadu_si256(levels.as_ptr().add(i).cast::<__m256i>());
            let c = if DQ1 {
                dequant8_dq1(l, q_ac, cfm)
            } else {
                dequant8(l, q_ac, cfm)
            };
            _mm256_storeu_si256(coeff_ptr.add(i).cast::<__m256i>(), c);
            i += 8;
        }
        coeff_u.assume_init()
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_dct8_v(c: &mut [I32x8; 8], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);

    // --- even half: inv_dct4 on c[0], c[2], c[4], c[6] ---
    let (e0, e1, e2, e3) = (c[0], c[2], c[4], c[6]);
    let d0 = e0.add(e2).muli(181).rsh::<8>(128); // ((in0+in2)*181+128)>>8
    let d1 = e0.sub(e2).muli(181).rsh::<8>(128); // ((in0-in2)*181+128)>>8
    let d2 = e1.muli(1567).sub(e3.muli(-312)).rsh::<12>(2048).sub(e3); // *1567 - *(3784-4096)
    let d3 = e1.muli(-312).add(e3.muli(1567)).rsh::<12>(2048).add(e1);
    let p0 = clip(d0.add(d3)); // even outputs (scalar c[0],c[2],c[4],c[6])
    let p2 = clip(d1.add(d2));
    let p4 = clip(d1.sub(d2));
    let p6 = clip(d0.sub(d3));

    // --- odd half ---
    let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
    // t4a = ((in1*799 - in7*(4017-4096) + 2048)>>12) - in7   ; (4017-4096) = -79
    let t4a = in1.muli(799).sub(in7.muli(-79)).rsh::<12>(2048).sub(in7);
    let t5a0 = in5.muli(1703).sub(in3.muli(1138)).rsh::<11>(1024);
    let t6a0 = in5.muli(1138).add(in3.muli(1703)).rsh::<11>(1024);
    let t7a = in1.muli(-79).add(in7.muli(799)).rsh::<12>(2048).add(in1);

    let t4 = clip(t4a.add(t5a0));
    let t5a = clip(t4a.sub(t5a0));
    let t7 = clip(t7a.add(t6a0));
    let t6a = clip(t7a.sub(t6a0));

    let t5 = t6a.sub(t5a).muli(181).rsh::<8>(128); // ((t6a-t5a)*181+128)>>8
    let t6 = t6a.add(t5a).muli(181).rsh::<8>(128); // ((t6a+t5a)*181+128)>>8

    // --- combine (scalar t0..t3 are the even outputs p0,p2,p4,p6) ---
    c[0] = clip(p0.add(t7));
    c[1] = clip(p2.add(t6));
    c[2] = clip(p4.add(t5));
    c[3] = clip(p6.add(t4));
    c[4] = clip(p6.sub(t4));
    c[5] = clip(p4.sub(t5));
    c[6] = clip(p2.sub(t6));
    c[7] = clip(p0.sub(t7));
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_dct16_v(c: &mut [I32x8; 16], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);

    let mut e: [I32x8; 8] = std::array::from_fn(|i| c[2 * i]);
    inv_dct8_v(&mut e, min, max);

    // odd inputs (read before any write-back to c)
    let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
    let (in9, in11, in13, in15) = (c[9], c[11], c[13], c[15]);

    // stage 1 ; (4076-4096)=-20, (3612-4096)=-484, (3920-4096)=-176
    let t8a = in1.muli(401).sub(in15.muli(-20)).rsh::<12>(2048).sub(in15);
    let t9a = in9.muli(1583).sub(in7.muli(1299)).rsh::<11>(1024);
    let t10a = in5
        .muli(1931)
        .sub(in11.muli(-484))
        .rsh::<12>(2048)
        .sub(in11);
    let t11a = in13
        .muli(-176)
        .sub(in3.muli(1189))
        .rsh::<12>(2048)
        .add(in13);
    let t12a = in13.muli(1189).add(in3.muli(-176)).rsh::<12>(2048).add(in3);
    let t13a = in5.muli(-484).add(in11.muli(1931)).rsh::<12>(2048).add(in5);
    let t14a = in9.muli(1299).add(in7.muli(1583)).rsh::<11>(1024);
    let t15a = in1.muli(-20).add(in15.muli(401)).rsh::<12>(2048).add(in1);

    // stage 2 (butterflies)
    let t8 = clip(t8a.add(t9a));
    let t9 = clip(t8a.sub(t9a));
    let t10 = clip(t11a.sub(t10a));
    let t11 = clip(t11a.add(t10a));
    let t12 = clip(t12a.add(t13a));
    let t13 = clip(t12a.sub(t13a));
    let t14 = clip(t15a.sub(t14a));
    let t15 = clip(t15a.add(t14a));

    // stage 3 (rotations) ; (3784-4096)=-312
    // t10a' = ((-(t13*(-312) + t10*1567) + 2048)>>12) - t13 = ((t13*312 - t10*1567 + 2048)>>12) - t13
    let u9a = t14.muli(1567).sub(t9.muli(-312)).rsh::<12>(2048).sub(t9);
    let u14a = t14.muli(-312).add(t9.muli(1567)).rsh::<12>(2048).add(t14);
    let u10a = t13.muli(312).sub(t10.muli(1567)).rsh::<12>(2048).sub(t13);
    let u13a = t13.muli(1567).sub(t10.muli(-312)).rsh::<12>(2048).sub(t10);

    // stage 4 (butterflies)
    let v8a = clip(t8.add(t11));
    let v9 = clip(u9a.add(u10a));
    let v10 = clip(u9a.sub(u10a));
    let v11a = clip(t8.sub(t11));
    let v12a = clip(t15.sub(t12));
    let v13 = clip(u14a.sub(u13a));
    let v14 = clip(u14a.add(u13a));
    let v15a = clip(t15.add(t12));

    // stage 5 (181/256 rotations)
    let w10a = v13.sub(v10).muli(181).rsh::<8>(128);
    let w13a = v13.add(v10).muli(181).rsh::<8>(128);
    let w11 = v12a.sub(v11a).muli(181).rsh::<8>(128);
    let w12 = v12a.add(v11a).muli(181).rsh::<8>(128);

    // combine with even outputs e[0..8] (= scalar t0..t7)
    c[0] = clip(e[0].add(v15a));
    c[1] = clip(e[1].add(v14));
    c[2] = clip(e[2].add(w13a));
    c[3] = clip(e[3].add(w12));
    c[4] = clip(e[4].add(w11));
    c[5] = clip(e[5].add(w10a));
    c[6] = clip(e[6].add(v9));
    c[7] = clip(e[7].add(v8a));
    c[8] = clip(e[7].sub(v8a));
    c[9] = clip(e[6].sub(v9));
    c[10] = clip(e[5].sub(w10a));
    c[11] = clip(e[4].sub(w11));
    c[12] = clip(e[3].sub(w12));
    c[13] = clip(e[2].sub(w13a));
    c[14] = clip(e[1].sub(v14));
    c[15] = clip(e[0].sub(v15a));
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_dct32_v(c: &mut [I32x8; 32], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);

    // even half: inv_dct16 on the 16 even-indexed vectors -> e[0..16] = t0..t15
    let mut e: [I32x8; 16] = std::array::from_fn(|i| c[2 * i]);
    inv_dct16_v(&mut e, min, max);

    // odd inputs (read before any write-back)
    let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
    let (in9, in11, in13, in15) = (c[9], c[11], c[13], c[15]);
    let (in17, in19, in21, in23) = (c[17], c[19], c[21], c[23]);
    let (in25, in27, in29, in31) = (c[25], c[27], c[29], c[31]);

    // stage 1
    let mut t16a = in1
        .muli(201)
        .sub(in31.muli(4091 - 4096))
        .rsh::<12>(2048)
        .sub(in31);
    let mut t17a = in17
        .muli(3035 - 4096)
        .sub(in15.muli(2751))
        .rsh::<12>(2048)
        .add(in17);
    let mut t18a = in9
        .muli(1751)
        .sub(in23.muli(3703 - 4096))
        .rsh::<12>(2048)
        .sub(in23);
    let mut t19a = in25
        .muli(3857 - 4096)
        .sub(in7.muli(1380))
        .rsh::<12>(2048)
        .add(in25);
    let mut t20a = in5
        .muli(995)
        .sub(in27.muli(3973 - 4096))
        .rsh::<12>(2048)
        .sub(in27);
    let mut t21a = in21
        .muli(3513 - 4096)
        .sub(in11.muli(2106))
        .rsh::<12>(2048)
        .add(in21);
    let mut t22a = in13.muli(1220).sub(in19.muli(1645)).rsh::<11>(1024);
    let mut t23a = in29
        .muli(4052 - 4096)
        .sub(in3.muli(601))
        .rsh::<12>(2048)
        .add(in29);
    let mut t24a = in29
        .muli(601)
        .add(in3.muli(4052 - 4096))
        .rsh::<12>(2048)
        .add(in3);
    let mut t25a = in13.muli(1645).add(in19.muli(1220)).rsh::<11>(1024);
    let mut t26a = in21
        .muli(2106)
        .add(in11.muli(3513 - 4096))
        .rsh::<12>(2048)
        .add(in11);
    let mut t27a = in5
        .muli(3973 - 4096)
        .add(in27.muli(995))
        .rsh::<12>(2048)
        .add(in5);
    let mut t28a = in25
        .muli(1380)
        .add(in7.muli(3857 - 4096))
        .rsh::<12>(2048)
        .add(in7);
    let mut t29a = in9
        .muli(3703 - 4096)
        .add(in23.muli(1751))
        .rsh::<12>(2048)
        .add(in9);
    let mut t30a = in17
        .muli(2751)
        .add(in15.muli(3035 - 4096))
        .rsh::<12>(2048)
        .add(in15);
    let mut t31a = in1
        .muli(4091 - 4096)
        .add(in31.muli(201))
        .rsh::<12>(2048)
        .add(in1);

    // stage 2
    let mut t16 = clip(t16a.add(t17a));
    let mut t17 = clip(t16a.sub(t17a));
    let mut t18 = clip(t19a.sub(t18a));
    let mut t19 = clip(t19a.add(t18a));
    let mut t20 = clip(t20a.add(t21a));
    let mut t21 = clip(t20a.sub(t21a));
    let mut t22 = clip(t23a.sub(t22a));
    let mut t23 = clip(t23a.add(t22a));
    let mut t24 = clip(t24a.add(t25a));
    let mut t25 = clip(t24a.sub(t25a));
    let mut t26 = clip(t27a.sub(t26a));
    let mut t27 = clip(t27a.add(t26a));
    let mut t28 = clip(t28a.add(t29a));
    let mut t29 = clip(t28a.sub(t29a));
    let mut t30 = clip(t31a.sub(t30a));
    let mut t31 = clip(t31a.add(t30a));

    // stage 3
    t17a = t30
        .muli(799)
        .sub(t17.muli(4017 - 4096))
        .rsh::<12>(2048)
        .sub(t17);
    t30a = t30
        .muli(4017 - 4096)
        .add(t17.muli(799))
        .rsh::<12>(2048)
        .add(t30);
    t18a = t29
        .muli(4017 - 4096)
        .add(t18.muli(799))
        .neg()
        .rsh::<12>(2048)
        .sub(t29);
    t29a = t29
        .muli(799)
        .sub(t18.muli(4017 - 4096))
        .rsh::<12>(2048)
        .sub(t18);
    t21a = t26.muli(1703).sub(t21.muli(1138)).rsh::<11>(1024);
    t26a = t26.muli(1138).add(t21.muli(1703)).rsh::<11>(1024);
    t22a = t25.muli(1138).add(t22.muli(1703)).neg().rsh::<11>(1024);
    t25a = t25.muli(1703).sub(t22.muli(1138)).rsh::<11>(1024);

    // stage 4
    t16a = clip(t16.add(t19));
    t17 = clip(t17a.add(t18a));
    t18 = clip(t17a.sub(t18a));
    t19a = clip(t16.sub(t19));
    t20a = clip(t23.sub(t20));
    t21 = clip(t22a.sub(t21a));
    t22 = clip(t22a.add(t21a));
    t23a = clip(t23.add(t20));
    t24a = clip(t24.add(t27));
    t25 = clip(t25a.add(t26a));
    t26 = clip(t25a.sub(t26a));
    t27a = clip(t24.sub(t27));
    t28a = clip(t31.sub(t28));
    t29 = clip(t30a.sub(t29a));
    t30 = clip(t30a.add(t29a));
    t31a = clip(t31.add(t28));

    // stage 5
    t18a = t29
        .muli(1567)
        .sub(t18.muli(3784 - 4096))
        .rsh::<12>(2048)
        .sub(t18);
    t29a = t29
        .muli(3784 - 4096)
        .add(t18.muli(1567))
        .rsh::<12>(2048)
        .add(t29);
    t19 = t28a
        .muli(1567)
        .sub(t19a.muli(3784 - 4096))
        .rsh::<12>(2048)
        .sub(t19a);
    t28 = t28a
        .muli(3784 - 4096)
        .add(t19a.muli(1567))
        .rsh::<12>(2048)
        .add(t28a);
    t20 = t27a
        .muli(3784 - 4096)
        .add(t20a.muli(1567))
        .neg()
        .rsh::<12>(2048)
        .sub(t27a);
    t27 = t27a
        .muli(1567)
        .sub(t20a.muli(3784 - 4096))
        .rsh::<12>(2048)
        .sub(t20a);
    t21a = t26
        .muli(3784 - 4096)
        .add(t21.muli(1567))
        .neg()
        .rsh::<12>(2048)
        .sub(t26);
    t26a = t26
        .muli(1567)
        .sub(t21.muli(3784 - 4096))
        .rsh::<12>(2048)
        .sub(t21);

    // stage 6
    t16 = clip(t16a.add(t23a));
    t17a = clip(t17.add(t22));
    t18 = clip(t18a.add(t21a));
    t19a = clip(t19.add(t20));
    t20a = clip(t19.sub(t20));
    t21 = clip(t18a.sub(t21a));
    t22a = clip(t17.sub(t22));
    t23 = clip(t16a.sub(t23a));
    t24 = clip(t31a.sub(t24a));
    t25a = clip(t30.sub(t25));
    t26 = clip(t29a.sub(t26a));
    t27a = clip(t28.sub(t27));
    t28a = clip(t28.add(t27));
    t29 = clip(t29a.add(t26a));
    t30a = clip(t30.add(t25));
    t31 = clip(t31a.add(t24a));

    // stage 7 (181/256 rotations)
    t20 = t27a.sub(t20a).muli(181).rsh::<8>(128);
    t27 = t27a.add(t20a).muli(181).rsh::<8>(128);
    t21a = t26.sub(t21).muli(181).rsh::<8>(128);
    t26a = t26.add(t21).muli(181).rsh::<8>(128);
    t22 = t25a.sub(t22a).muli(181).rsh::<8>(128);
    t25 = t25a.add(t22a).muli(181).rsh::<8>(128);
    t23a = t24.sub(t23).muli(181).rsh::<8>(128);
    t24a = t24.add(t23).muli(181).rsh::<8>(128);

    // combine with even outputs e[0..16] (= scalar t0..t15)
    c[0] = clip(e[0].add(t31));
    c[1] = clip(e[1].add(t30a));
    c[2] = clip(e[2].add(t29));
    c[3] = clip(e[3].add(t28a));
    c[4] = clip(e[4].add(t27));
    c[5] = clip(e[5].add(t26a));
    c[6] = clip(e[6].add(t25));
    c[7] = clip(e[7].add(t24a));
    c[8] = clip(e[8].add(t23a));
    c[9] = clip(e[9].add(t22));
    c[10] = clip(e[10].add(t21a));
    c[11] = clip(e[11].add(t20));
    c[12] = clip(e[12].add(t19a));
    c[13] = clip(e[13].add(t18));
    c[14] = clip(e[14].add(t17a));
    c[15] = clip(e[15].add(t16));
    c[16] = clip(e[15].sub(t16));
    c[17] = clip(e[14].sub(t17a));
    c[18] = clip(e[13].sub(t18));
    c[19] = clip(e[12].sub(t19a));
    c[20] = clip(e[11].sub(t20));
    c[21] = clip(e[10].sub(t21a));
    c[22] = clip(e[9].sub(t22));
    c[23] = clip(e[8].sub(t23a));
    c[24] = clip(e[7].sub(t24a));
    c[25] = clip(e[6].sub(t25));
    c[26] = clip(e[5].sub(t26a));
    c[27] = clip(e[4].sub(t27));
    c[28] = clip(e[3].sub(t28a));
    c[29] = clip(e[2].sub(t29));
    c[30] = clip(e[1].sub(t30a));
    c[31] = clip(e[0].sub(t31));
}

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x8_avx2(levels: &[i32; 64], dequant: &IdctDequant) -> [i32; 64] {
    let coeff = dequant_levels::<64, false>(levels, dequant);
    let mut v: [I32x8; 8] =
        std::array::from_fn(|x| load_i32x8(unsafe { coeff.as_ptr().add(x * 8) }));

    inv_dct8_v(&mut v, dequant.rmin, dequant.rmax);

    let cmn = splat(dequant.cmin);
    let cmx = splat(dequant.cmax);
    for vv in v.iter_mut() {
        *vv = vv.rsh::<1>(1).clip(cmn, cmx);
    }

    transpose_8x8(&mut v);
    inv_dct8_v(&mut v, dequant.cmin, dequant.cmax);

    let mut out = MaybeUninit::<[i32; 64]>::uninit();
    for (y, vv) in v.iter().copied().enumerate() {
        let r = vv.rsh::<4>(8);
        store_i32x8(unsafe { (out.as_mut_ptr() as *mut i32).add(y * 8) }, r);
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x16_avx2(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    let coeff = dequant_levels::<256, false>(levels, dequant);

    let mut c_l: [I32x8; 16] =
        std::array::from_fn(|x| load_i32x8(unsafe { coeff.as_ptr().add(x * 16) }));
    let mut c_r: [I32x8; 16] =
        std::array::from_fn(|x| load_i32x8(unsafe { coeff.as_ptr().add(x * 16 + 8) }));

    inv_dct16_v(&mut c_l, dequant.rmin, dequant.rmax);
    inv_dct16_v(&mut c_r, dequant.rmin, dequant.rmax);

    let cmn = splat(dequant.cmin);
    let cmx = splat(dequant.cmax);
    for vv in c_l.iter_mut().chain(c_r.iter_mut()) {
        *vv = vv.rsh::<2>(2).clip(cmn, cmx);
    }

    let mut ll: [I32x8; 8] = c_l[0..8].try_into().unwrap();
    let mut lr: [I32x8; 8] = c_l[8..16].try_into().unwrap();
    let mut rl: [I32x8; 8] = c_r[0..8].try_into().unwrap();
    let mut rr: [I32x8; 8] = c_r[8..16].try_into().unwrap();
    transpose_8x8(&mut ll);
    transpose_8x8(&mut lr);
    transpose_8x8(&mut rl);
    transpose_8x8(&mut rr);

    let mut w_l: [I32x8; 16] = std::array::from_fn(|i| if i < 8 { ll[i] } else { rl[i - 8] });
    let mut w_r: [I32x8; 16] = std::array::from_fn(|i| if i < 8 { lr[i] } else { rr[i - 8] });

    inv_dct16_v(&mut w_l, dequant.cmin, dequant.cmax);
    inv_dct16_v(&mut w_r, dequant.cmin, dequant.cmax);

    let mut out = MaybeUninit::<[i32; 256]>::uninit();
    for r in 0..16 {
        let lo = w_l[r].rsh::<4>(8);
        let hi = w_r[r].rsh::<4>(8);
        let dst = unsafe { (out.as_mut_ptr() as *mut i32).add(r * 16) };
        store_i32x8(dst, lo);
        store_i32x8(unsafe { dst.add(8) }, hi);
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x32_avx2(levels: &[i32; 1024], dequant: &IdctDequant) -> [i32; 1024] {
    let coeff = dequant_levels::<1024, true>(levels, dequant);
    let cmn = splat(dequant.cmin);
    let cmx = splat(dequant.cmax);

    let mut scratch_u = MaybeUninit::<[i32; 1024]>::uninit();
    for yg in (0..32).step_by(8) {
        let mut cols: [I32x8; 32] =
            std::array::from_fn(|x| load_i32x8(unsafe { coeff.as_ptr().add(x * 32 + yg) }));
        inv_dct32_v(&mut cols, dequant.rmin, dequant.rmax);
        for (x, cv) in cols.iter().copied().enumerate() {
            let r = cv.rsh::<2>(2).clip(cmn, cmx);
            store_i32x8(
                unsafe { (scratch_u.as_mut_ptr() as *mut i32).add(x * 32 + yg) },
                r,
            );
        }
    }
    let scratch = unsafe { scratch_u.assume_init() };

    let mut out_u = MaybeUninit::<[i32; 1024]>::uninit();
    for xg in (0..32).step_by(8) {
        let mut rows2: [I32x8; 32] = std::array::from_fn(|_| I32x8(_mm256_setzero_si256()));
        for yg2 in (0..32).step_by(8) {
            let mut seg: [I32x8; 8] = std::array::from_fn(|v| {
                load_i32x8(unsafe { scratch.as_ptr().add((xg + v) * 32 + yg2) })
            });
            transpose_8x8(&mut seg);
            rows2[yg2..yg2 + 8].copy_from_slice(&seg);
        }
        inv_dct32_v(&mut rows2, dequant.cmin, dequant.cmax);
        for (y, rv) in rows2.iter().copied().enumerate() {
            let r = rv.rsh::<4>(8);
            store_i32x8(
                unsafe { (out_u.as_mut_ptr() as *mut i32).add(y * 32 + xg) },
                r,
            );
        }
    }
    unsafe { out_u.assume_init() }
}
