/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

use anyhow::{Context, Result, ensure};
use image::{ImageBuffer, Pixel};
use pic_scale::{
    BufferStore, ImageStore, ImageStoreMut, ImageStoreScaling, ResamplingFunction, ScalingOptions,
};

pub(crate) fn resize_bilinear_u16<const N: usize, P: Pixel<Subpixel = u16>>(
    image: &ImageBuffer<P, Vec<u16>>,
    (width, height): (u32, u32),
) -> Result<ImageBuffer<P, Vec<u16>>>
where
    for<'a> ImageStore<'a, u16, N>: ImageStoreScaling<'a, u16, N>,
{
    ensure!(N == usize::from(P::CHANNEL_COUNT), "invalid channel count");
    let (dst_width, dst_height) = (width as usize, height as usize);
    ensure!(
        dst_width
            .checked_mul(N)
            .and_then(|stride| stride.checked_mul(dst_height))
            .is_some(),
        "resized image is too large"
    );

    let mut source = ImageStore::<u16, N>::from_slice(
        image.as_raw(),
        image.width() as usize,
        image.height() as usize,
    )?;
    let bit_depth = u16::BITS as usize;
    source.bit_depth = bit_depth;
    let mut output = ImageStoreMut::try_alloc_with_depth(dst_width, dst_height, bit_depth)?;
    source.scale(
        &mut output,
        ScalingOptions {
            resampling_function: ResamplingFunction::Bilinear,
            ..Default::default()
        },
    )?;

    let BufferStore::Owned(samples) = output.buffer else {
        unreachable!("allocated pic-scale storage owns its samples");
    };
    ImageBuffer::from_raw(width, height, samples).context("invalid resized image buffer")
}
