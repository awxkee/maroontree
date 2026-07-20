/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#[allow(clippy::vec_box)]
#[derive(Default)]
pub(crate) struct CoderScratch {
    i64s: Vec<Box<[i32; 64]>>,
    i128s: Vec<Box<[i32; 128]>>,
    u64s: Vec<Box<[u16; 64]>>,
    u128s: Vec<Box<[u16; 128]>>,
    i256: Vec<Box<[i32; 256]>>,
    i512: Vec<Box<[i32; 512]>>,
    i1024: Vec<Box<[i32; 1024]>>,
    i4096: Vec<Box<[i32; 4096]>>,
    u256: Vec<Box<[u16; 256]>>,
    u512: Vec<Box<[u16; 512]>>,
    u1024: Vec<Box<[u16; 1024]>>,
    u4096: Vec<Box<[u16; 4096]>>,
    f256: Vec<Box<[f32; 256]>>,
    f1024: Vec<Box<[f32; 1024]>>,
}

const SCRATCH_POISON: u8 = 0;

pub(crate) trait ScratchClass: Sized {
    fn pool(sc: &mut CoderScratch) -> &mut Vec<Box<Self>>;
    fn fresh() -> Box<Self>;
}

macro_rules! scratch_class {
    ($field:ident, $elem:ty, $n:expr, $take:ident, $put:ident, $sbuf:ident) => {
        impl ScratchClass for [$elem; $n] {
            #[inline]
            fn pool(sc: &mut CoderScratch) -> &mut Vec<Box<Self>> {
                &mut sc.$field
            }
            #[inline]
            fn fresh() -> Box<Self> {
                if SCRATCH_POISON != 0 {
                    let mut b = Box::new([<$elem>::default(); $n]);
                    let bytes = unsafe {
                        std::slice::from_raw_parts_mut(
                            b.as_mut_ptr().cast::<u8>(),
                            std::mem::size_of::<Self>(),
                        )
                    };
                    bytes.fill(SCRATCH_POISON);
                    return b;
                }
                Box::new([<$elem>::default(); $n])
            }
        }

        impl CoderScratch {
            // Some size classes are guard-only today; keep the Box API uniform.
            #[allow(dead_code)]
            #[inline]
            pub(crate) fn $take(&mut self) -> Box<[$elem; $n]> {
                <[$elem; $n] as ScratchClass>::pool(self)
                    .pop()
                    .unwrap_or_else(<[$elem; $n] as ScratchClass>::fresh)
            }
            #[allow(dead_code)]
            #[inline]
            pub(crate) fn $put(&mut self, buf: Box<[$elem; $n]>) {
                <[$elem; $n] as ScratchClass>::pool(self).push(buf);
            }
        }

        impl<'a> LossyTile<'a> {
            #[allow(dead_code)]
            #[inline]
            pub(crate) fn $sbuf(&self) -> SBuf<[$elem; $n]> {
                SBuf::take(&self.scratch)
            }
        }
    };
}

scratch_class!(i64s, i32, 64, take_i64s, put_i64s, sbuf_i64);
scratch_class!(i128s, i32, 128, take_i128s, put_i128s, sbuf_i128);
scratch_class!(u64s, u16, 64, take_u64s, put_u64s, sbuf_u64);
scratch_class!(u128s, u16, 128, take_u128s, put_u128s, sbuf_u128);
scratch_class!(i256, i32, 256, take_i256, put_i256, sbuf_i256);
scratch_class!(i512, i32, 512, take_i512, put_i512, sbuf_i512);
scratch_class!(i1024, i32, 1024, take_i1024, put_i1024, sbuf_i1024);
scratch_class!(i4096, i32, 4096, take_i4096, put_i4096, sbuf_i4096);
scratch_class!(u256, u16, 256, take_u256, put_u256, sbuf_u256);
scratch_class!(u512, u16, 512, take_u512, put_u512, sbuf_u512);
scratch_class!(u1024, u16, 1024, take_u1024, put_u1024, sbuf_u1024);
scratch_class!(u4096, u16, 4096, take_u4096, put_u4096, sbuf_u4096);
scratch_class!(f256, f32, 256, take_f256, put_f256, sbuf_f256);
scratch_class!(f1024, f32, 1024, take_f1024, put_f1024, sbuf_f1024);

/// RAII scratch buffer: derefs to the array, returns itself to the pool on
/// drop. Holds the pool by `Rc`, so it never borrows the tile.
pub(crate) struct SBuf<T: ScratchClass> {
    buf: std::mem::ManuallyDrop<Box<T>>,
    pool: std::rc::Rc<std::cell::RefCell<CoderScratch>>,
}

impl<T: ScratchClass> SBuf<T> {
    #[inline]
    fn take(pool: &std::rc::Rc<std::cell::RefCell<CoderScratch>>) -> Self {
        let buf = {
            let mut sc = pool.borrow_mut();
            T::pool(&mut sc).pop().unwrap_or_else(T::fresh)
        };
        Self {
            buf: std::mem::ManuallyDrop::new(buf),
            pool: pool.clone(),
        }
    }
}

impl<T: ScratchClass> Drop for SBuf<T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `buf` is never touched again after this take.
        let buf = unsafe { std::mem::ManuallyDrop::take(&mut self.buf) };
        T::pool(&mut self.pool.borrow_mut()).push(buf);
    }
}

impl<T: ScratchClass> std::ops::Deref for SBuf<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.buf
    }
}

impl<T: ScratchClass> std::ops::DerefMut for SBuf<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.buf
    }
}

impl<'a> LossyTile<'a> {
    /// Short-lived access to the tile's scratch pool; usable from the `&self`
    /// decision paths. Never hold the guard across another `self` call.
    #[inline]
    fn sc(&self) -> std::cell::RefMut<'_, CoderScratch> {
        self.scratch.borrow_mut()
    }
}
