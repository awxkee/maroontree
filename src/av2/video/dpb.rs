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
//! Decoded picture buffer for the low-delay forward-reference set.

use crate::Pixel;

/// A reconstructed reference frame: planar samples the encoder can predict from.
#[derive(Clone)]
pub struct RefFrame<T: Pixel> {
    pub planes: std::sync::Arc<Vec<Vec<T>>>, // shared immutable Y, [U, V]
    pub width: usize,
    pub height: usize,
    pub strides: Vec<usize>,
    pub order_hint: u64,
}

/// Three rotating decoded slots. Presets may search the newest one, two or
/// three forward references without copying frame data.
pub struct Dpb<T: Pixel> {
    slots: [Option<RefFrame<T>>; 3],
    last_slot: usize,
    next_refresh_slot: usize,
}

impl<T: Pixel> Default for Dpb<T> {
    fn default() -> Self {
        Self {
            slots: [None, None, None],
            last_slot: 0,
            next_refresh_slot: 0,
        }
    }
}

impl<T: Pixel> Dpb<T> {
    pub fn last(&self) -> Option<&RefFrame<T>> {
        self.slots[self.last_slot].as_ref()
    }

    pub fn slot(&self, slot: usize) -> Option<&RefFrame<T>> {
        self.slots.get(slot).and_then(Option::as_ref)
    }

    pub fn populated_slots(&self, limit: usize) -> impl Iterator<Item = usize> + '_ {
        let mut slots = [0usize, 1, 2];
        slots.sort_by_key(|&slot| {
            std::cmp::Reverse(self.slots[slot].as_ref().map(|frame| frame.order_hint))
        });
        slots
            .into_iter()
            .filter(|&slot| self.slots[slot].is_some())
            .take(limit.min(self.slots.len()))
    }

    pub fn next_refresh_slot(&self) -> usize {
        self.next_refresh_slot
    }

    pub fn latest_slot(&self) -> usize {
        self.last_slot
    }

    /// The current closed-loop key syntax establishes decoder slot 0. Slot 1
    /// becomes valid when the first inter frame refreshes it.
    pub fn reset_with_key(&mut self, frame: RefFrame<T>) {
        self.slots = [Some(frame), None, None];
        self.last_slot = 0;
        // The closed-loop key is available in decoder slot 0. Populate slot 1
        // with the first inter reconstruction, then alternate thereafter.
        self.next_refresh_slot = 1;
    }

    /// Install an inter reconstruction into the slot signaled in the header.
    pub fn refresh_slot(&mut self, slot: usize, frame: RefFrame<T>) {
        debug_assert!(slot < self.slots.len());
        self.slots[slot] = Some(frame);
        self.last_slot = slot;
        self.next_refresh_slot = (slot + 1) % self.slots.len();
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(order_hint: u64) -> RefFrame<f32> {
        RefFrame {
            planes: std::sync::Arc::new(vec![vec![order_hint as f32]]),
            width: 1,
            height: 1,
            strides: vec![1],
            order_hint,
        }
    }

    #[test]
    fn rotating_slots_are_returned_in_recency_order() {
        let mut dpb = Dpb::default();
        dpb.reset_with_key(frame(0));
        assert_eq!(dpb.populated_slots(3).collect::<Vec<_>>(), [0]);
        assert_eq!(dpb.next_refresh_slot(), 1);

        dpb.refresh_slot(1, frame(1));
        dpb.refresh_slot(2, frame(2));
        assert_eq!(dpb.populated_slots(3).collect::<Vec<_>>(), [2, 1, 0]);
        assert_eq!(dpb.populated_slots(2).collect::<Vec<_>>(), [2, 1]);

        dpb.refresh_slot(0, frame(3));
        assert_eq!(dpb.populated_slots(3).collect::<Vec<_>>(), [0, 2, 1]);
        assert_eq!(dpb.next_refresh_slot(), 1);
    }
}
