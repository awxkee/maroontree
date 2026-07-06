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

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Task<'a> = &'a (dyn Fn() + Sync);

struct Shared {
    job: Mutex<State>,
    post: Condvar,
    done: Condvar,
}

struct State {
    // Erased `&(dyn Fn()+Sync)`; `run` waits for all claimants before returning,
    // so the real borrow always outlives every use.
    job: Option<*const (dyn Fn() + Sync)>,
    epoch: u64,
    unclaimed: usize,
    running: usize,
    shutdown: bool,
    panic: Option<Box<dyn std::any::Any + Send>>,
}
// SAFETY: the pointer is only dereferenced under the completion invariant above.
unsafe impl Send for State {}

/// A pool of `width` worker slots (the caller is one; `width - 1` are threads).
pub(crate) struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    width: usize,
}

impl Pool {
    /// `threads` follows the encoder's budget (`0` = all cores); serial when 1.
    pub(crate) fn new(threads: usize) -> Self {
        let width = crate::av1_coder::resolve_threads(threads).max(1);
        let shared = Arc::new(Shared {
            job: Mutex::new(State {
                job: None,
                epoch: 0,
                unclaimed: 0,
                running: 0,
                shutdown: false,
                panic: None,
            }),
            post: Condvar::new(),
            done: Condvar::new(),
        });
        let handles = (1..width)
            .map(|i| {
                let sh = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("maroontree-{i}"))
                    .spawn(move || worker(&sh))
                    .expect("spawn pool worker")
            })
            .collect();
        Pool {
            shared,
            handles,
            width,
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Run `f` across up to `cap` workers (caller included) with work stealing;
    /// `f` pulls its own indices, so scheduling is dynamic.
    fn run(&self, cap: usize, f: Task) {
        let want = cap.min(self.width);
        if want <= 1 {
            f();
            return;
        }
        let sh = &*self.shared;
        // SAFETY: erase the borrow lifetime; the completion wait below ensures no
        // worker dereferences `f` after `run` returns (see `State::job`).
        let erased: *const (dyn Fn() + Sync) =
            unsafe { std::mem::transmute::<Task, *const (dyn Fn() + Sync)>(f) };
        {
            let mut s = sh.job.lock().expect("pool poisoned");
            debug_assert!(s.job.is_none(), "nested pool run");
            s.job = Some(erased);
            s.epoch += 1;
            s.unclaimed = want - 1;
            s.panic = None;
            sh.post.notify_all();
        }
        let local = catch_unwind(AssertUnwindSafe(f)).err();
        let mut s = sh.job.lock().expect("pool poisoned");
        s.unclaimed = 0;
        while s.running > 0 {
            s = sh.done.wait(s).expect("pool poisoned");
        }
        s.job = None;
        let worker_panic = s.panic.take();
        drop(s);
        if let Some(p) = local.or(worker_panic) {
            resume_unwind(p);
        }
    }

    /// Map `f` over `0..n`; results in index order.
    pub(crate) fn map_indexed<T, F>(&self, cap: usize, n: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        if self.width <= 1 || cap <= 1 || n <= 1 {
            return (0..n).map(f).collect();
        }
        let slots: Vec<Mutex<Option<T>>> = std::iter::repeat_with(|| Mutex::new(None))
            .take(n)
            .collect();
        let next = AtomicUsize::new(0);
        self.run(cap.min(n), &|| loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            let v = f(i);
            *slots[i].lock().expect("slot poisoned") = Some(v);
        });
        slots
            .into_iter()
            .map(|c| {
                c.into_inner()
                    .expect("slot poisoned")
                    .expect("index produced")
            })
            .collect()
    }

    /// Consume `items` with `f`.
    pub(crate) fn for_each<T, F>(&self, cap: usize, items: Vec<T>, f: F)
    where
        T: Send,
        F: Fn(T) + Sync,
    {
        let n = items.len();
        if self.width <= 1 || cap <= 1 || n <= 1 {
            items.into_iter().for_each(f);
            return;
        }
        let slots: Vec<Mutex<Option<T>>> = items.into_iter().map(|t| Mutex::new(Some(t))).collect();
        let next = AtomicUsize::new(0);
        self.run(cap.min(n), &|| loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            let it = slots[i].lock().expect("slot poisoned").take();
            f(it.expect("index claimed once"));
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut s = self.shared.job.lock().expect("pool poisoned");
            s.shutdown = true;
            self.shared.post.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker(sh: &Shared) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut s = sh.job.lock().expect("pool poisoned");
            loop {
                if s.shutdown {
                    return;
                }
                if s.epoch != seen && s.unclaimed > 0 {
                    if let Some(job) = s.job {
                        seen = s.epoch;
                        s.unclaimed -= 1;
                        s.running += 1;
                        break job;
                    }
                }
                s = sh.post.wait(s).expect("pool poisoned");
            }
        };
        // SAFETY: valid until `run` observes `running == 0` (see `State::job`).
        let r = catch_unwind(AssertUnwindSafe(|| unsafe { (*job)() }));
        let mut s = sh.job.lock().expect("pool poisoned");
        s.running -= 1;
        if let Err(p) = r {
            s.panic.get_or_insert(p);
        }
        if s.running == 0 {
            sh.done.notify_all();
        }
    }
}
