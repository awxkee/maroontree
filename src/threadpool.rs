/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

//! Persistent work-stealing threadpool. Workers spawn once and park; each
//! `map_indexed` publishes a scoped job and joins on a barrier. Zero deps.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

struct Shared {
    // Current job pointer (erased lifetime) + generation, or None when idle.
    job: Mutex<Option<(*const (dyn Fn() + Sync + 'static), usize)>>,
    cv: Condvar,
    // Workers that finished the current generation.
    done: AtomicUsize,
    done_cv: Condvar,
    done_lock: Mutex<()>,
    gen0: AtomicUsize,
    shutdown: AtomicUsize,
}

// Job pointers are only ever valid for the duration of a blocking map call,
// so sharing the raw pointer across workers within that scope is sound.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    nthreads: usize,
}

impl ThreadPool {
    /// Create a pool with `threads` total workers (including the caller thread,
    /// which participates in `map_indexed`). `<=1` yields a serial pool.
    pub fn new(threads: usize) -> Arc<Self> {
        let nthreads = threads.max(1);
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            cv: Condvar::new(),
            done: AtomicUsize::new(0),
            done_cv: Condvar::new(),
            done_lock: Mutex::new(()),
            gen0: AtomicUsize::new(0),
            shutdown: AtomicUsize::new(0),
        });
        // Caller participates, so spawn nthreads-1 background workers.
        let mut workers = Vec::new();
        for _ in 0..nthreads.saturating_sub(1) {
            let sh = shared.clone();
            workers.push(std::thread::spawn(move || worker_loop(sh)));
        }
        Arc::new(ThreadPool {
            shared,
            workers,
            nthreads,
        })
    }

    pub fn threads(&self) -> usize {
        self.nthreads
    }

    /// Work-stealing parallel map; same contract as the old `par_map_indexed`
    /// (caller participates, order preserved). Not reentrant: one map at a time.
    pub fn map_indexed<T, F>(&self, n: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        if self.nthreads <= 1 || n <= 1 {
            return (0..n).map(f).collect();
        }
        let next = AtomicUsize::new(0);
        // Each worker collects its own (index, value) pairs — no per-index lock,
        // matching the original scoped-scope scheme. Results pushed into a shared
        // Mutex<Vec<..>> once per worker (not per item).
        let out: Mutex<Vec<(usize, T)>> = Mutex::new(Vec::with_capacity(n));
        let work = || {
            let mut got: Vec<(usize, T)> = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                got.push((i, f(i)));
            }
            out.lock().unwrap().extend(got);
        };
        // Publish erased job, wake workers, participate, then barrier.
        let erased: *const (dyn Fn() + Sync) = &work;
        // SAFETY: job lives until we clear it below; workers finish (barrier)
        // before `work` drops.
        let erased: *const (dyn Fn() + Sync + 'static) = unsafe { std::mem::transmute(erased) };
        let spawned = self.workers.len();
        {
            // Reset completion count and publish the job atomically w.r.t. the
            // job lock so no prior-gen0eration worker can touch this count.
            let mut j = self.shared.job.lock().unwrap();
            self.shared.done.store(0, Ordering::Release);
            let g = self.shared.gen0.fetch_add(1, Ordering::AcqRel) + 1;
            *j = Some((erased, g));
            self.shared.cv.notify_all();
        }
        work(); // caller participates
        // Wait for all spawned workers to finish this gen0eration.
        if spawned > 0 {
            let mut guard = self.shared.done_lock.lock().unwrap();
            while self.shared.done.load(Ordering::Acquire) < spawned {
                guard = self.shared.done_cv.wait(guard).unwrap();
            }
        }
        // Clear the job only after the barrier: workers already past their gen0
        // check won't re-run it, and the next call republishes with a new gen0.
        *self.shared.job.lock().unwrap() = None;
        // Reassemble in index order.
        let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
        for (i, v) in out.into_inner().unwrap() {
            slots[i] = Some(v);
        }
        slots
            .into_iter()
            .map(|s| s.expect("index produced"))
            .collect()
    }
}

fn worker_loop(shared: Arc<Shared>) {
    let mut last_gen0 = 0usize;
    loop {
        // Wait for a new job gen0eration (or shutdown).
        let ptr = {
            let mut j = shared.job.lock().unwrap();
            loop {
                if shared.shutdown.load(Ordering::Acquire) != 0 {
                    return;
                }
                if let Some((p, g)) = *j
                    && g != last_gen0
                {
                    last_gen0 = g;
                    break p;
                }
                j = shared.cv.wait(j).unwrap();
            }
        };
        // SAFETY: caller keeps the job alive until the barrier releases.
        unsafe { (*ptr)() };
        // Signal completion.
        {
            let _l = shared.done_lock.lock().unwrap();
            shared.done.fetch_add(1, Ordering::AcqRel);
            shared.done_cv.notify_all();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(1, Ordering::Release);
        self.shared.cv.notify_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}
