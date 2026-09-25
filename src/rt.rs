//! libc-backed allocator and abort-on-panic for the no_std builds.

use core::alloc::{GlobalAlloc, Layout};

// Without std nothing links the MSVC C runtime; ask for it explicitly.
#[cfg(all(windows, target_env = "msvc", not(target_feature = "crt-static")))]
#[link(name = "msvcrt")]
extern "C" {}
#[cfg(all(windows, target_env = "msvc", target_feature = "crt-static"))]
#[link(name = "libcmt")]
extern "C" {}

struct Malloc;

// malloc alignment covers every type used here (f32, usize, u8).
const MAX_ALIGN: usize = 8;

unsafe impl GlobalAlloc for Malloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.align() > MAX_ALIGN {
            return core::ptr::null_mut();
        }
        libc::malloc(l.size()).cast()
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if l.align() > MAX_ALIGN {
            return core::ptr::null_mut();
        }
        libc::calloc(1, l.size()).cast()
    }

    unsafe fn dealloc(&self, p: *mut u8, _: Layout) {
        libc::free(p.cast());
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        if l.align() > MAX_ALIGN {
            return core::ptr::null_mut();
        }
        libc::realloc(p.cast(), size).cast()
    }
}

#[global_allocator]
static GLOBAL: Malloc = Malloc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // SAFETY: abort() is always safe to call.
    unsafe { libc::abort() }
}

// The prebuilt core/alloc still carry unwind landing pads that name this symbol.
// Panics abort, so it is never called, but a shared library must define it or
// fail at load time with "undefined symbol: rust_eh_personality".
#[cfg(not(target_env = "msvc"))]
#[no_mangle]
extern "C" fn rust_eh_personality() {}
