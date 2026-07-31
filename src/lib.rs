//! # simdet — SIMD backend detection
//!
//! **Name reservation. There is no API here yet.**
//!
//! `simdet` will hold the runtime CPU feature detection and backend selection
//! layer currently duplicated across
//! [`fff`](https://github.com/nanithefkuc/fff) and
//! [`cafft`](https://github.com/nanithefkuc/cafft), and needed by further
//! SIMD-accelerated crates.
//!
//! Planned scope, and nothing beyond it:
//!
//! - A `Backend` enum whose ordering encodes capability, with parsing,
//!   display, and a stable name per backend.
//! - One host probe implementation, so a consumer's supported-set table
//!   cannot silently fall out of date with a newly added backend.
//! - A selection primitive: declare the backends a crate implements, get a
//!   resolved choice that is detected, narrowed, re-proven against the host,
//!   and then adjusted by a **downgrade-only** environment override. Refusing
//!   to upgrade is a soundness property, not a preference.
//!
//! No kernels, no intrinsics, no dependencies, no build script. Consumers keep
//! their own `#[target_feature]` code; this crate only answers which of it is
//! legal to call on the current host.

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_code)]
