//! Architecture-specific capability tokens and intrinsic surfaces.
//!
//! The subtree is declared only under the `simdispatch_sve` cfg, which
//! `build.rs` defines when the `sve` feature meets a little-endian AArch64
//! target on an eligible nightly compiler; every other build omits it
//! entirely.

pub mod aarch64;
