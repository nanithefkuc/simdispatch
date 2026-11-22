//! Toolchain gate for the `sve` feature.
//!
//! The SVE/SVE2 intrinsic surface requires the unstable
//! `stdarch_aarch64_sve` library feature, so the code under
//! `src/arch/aarch64.rs` may compile only where the toolchain can carry
//! that gate. This script is the single place that decision is made: it
//! emits the `simdispatch_sve` cfg when the `sve` feature is requested,
//! the target is little-endian AArch64, and the compiler is a nightly at
//! or above the verified floor; every other build keeps the surface
//! absent and reports why with a visible warning.

use std::env;
use std::process::Command;

/// Verified compiler floor for the complete SVE and SVE2 surface.
const MIN_RELEASE: (u32, u32, u32) = (1, 101, 0);
/// Commit date of that nightly, `rustc 1.101.0-nightly (d080e7dff
/// 2026-09-27)`. Commit dates are zero-padded ISO, so the comparison is
/// lexicographic.
const MIN_COMMIT_DATE: &str = "2026-09-27";

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    // Declared in every configuration so the cfg reads below never trip
    // the `unexpected_cfgs` lint, feature on or off.
    println!("cargo::rustc-check-cfg=cfg(simdispatch_sve)");
    println!("cargo::rustc-check-cfg=cfg(sve_canary_required)");

    if env::var_os("CARGO_FEATURE_SVE").is_none() {
        return;
    }

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap_or_default();
    if arch != "aarch64" || endian != "little" {
        warn("the target is not little-endian aarch64");
        return;
    }

    match probe_rustc() {
        Some((release, commit_date)) if eligible(&release, commit_date.as_deref()) => {
            println!("cargo::rustc-cfg=simdispatch_sve");
        }
        Some((release, commit_date)) => warn(&format!(
            "this compiler is {release} (commit date {}), below the floor",
            commit_date.unwrap_or_else(|| "unknown".to_owned())
        )),
        None => warn("the compiler version could not be determined"),
    }
}

fn warn(reason: &str) {
    println!(
        "cargo::warning=sve feature not compiled: {reason}; the SVE surface \
         requires a nightly rustc of 1.101.0 or newer with a commit date of \
         {MIN_COMMIT_DATE} or later on a little-endian aarch64 target"
    );
}

/// `(release, commit_date)` as reported by `rustc -vV`, e.g.
/// `("1.101.0-nightly", Some("2026-09-27"))`.
fn probe_rustc() -> Option<(String, Option<String>)> {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let output = Command::new(rustc).arg("-vV").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let verbose = String::from_utf8(output.stdout).ok()?;
    let release = verbose
        .lines()
        .find_map(|line| line.strip_prefix("release: "))?
        .to_owned();
    let commit_date = verbose
        .lines()
        .find_map(|line| line.strip_prefix("commit-date: "))
        .filter(|date| *date != "unknown")
        .map(str::to_owned);
    Some((release, commit_date))
}

fn eligible(release: &str, commit_date: Option<&str>) -> bool {
    let Some((version, channel)) = release.split_once('-') else {
        return false;
    };
    if channel != "nightly" {
        return false;
    }
    let mut numbers = version.split('.');
    let major = numbers.next().and_then(|n| n.parse().ok());
    let minor = numbers.next().and_then(|n| n.parse().ok());
    let patch = numbers.next().and_then(|n| n.parse().ok());
    if !matches!((major, minor, patch), (Some(a), Some(b), Some(c)) if (a, b, c) >= MIN_RELEASE) {
        return false;
    }
    commit_date.is_some_and(|date| date >= MIN_COMMIT_DATE)
}
