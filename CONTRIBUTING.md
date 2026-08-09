# Contributing

Thanks for your interest. Contributions go through Pull Requests; what changed
and why belong in the PR and in `CHANGELOG.md`.

## Running the tests

```sh
cargo test --all-features
cargo test --no-default-features   # no-std: resolution reports Scalar
```

Selection resolves once per process, so one run only covers the host's best
backend. Sweep the weaker ones explicitly with the `SIMD_BACKEND` override; it
is downgrade-only, so a request for a backend the host cannot execute is
ignored rather than faked:

```sh
SIMD_BACKEND=scalar cargo test --all-features
```

## Before opening a PR

```sh
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

New public items need doc comments. MSRV is 1.89 and is checked in CI; do not
reach for newer standard-library APIs without raising it deliberately.

## Commit messages

Subject lines are at most ~10 words and carry the change at a glance
(`simdispatch: short verb phrase`). Implementation detail belongs in the Pull
Request and `CHANGELOG.md`, not the message, and messages never reference
planning artifacts.
