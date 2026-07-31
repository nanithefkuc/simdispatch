# simdet — SIMD backend detection

**This is a name reservation. There is no API here yet.**

`simdet` will hold the runtime CPU feature detection and backend selection
layer currently duplicated across [`fff`](https://github.com/nanithefkuc/fff)
and [`cafft`](https://github.com/nanithefkuc/cafft), and needed by further
SIMD-accelerated crates in the same stack.

## Planned scope

- A `Backend` enum whose variant ordering encodes capability, with parsing,
  display, and a stable identifier per backend.
- One host probe implementation, so a consumer's supported-set table cannot
  silently fall out of date when a backend is added.
- A selection primitive: a crate declares the backends it implements and gets
  a resolved choice that is detected, narrowed to that set, re-proven against
  the host, and then adjusted by a **downgrade-only** environment override.
  Refusing to upgrade is a soundness property, not a preference — running
  vector code the CPU cannot execute is undefined behaviour, not a
  configuration choice.

## Explicit non-scope

No kernels, no intrinsics, no dependencies, no build script, no global
registry, no dynamic dispatch table. Consumers keep their own
`#[target_feature]` functions; this crate only answers which of them are legal
to call on the current host.

## License

MIT. See [LICENSE](LICENSE).
