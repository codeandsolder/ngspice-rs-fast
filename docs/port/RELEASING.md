# Releasing ngspice-rs to crates.io

`ngspice-rs` is published as a single crate by
[`.github/workflows/publish.yml`](../../.github/workflows/publish.yml). The
`xtask` workspace member has `publish = false` and is never uploaded.

## What the workflow does

1. **Shared strict gate** (`strict-rust` job): runs the immutable
   rust-skills2 workflow against the repository's dated, quarantined nightly.
2. **Release gate** (`verify` job, no credentials):
   - checks that the pushed tag equals `v` + `version` in `Cargo.toml`;
   - checks new registry dependency versions against the default branch with
     the shared 72-hour release quarantine;
   - verifies all five archive layouts and negative tag/target checks;
   - `cargo test --workspace --locked`;
   - `cargo xtask golden verify`;
   - `cargo publish --dry-run --locked`, which builds the packaged crate in
     isolation, then lists the packaged files in the log.
3. **Publish** (`publish` job): runs only after both gates pass, in the
   `crates-io` GitHub environment, and uploads with `cargo publish --locked`.

Every compiling job resolves the same repository-pinned nightly through the
shared 72-hour compiler-age resolver and uses canonical sccache setup and
finalization.

The opt-in `NGSPICE_BIN` live-C checks are not part of the workflow: CI has no
C reference build. Run them locally before tagging (see
[VERIFICATION.md](VERIFICATION.md)).

The package contents are set by `include` in `Cargo.toml`: `src/`, `examples/`,
the manifest and lockfile, `README.md`, `RUST_PORT.md` and the licence and
attribution files (`COPYING`, `NOTICE`, `AUTHORS`). `conformance/`, `tests/`,
`docs/` and `xtask/` stay in the repository only. Check the list with
`cargo package --list`.

## Triggers

| Trigger | Result |
| --- | --- |
| push tag `vX.Y.Z` | gate, then publish |
| Actions → Publish → Run workflow, `dry-run` checked (default) | gate only |
| Actions → Publish → Run workflow, `dry-run` unchecked | gate, then publish the ref's `Cargo.toml` version |

## Registry dependency prerequisite

The solver consumes the versioned model kernel contract from the preprocessing
workspace. A Git-only dependency is valid for development and binary release
builds but is not valid for publishing to crates.io. Before a crate release,
publish the intended public kernel API package, add its exact compatible
registry version requirement to the solver dependency, and confirm a locked
`cargo publish --dry-run --package ngspice-rs` passes. Keep the existing
independent kernel reference tests and do not bypass Cargo's package check.
The current Git-only dependency therefore blocks crates.io publication, not
binary archive packaging.

## Authentication: crates.io Trusted Publishing

The publish job exchanges a GitHub OIDC token for a short-lived crates.io token
via `rust-lang/crates-io-auth-action`. No API token is stored in the
repository.

Before publishing, verify the crate's authorized owner has configured this
repository's actual publisher identity on crates.io: owner `codeandsolder`,
repository `ngspice-rs-fast`, workflow `publish.yml`, environment
`crates-io`. Existing upstream crate ownership does not automatically grant
a fork permission to publish. Configure the GitHub environment with the
required reviewers; no persistent registry token is needed for routine
publication. Do not attempt to republish an existing crate version.

## Prebuilt binaries and cargo-binstall

The same tag also triggers
[`.github/workflows/release.yml`](../../.github/workflows/release.yml). It
builds `spice-rs` in release mode for `x86_64-unknown-linux-musl`,
`aarch64-unknown-linux-musl`, `x86_64-apple-darwin`, `aarch64-apple-darwin`
and `x86_64-pc-windows-msvc`. Each binary is smoke-tested on its native
architecture with the pinned compiler and canonical cache. It then creates the GitHub release for the tag
(with generated notes) and attaches the archives and a `SHA256SUMS` file:

```
ngspice-rs-<target>-v<version>.tar.gz      (.zip for Windows)
└── ngspice-rs-<target>-v<version>/
    ├── spice-rs                            (spice-rs.exe on Windows)
    └── README.md, COPYING, NOTICE, AUTHORS, LICENSE-ECL-2.0
```

`[package.metadata.binstall]` in `Cargo.toml` points `cargo binstall
ngspice-rs` at these archives; glibc Linux hosts are mapped to the static musl
builds. binstall reads that metadata from the version published on crates.io,
so the names in the workflow and the manifest must change together. The
binaries job is independent of the crates.io publish job and needs no approval.

Manual *Release binaries* runs default to dry-run: build and package every
native target, upload workflow artifacts only, and do not modify a GitHub
release. Leave the tag empty to validate the selected ref. To replace assets
on an existing tag containing the current release scripts and dated nightly
pin, specify that tag and explicitly disable dry-run. Old tags without those
contracts are not supported by the current workflow. Both modes check the
release-age quarantine against the default branch.

## Cutting a release

1. Resolve the registry dependency prerequisite, then bump `version` in
   `Cargo.toml`, run `cargo update --workspace` so
   `Cargo.lock` records it, and run the local gate, including the
   `NGSPICE_BIN` live-C checks.
2. Merge the bump to `main`.
3. Optionally run the workflow by hand with `dry-run` checked.
4. Tag the merge commit on `main` and push: `git tag -a vX.Y.Z -m "ngspice-rs X.Y.Z"
   && git push origin vX.Y.Z`. This starts both the crates.io publish (approve
   it in the `crates-io` environment) and the binary release.

crates.io versions are permanent. A bad release can be yanked
(`cargo yank --version X.Y.Z`), but the version number cannot be reused.
