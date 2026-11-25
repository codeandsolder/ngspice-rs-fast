#!/usr/bin/env bash
# Exercise every release layout and the version guard without uploading.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
version=$(bash "$root/.github/scripts/release-version.sh" "")
tag="v$version"
[[ $(bash "$root/.github/scripts/release-version.sh" "$tag") == "$version" ]]
if bash "$root/.github/scripts/release-version.sh" "v0.0.0-invalid" >/dev/null 2>&1; then
    echo "Mismatched release tag was accepted" >&2
    exit 1
fi
for target in x86_64-unknown-linux-musl aarch64-unknown-linux-musl x86_64-apple-darwin aarch64-apple-darwin x86_64-pc-windows-msvc; do
    binary=spice-rs
    extension=tar.gz
    if [[ "$target" == *windows* ]]; then
        binary=spice-rs.exe
        extension=zip
    fi
    mkdir -p "$scratch/target/$target/release" "$scratch/unpacked/$target"
    printf 'fixture for %s\n' "$target" > "$scratch/target/$target/release/$binary"
    archive=$(bash "$root/.github/scripts/package-release.sh" "$target" "$tag" "$scratch/target" "$scratch/dist")
    name="ngspice-rs-$target-$tag"
    [[ "$archive" == "$scratch/dist/$name.$extension" ]]
    unpacked="$scratch/unpacked/$target"
    if [[ "$extension" == zip ]]; then
        7z x "-o$unpacked" "$archive" >/dev/null
    else
        tar -xzf "$archive" -C "$unpacked"
    fi
    cmp "$scratch/target/$target/release/$binary" "$unpacked/$name/$binary"
    for file in README.md COPYING NOTICE AUTHORS LICENSE-ECL-2.0; do
        cmp "$root/$file" "$unpacked/$name/$file"
    done
    [[ $(find "$unpacked" -type f | wc -l) -eq 6 ]]
done
if bash "$root/.github/scripts/package-release.sh" x86_64-unknown-linux-musl "$tag" "$scratch/target" "$scratch/dist" >/dev/null 2>&1; then
    echo "Existing release archive was overwritten" >&2
    exit 1
fi
if bash "$root/.github/scripts/package-release.sh" unsupported "$tag" "$scratch/target" "$scratch/dist" >/dev/null 2>&1; then
    echo "Unsupported release target was accepted" >&2
    exit 1
fi
printf 'All five release archive layouts and negative guards passed\n'
