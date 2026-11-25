#!/usr/bin/env bash
# Produce exactly the archive layout consumed by cargo-binstall.
set -euo pipefail
[[ $# == 4 ]] || { echo "Usage: $0 TARGET TAG CARGO_TARGET_DIR OUTPUT_DIR" >&2; exit 2; }
target=$1
tag=$2
target_dir=$3
output_dir=$4
root=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
[[ -n "$tag" ]] || { echo "release tag required" >&2; exit 2; }
bash "$(dirname "$0")/release-version.sh" "$tag" >/dev/null
case "$target" in
    x86_64-unknown-linux-musl|aarch64-unknown-linux-musl|x86_64-apple-darwin|aarch64-apple-darwin)
        binary=spice-rs
        extension=tar.gz
        ;;
    x86_64-pc-windows-msvc)
        binary=spice-rs.exe
        extension=zip
        ;;
    *) printf 'Unsupported release target: %s\n' "$target" >&2; exit 2 ;;
esac
name="ngspice-rs-$target-$tag"
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd -P)
archive="$output_dir/$name.$extension"
[[ ! -e "$archive" ]] || { echo "Archive already exists: $archive" >&2; exit 1; }
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mkdir "$scratch/$name"
cp "$target_dir/$target/release/$binary" "$scratch/$name/"
for file in README.md COPYING NOTICE AUTHORS LICENSE-ECL-2.0; do
    cp "$root/$file" "$scratch/$name/"
done
case "$extension" in
    tar.gz)
        tar -czf "$archive" -C "$scratch" "$name"
        tar -tzf "$archive" >/dev/null
        ;;
    zip)
        (cd "$scratch" && 7z a -tzip "$archive" "$name" >/dev/null)
        7z t "$archive" >/dev/null
        ;;
esac
printf '%s\n' "$archive"
