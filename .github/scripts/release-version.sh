#!/usr/bin/env bash
# A tag must describe the version of the checked-out crate.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 TAG (empty for untagged dry run)" >&2; exit 2; }
root=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
cd "$root"
version=$(cargo metadata --no-deps --format-version 1 --locked |
    jq -er '[.packages[] | select(.name == "ngspice-rs") | .version] | if length == 1 then .[0] else error("expected one ngspice-rs package") end')
if [[ -n "$1" && "$1" != "v$version" ]]; then
    printf 'Tag %s does not match Cargo.toml version %s\n' "$1" "$version" >&2
    exit 1
fi
printf '%s\n' "$version"
