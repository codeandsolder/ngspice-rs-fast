//! Independent limits on streaming text/schema work and retained results.
use crate::primitives::SpiceResult;

use super::unsupported;

/// Resource limits for decoding one rawfile, including lending row readers.
///
/// Limits are inclusive. Streaming has no cell or point count ceiling beyond
/// checked representability; its retained data is one header and one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawReadPolicy {
    /// Maximum bytes of any text line, including its line ending if present.
    pub line_bytes: usize,
    /// Maximum text bytes per header, including variable lines, section
    /// marker and any leading whitespace.
    pub header_bytes: usize,
    /// Maximum physical lines per header (including leading blank lines).
    pub header_lines: usize,
    /// Maximum bytes of consecutive blank lines before an ASCII value or
    /// header. Prevents unbounded work without bounding valid payload size.
    pub blank_bytes: usize,
    /// Maximum columns in any one plot.
    pub variables: usize,
    /// Maximum plots per input, including empty plots.
    pub plots: usize,
    /// Additional cumulative limits used only by collecting readers.
    pub collect: RawCollectPolicy,
}

/// Logical retained-data limits, across all plots in a single collection.
///
/// These count rows, cells and input header bytes, not allocator RSS. The row
/// ceiling controls per-row allocation overhead even for narrow plots; the
/// header and variable ceilings control metadata and schema indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawCollectPolicy {
    /// Total decoded cells (one complex value per cell even for real data).
    pub cells: usize,
    /// Total materialized rows.
    pub rows: usize,
    /// Total encoded header bytes, including variable names and whitespace.
    pub header_bytes: usize,
    /// Total retained columns across plots.
    pub variables: usize,
}

impl RawReadPolicy {
    /// Defaults bound metadata and collected data while allowing large streams.
    pub const DEFAULT: Self = Self {
        line_bytes: 64 * 1024,
        header_bytes: 16 * 1024 * 1024,
        header_lines: 131_072,
        blank_bytes: 1024 * 1024,
        variables: 65_536,
        plots: 65_536,
        collect: RawCollectPolicy {
            cells: 16 * 1024 * 1024,
            rows: 2 * 1024 * 1024,
            header_bytes: 64 * 1024 * 1024,
            variables: 262_144,
        },
    };
}

impl Default for RawReadPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A remaining budget cannot wrap, even for a caller-selected `usize::MAX`.
pub(super) fn spend(remaining: &mut usize, amount: usize, resource: &str) -> SpiceResult<()> {
    *remaining = remaining
        .checked_sub(amount)
        .ok_or_else(|| limit(resource))?;
    Ok(())
}

pub(super) fn limit(resource: &str) -> crate::primitives::SpiceError {
    unsupported(format!("rawfile resource limit exceeded: {resource}"))
}

pub(super) fn reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &str,
) -> SpiceResult<()> {
    values
        .try_reserve(additional)
        .map_err(|error| unsupported(format!("rawfile allocation failed for {resource}: {error}")))
}
