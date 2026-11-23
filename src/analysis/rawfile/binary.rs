//! The binary rawfile codec: native `double` payloads behind a text header.
//!
//! Ported from the `if (binary)` branch of `raw_write()`
//! (`src/frontend/rawfile.c:74-91` for the file mode, `:223-259` for the
//! payload) and from the `is_ascii == FALSE` branch of `raw_read()`
//! (`:591-593` for the `Values:`/`Binary:` dispatch, `:667-694` for the reads).
//! See `docs/port/RAWFILES.md` for the same layout in table form and for the
//! variants this codec deliberately rejects.
//!
//! A binary rawfile is the ASCII header, the line `Binary:`, and then the
//! payload:
//!
//! | bytes | field |
//! | --- | --- |
//! | text | `Title:`, `Date:`, `Command:`, `Plotname:`, `Flags:`, `No. Variables:`, `No. Points:`, `Variables:` and one line per variable — byte for byte what the ASCII form writes |
//! | text | `Binary:` |
//! | payload | point-major values, no alignment or padding: for every point, every variable in header order |
//!
//! One value occupies 8 bytes in a `real` plot (the real part) and 16 bytes in a
//! `complex` plot (`re` then `im`). Every value is a C `double` stored with a
//! plain `fwrite(&dd, sizeof(double), 1, fp)`, so the payload is 8-byte
//! IEEE-754 binary64 in the writing machine's native byte order; the file
//! records nothing about that order, so [`super::RawFileReader::with_byte_order`] is the
//! only way to read a payload other than the little-endian default.
//!
//! `raw_read()` reads exactly `No. Points:` x `No. Variables:` values
//! (`rawfile.c:637-694`) and skips values it does not need only when the plot is
//! padded (the default). This codec therefore requires a padded payload: the
//! length of an unpadded one is not derivable from the header, and the port's
//! [`Plot`] is rectangular.

use std::io::{self, BufRead, Write};

use crate::primitives::{Complex, Real};

use crate::analysis::results::{Plot, PlotFlags};

/// Bytes in one stored `double`.
const DOUBLE_BYTES: usize = 8;

/// Byte order of the `double`s in a binary payload.
///
/// `fwrite()` stores the writing machine's representation, and the file does not
/// say which that was, so the reader never guesses: it decodes little-endian
/// unless [`super::RawFileReader::with_byte_order`] selects otherwise. Little-endian is
/// what x86-64 and aarch64 use, the platforms the port supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BinaryByteOrder {
    /// The order [`super::RawFile::to_binary`] writes.
    #[default]
    LittleEndian,
    /// For a rawfile written on a big-endian machine.
    BigEndian,
}

impl BinaryByteOrder {
    /// Decodes one stored `double`.
    const fn decode(self, bytes: [u8; DOUBLE_BYTES]) -> Real {
        match self {
            Self::LittleEndian => Real::from_le_bytes(bytes),
            Self::BigEndian => Real::from_be_bytes(bytes),
        }
    }

    /// Encodes one `double` for storage.
    const fn encode(self, value: Real) -> [u8; DOUBLE_BYTES] {
        match self {
            Self::LittleEndian => value.to_le_bytes(),
            Self::BigEndian => value.to_be_bytes(),
        }
    }
}

/// Bytes one value occupies: `re` for a real plot, `re` and `im` for a complex one.
pub(super) const fn value_width(flags: PlotFlags) -> usize {
    match flags {
        PlotFlags::Real => DOUBLE_BYTES,
        PlotFlags::Complex => 2 * DOUBLE_BYTES,
    }
}

/// Decode complete cells directly from a borrowed block. Both widths consist
/// of whole binary64 words; conversion preserves every bit, including NaNs
/// and signed zero. No native struct representation is used.
fn decode_cells(bytes: &[u8], flags: PlotFlags, order: BinaryByteOrder, row: &mut Vec<Complex>) {
    let (words, _) = bytes.as_chunks::<DOUBLE_BYTES>();
    match flags {
        PlotFlags::Real => row.extend(words.iter().map(|&word| Complex::real(order.decode(word)))),
        PlotFlags::Complex => {
            let (pairs, _) = words.as_chunks::<2>();
            row.extend(
                pairs
                    .iter()
                    .map(|&[re, im]| Complex::new(order.decode(re), order.decode(im))),
            );
        }
    }
}

/// Lend complete cells from the input buffer, without consuming any bytes of
/// the following row or plot. A sub-cell fragment is the sole copied input.
/// The caller checked the total payload length, and is requesting one of its
/// declared rows. Count every byte even when EOF occurs within a word.
pub(super) fn read_row(
    input: &mut impl BufRead,
    columns: usize,
    flags: PlotFlags,
    order: BinaryByteOrder,
    consumed: &mut usize,
    row: &mut Vec<Complex>,
) -> io::Result<bool> {
    let width = value_width(flags);
    let mut remaining = columns * width;
    let mut tail = [0; 2 * DOUBLE_BYTES];
    let mut filled = 0;
    while remaining != 0 {
        let available = match input.fill_buf() {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        let count = available.len().min(remaining);
        if count == 0 {
            return Ok(false);
        }
        let mut bytes = &available[..count];
        // The ordinary in-memory or fully buffered row needs no fragment
        // arithmetic or local copy at all.
        if filled == 0 && count == remaining {
            decode_cells(bytes, flags, order, row);
            input.consume(count);
            *consumed += count;
            return Ok(true);
        }
        if filled != 0 {
            let take = (width - filled).min(bytes.len());
            tail[filled..filled + take].copy_from_slice(&bytes[..take]);
            filled += take;
            bytes = &bytes[take..];
            if filled == width {
                decode_cells(&tail[..width], flags, order, row);
                filled = 0;
            }
        }
        let aligned = bytes.len() / width * width;
        if aligned != 0 {
            decode_cells(&bytes[..aligned], flags, order, row);
        }
        let remainder = &bytes[aligned..];
        if !remainder.is_empty() {
            tail[..remainder.len()].copy_from_slice(remainder);
            filled = remainder.len();
        }
        input.consume(count);
        remaining -= count;
        *consumed += count;
    }
    Ok(true)
}

/// The scratch size is independent of both row width and declared point
/// count, and divisible by either cell width.
const BLOCK_BYTES: usize = 32 * 1024;

/// Reusable encoding scratch. Both retained and directly streamed plots
/// coalesce rows into bounded blocks; finish_plot flushes the final partial
/// block before backpatching the point count. The allocation is lazy, so ASCII
/// and zero-width binary plots do not allocate it.
#[derive(Debug, Default)]
pub(super) struct Encoder {
    bytes: Vec<u8>,
    used: usize,
}

impl Encoder {
    pub(super) const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            used: 0,
        }
    }

    /// Emit a full or final partial block. On failure, discard that block
    /// rather than retrying a partially written prefix on a subsequent call.
    pub(super) fn flush(&mut self, out: &mut impl Write) -> io::Result<()> {
        if self.used == 0 {
            return Ok(());
        }
        let result = out.write_all(&self.bytes[..self.used]);
        self.used = 0;
        result
    }

    /// Append one validated row in header order, flushing full blocks.
    pub(super) fn write_row(
        &mut self,
        out: &mut impl Write,
        plot: &Plot,
        point: &[Complex],
        order: BinaryByteOrder,
    ) -> io::Result<()> {
        if point.is_empty() {
            return Ok(());
        }
        if self.bytes.is_empty() {
            // Initialize the storage once; every active word is subsequently
            // overwritten, so neither a row nor a flush clears its bytes.
            self.bytes.resize(BLOCK_BYTES, 0);
        }
        let flags = plot.flags();
        let width = value_width(flags);
        let mut offset = 0;
        while offset < point.len() {
            let count = ((BLOCK_BYTES - self.used) / width).min(point.len() - offset);
            let end = offset + count;
            let start = self.used;
            self.used += count * width;
            let (words, _) = self.bytes[start..self.used].as_chunks_mut::<DOUBLE_BYTES>();
            let values = &point[offset..end];
            match flags {
                PlotFlags::Real => {
                    for (word, value) in words.iter_mut().zip(values) {
                        *word = order.encode(value.re);
                    }
                }
                PlotFlags::Complex => {
                    let (pairs, _) = words.as_chunks_mut::<2>();
                    for ([re, im], (variable, value)) in pairs
                        .iter_mut()
                        .zip(plot.variables()[offset..end].iter().zip(values))
                    {
                        *re = order.encode(value.re);
                        // raw_write writes positive zero for a real vector.
                        *im = order.encode(if variable.is_real { 0.0 } else { value.im });
                    }
                }
            }
            if self.used == BLOCK_BYTES {
                self.flush(out)?;
            }
            offset = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::results::Variable;

    #[derive(Default)]
    struct Fragments {
        bytes: Vec<u8>,
        limit: usize,
        calls: usize,
        interrupt: bool,
        fail_at: Option<usize>,
    }

    impl Write for Fragments {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.interrupt && self.calls % 3 == 0 {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let remaining = self
                .fail_at
                .map_or(usize::MAX, |end| end - self.bytes.len());
            if remaining == 0 {
                return Err(io::Error::other("injected write failure"));
            }
            let count = remaining.min(self.limit).min(bytes.len());
            self.bytes.extend_from_slice(&bytes[..count]);
            Ok(count)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn plot(flags: PlotFlags, columns: usize) -> Plot {
        let mut builder = Plot::builder("p", "AC Analysis", flags);
        for i in 0..columns {
            let name = format!("v{i}");
            builder.push_variable(if !flags.is_complex() || i % 2 == 0 {
                Variable::new(name, "voltage")
            } else {
                Variable::complex(name, "voltage")
            });
        }
        builder.build()
    }

    // Construct expected bits independently; use the scalar C layout rather
    // than invoking the new block encoder to generate its own oracle.
    fn reference(plot: &Plot, row: &[Complex], order: BinaryByteOrder) -> Vec<u8> {
        let mut bytes = Vec::new();
        let word = |value: Real| match order {
            BinaryByteOrder::LittleEndian => value.to_bits().to_le_bytes(),
            BinaryByteOrder::BigEndian => value.to_bits().to_be_bytes(),
        };
        for (variable, value) in plot.variables().iter().zip(row) {
            bytes.extend(word(value.re));
            if plot.flags().is_complex() {
                bytes.extend(word(if variable.is_real { 0. } else { value.im }));
            }
        }
        bytes
    }

    #[test]
    fn binary_blocks_match_scalar_layout_and_bound_io_calls() {
        for flags in [PlotFlags::Real, PlotFlags::Complex] {
            let width = value_width(flags);
            // Narrow rows coalesce; wide rows cross multiple blocks.
            for columns in [
                3,
                BLOCK_BYTES / width - 1,
                BLOCK_BYTES / width + 3,
                2 * BLOCK_BYTES / width + 1,
            ] {
                let plot = plot(flags, columns);
                let values = [
                    0.,
                    -0.,
                    Real::INFINITY,
                    Real::NEG_INFINITY,
                    Real::from_bits(0x7ff8_1234_0000_0001),
                    Real::from_bits(1),
                    -123.25,
                ];
                let row: Vec<_> = (0..columns)
                    .map(|i| Complex::new(values[i % values.len()], values[(i + 1) % values.len()]))
                    .collect();
                for order in [BinaryByteOrder::LittleEndian, BinaryByteOrder::BigEndian] {
                    let expected = reference(&plot, &row, order);
                    let mut encoder = Encoder::new();
                    let mut out = Fragments {
                        limit: usize::MAX,
                        ..Fragments::default()
                    };
                    for _ in 0..3 {
                        encoder.write_row(&mut out, &plot, &row, order).unwrap();
                    }
                    encoder.flush(&mut out).unwrap();
                    assert_eq!(out.calls, (expected.len() * 3).div_ceil(BLOCK_BYTES));
                    assert_eq!(out.bytes, expected.repeat(3));
                    assert_eq!(encoder.bytes.capacity(), BLOCK_BYTES);
                    // Reuse the same scratch for streamed rows, including a
                    // write_all with partial writes and Interrupted results.
                    let mut tiny = Fragments {
                        limit: 3,
                        interrupt: true,
                        ..Fragments::default()
                    };
                    encoder.write_row(&mut tiny, &plot, &row, order).unwrap();
                    encoder.flush(&mut tiny).unwrap();
                    assert_eq!(tiny.bytes, expected);
                    // Read exactly one row from a buffer that also includes
                    // the next row and a following text header.
                    let mut payload = expected.repeat(2);
                    payload.extend_from_slice(b"Title: next\n");
                    let mut input = payload.as_slice();
                    let mut decoded = Vec::with_capacity(columns);
                    let mut consumed = 0;
                    for _ in 0..2 {
                        decoded.clear();
                        assert!(
                            read_row(
                                &mut input,
                                columns,
                                flags,
                                order,
                                &mut consumed,
                                &mut decoded
                            )
                            .unwrap()
                        );
                        for ((value, original), variable) in
                            decoded.iter().zip(&row).zip(plot.variables())
                        {
                            assert_eq!(value.re.to_bits(), original.re.to_bits());
                            let im = if flags.is_complex() && !variable.is_real {
                                original.im
                            } else {
                                0.
                            };
                            assert_eq!(value.im.to_bits(), Real::to_bits(im));
                        }
                    }
                    assert_eq!(consumed, expected.len() * 2);
                    assert_eq!(input, b"Title: next\n");
                }
            }
        }
    }

    #[test]
    fn binary_encoder_discards_failed_blocks_without_replaying_a_prefix() {
        let plot = plot(PlotFlags::Complex, BLOCK_BYTES / 16 + 1);
        let row = vec![Complex::new(-0., 4.); plot.variable_count()];
        let order = BinaryByteOrder::BigEndian;
        let expected = reference(&plot, &row, order);
        let mut encoder = Encoder::new();
        let mut zero = Fragments::default();
        let error = encoder
            .write_row(&mut zero, &plot, &row, order)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::WriteZero);
        assert_eq!(encoder.used, 0);
        for stop in [0, 1, 15, BLOCK_BYTES - 1, BLOCK_BYTES, expected.len() - 1] {
            let mut out = Fragments {
                limit: 31,
                interrupt: true,
                fail_at: Some(stop),
                ..Fragments::default()
            };
            let error = encoder
                .write_row(&mut out, &plot, &row, order)
                .and_then(|()| encoder.flush(&mut out))
                .unwrap_err();
            assert_eq!(error.to_string(), "injected write failure");
            assert_eq!(out.bytes, expected[..stop]);
            assert_eq!(encoder.used, 0);
            // Higher-level transactions poison on errors; even a directly
            // reused encoder must not replay any partially written block.
            let mut next = Vec::new();
            encoder.write_row(&mut next, &plot, &row, order).unwrap();
            encoder.flush(&mut next).unwrap();
            assert_eq!(next, expected);
        }
    }
}
