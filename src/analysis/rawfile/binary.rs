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

use std::io::{self, Read, Write};

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

/// Reads precisely one cell, counting bytes even on a truncated read.
/// No allocation is driven by the declared payload size.
pub(super) fn read_value(
    input: &mut impl Read,
    flags: PlotFlags,
    order: BinaryByteOrder,
    consumed: &mut usize,
) -> io::Result<Option<Complex>> {
    let mut buffer = [0; 2 * DOUBLE_BYTES];
    let width = value_width(flags);
    let mut filled = 0;
    while filled < width {
        match input.read(&mut buffer[filled..width]) {
            Ok(0) => return Ok(None),
            Ok(count) => {
                filled += count;
                *consumed += count;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    let (real, imaginary) = buffer.split_at(DOUBLE_BYTES);
    let mut word = [0; DOUBLE_BYTES];
    word.copy_from_slice(real);
    let re = order.decode(word);
    if flags.is_complex() {
        word.copy_from_slice(imaginary);
        Ok(Some(Complex::new(re, order.decode(word))))
    } else {
        Ok(Some(Complex::real(re)))
    }
}

/// Writes one point of `plot`: a `double` per value in a `real` plot, `re`
/// then `im` in a `complex` one.
pub(super) fn write_point(
    out: &mut impl Write,
    plot: &Plot,
    point: &[Complex],
    byte_order: BinaryByteOrder,
) -> io::Result<()> {
    for (variable, value) in plot.variables.iter().zip(point) {
        match plot.flags {
            PlotFlags::Real => out.write_all(&byte_order.encode(value.re))?,
            PlotFlags::Complex => {
                // `raw_write()` writes `(re, 0.0)` for a vector it has flagged
                // real, exactly as the ASCII writer does.
                let imaginary = if variable.is_real { 0.0 } else { value.im };
                out.write_all(&byte_order.encode(value.re))?;
                out.write_all(&byte_order.encode(imaginary))?;
            }
        }
    }
    Ok(())
}
