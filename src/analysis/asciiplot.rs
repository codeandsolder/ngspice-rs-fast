//! Deterministic line-printer plots.
//!
//! C: `frontend/plotting/agraf.c::ft_agraf`.
//! Like C, the independent axis runs down the page and dependent values across
//! it. Pagination, date headers and terminal-dependent widths are deliberately
//! replaced by a fixed 60-column field; no graphical backend is involved.
use crate::analysis::Plot;
use crate::primitives::{SpiceError, SpiceResult, exact_nonnegative_integer};
use std::fmt::Write as _;

const SYMBOLS: &[u8] = b"+*=$%!0123456789";
const LAST_COLUMN: u8 = 60;

/// Quantize the normalized coordinate to the nearest valid plot column.
fn plot_column(value: f64, low: f64, span: f64) -> Option<usize> {
    let position = (((value - low) / span) * f64::from(LAST_COLUMN))
        .round()
        .clamp(0., f64::from(LAST_COLUMN));
    exact_nonnegative_integer(position)
}

/// Render selected vectors, using C's legend and collision characters.
/// Complex vectors are plotted by their real component (C `ft_agraf`).
///
/// # Errors
/// Invalid, empty or nonmonotonic data, nonfinite scale arithmetic.
pub fn render(plot: &Plot) -> SpiceResult<String> {
    let invalid = |message: &str| SpiceError::Unsupported {
        feature: format!(".plot: {message}"),
        location: None,
    };
    if plot.variables.len() < 2 || plot.points.is_empty() {
        return Err(invalid("needs a scale and at least one dependent vector"));
    }
    if plot
        .points
        .iter()
        .any(|row| row.len() != plot.variables.len() || row.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid("inconsistent or nonfinite plot values"));
    }
    if plot.points.windows(2).any(|p| p[1][0].re < p[0][0].re) {
        return Err(invalid("scale must be nondecreasing"));
    }
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for row in &plot.points {
        for value in &row[1..] {
            low = low.min(value.re);
            high = high.max(value.re);
        }
    }
    if low >= high {
        let delta = low.abs().max(1.) * 0.1;
        low -= delta;
        high += delta;
    }
    let span = high - low;
    if !span.is_finite() || span <= 0. {
        return Err(invalid("dependent range overflows"));
    }
    let mut text = format!("plot: {}\nLegend:  ", plot.plotname);
    for (i, variable) in plot.variables[1..].iter().enumerate() {
        let _ = write!(
            text,
            "{} = {}  ",
            char::from(*SYMBOLS.get(i).unwrap_or(&b'#')),
            variable.name
        );
    }
    let _ = writeln!(
        text,
        "\n{}     {low:.3e} .. {high:.3e}",
        plot.variables[0].name
    );
    for row in &plot.points {
        let mut field = [b' '; LAST_COLUMN as usize + 1];
        for i in (0..=usize::from(LAST_COLUMN)).step_by(15) {
            field[i] = b'.';
        }
        for (i, value) in row[1..].iter().enumerate() {
            let at =
                plot_column(value.re, low, span).ok_or_else(|| invalid("invalid plot column"))?;
            field[at] = if matches!(field[at], b' ' | b'.') {
                *SYMBOLS.get(i).unwrap_or(&b'#')
            } else {
                b'X'
            };
        }
        let _ = writeln!(
            text,
            "{:.3e} {:.3e} {}",
            row[0].re,
            row[1].re,
            String::from_utf8_lossy(&field)
        );
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::{plot_column, render};
    use crate::analysis::{Plot, PlotFlags, Variable};
    use crate::primitives::Complex;

    #[test]
    fn malformed_rows_are_rejected_before_scale_ordering() {
        let mut plot = Plot::new("tran1", "Transient", PlotFlags::Real);
        plot.push_variable(Variable::new("time", "time"));
        plot.push_variable(Variable::new("v(out)", "voltage"));
        plot.push_point(vec![Complex::real(0.), Complex::real(1.)])
            .unwrap();
        plot.points.push(Vec::new());
        let error = render(&plot).expect_err("inconsistent row");
        assert!(error.to_string().contains("inconsistent or nonfinite"));
    }

    #[test]
    fn quantization_keeps_the_full_inclusive_plot_range() {
        assert_eq!(plot_column(-1., -1., 2.), Some(0));
        assert_eq!(plot_column(0., -1., 2.), Some(30));
        assert_eq!(plot_column(1., -1., 2.), Some(60));
        assert_eq!(plot_column(-2., -1., 2.), Some(0));
        assert_eq!(plot_column(2., -1., 2.), Some(60));
        assert_eq!(plot_column(f64::NAN, -1., 2.), None);
    }
}
