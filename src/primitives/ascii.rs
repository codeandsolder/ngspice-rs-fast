//! Borrowed ASCII-only identity normalization.
//!
//! Netlist names, model/bin identifiers and plot vector names use ASCII
//! case-insensitivity; Unicode case mapping would change their identity.

use std::borrow::Cow;

/// Canonicalize ASCII uppercase letters, borrowing the input when unchanged.
///
/// This preserves all non-ASCII bytes and performs no allocation on the
/// already-canonical fast path.
#[must_use]
pub fn folded(name: &str) -> Cow<'_, str> {
    if name.bytes().any(|byte| byte.is_ascii_uppercase()) {
        Cow::Owned(name.to_ascii_lowercase())
    } else {
        Cow::Borrowed(name)
    }
}

#[cfg(test)]
mod tests {
    use super::folded;
    use std::borrow::Cow;

    #[test]
    fn lower_ascii_and_unicode_only_are_borrowed_without_an_allocation() {
        for input in ["", "v(out)", "nch.01", "naïve", "λδ", "ß", "ε.gate"] {
            let folded = folded(input);
            assert!(matches!(folded, Cow::Borrowed(_)), "{input}");
            assert!(std::ptr::eq(folded.as_ref(), input));
        }
    }

    #[test]
    fn uppercase_ascii_allocates_once_and_non_ascii_is_byte_preserved() {
        for (input, expected) in [
            ("V(OUT)", "v(out)"),
            ("NCH.01", "nch.01"),
            ("ÉLECTRO", "Électro"),
            ("ΣV", "Σv"),
            ("CAFÉ_GATE", "cafÉ_gate"),
        ] {
            let folded = folded(input);
            assert!(matches!(folded, Cow::Owned(_)), "{input}");
            assert_eq!(folded, expected);
        }
    }
}
