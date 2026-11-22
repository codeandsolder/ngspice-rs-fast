//! Tokenizing a logical card.
//!
//! The C front end has no separate tokenizer: `src/spicelib/parser/inp2*.c`
//! functions walk the card text with `INPgetTok()` / `INPgetNetTok()` from
//! `src/spicelib/parser/inpgtok.c` as they parse. The port separates the two
//! steps so that the parser can be written against a positioned token stream.
//!
//! Delimiters are whitespace, `(`, `)`, `,`, `=` and the quote characters. That
//! means a **numeric node name becomes a [`TokenKind::Number`] token**, and
//! `.model d d(is=1e-14)` becomes the same sequence of tokens a human would
//! expect — but the parser must accept a `Number` wherever a name is legal,
//! because `0` is the ground node. The [`Token::text`] field always holds the
//! original spelling, so the parser can rely on it when the classification is
//! merely a hint.

use std::fmt;

use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number};

use crate::netlist::source::budget::{SourceBudget, SourceLimits, copy_text};
use crate::netlist::source::{LogicalLine, SourceMap, SourcePosition};

/// What kind of token this is.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// An identifier, node name, model name or keyword.
    Word,
    /// A numeric literal. [`Token::text`] keeps the original spelling, which may
    /// carry a scale suffix (`1k`) and a trailing unit name (`5V`).
    Number(Real),
    /// A `{ ... }` numparam expression, kept verbatim. The tokenizer only
    /// matches braces; grammar-level consumers parse the text (see
    /// [`crate::netlist::expr`]).
    Expression(String),
    /// A quoted string, with its quotes removed and escapes resolved.
    Quoted(String),
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `,`
    Comma,
    /// `=`
    Equals,
}

impl TokenKind {
    /// A short human-readable name, for diagnostics and the CLI's token dump.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Word => "word",
            Self::Number(_) => "number",
            Self::Expression(_) => "expression",
            Self::Quoted(_) => "quoted",
            Self::LParen => "'('",
            Self::RParen => "')'",
            Self::Comma => "','",
            Self::Equals => "'='",
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One lexical token, with physical position and a joined-card byte offset.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The token's classification.
    pub kind: TokenKind,
    /// The exact source spelling.
    pub text: String,
    /// Physical source position of the first byte.
    pub location: SourceLoc,
    /// Zero-based byte offset in the joined card spelling.
    pub raw_offset: usize,
    source_map: SourceMap,
}

impl Token {
    fn new(
        kind: TokenKind,
        text: String,
        line: &LogicalLine,
        raw_offset: usize,
        location: SourceLoc,
    ) -> Self {
        Self {
            kind,
            text,
            location,
            raw_offset,
            source_map: line.source_map.clone(),
        }
    }

    pub(crate) fn position(&self) -> SourcePosition<'_> {
        self.source_map
            .token_position(&self.location, self.raw_offset)
    }

    /// The value, when this token is a number.
    #[must_use]
    pub const fn number(&self) -> Option<Real> {
        match self.kind {
            TokenKind::Number(value) => Some(value),
            _ => None,
        }
    }

    /// Compares the token's spelling to a keyword, case-insensitively.
    #[must_use]
    pub fn is_keyword(&self, keyword: &str) -> bool {
        self.text.eq_ignore_ascii_case(keyword)
    }

    /// True when the token is a word or a number, i.e. something that can be
    /// used as a name.
    #[must_use]
    pub const fn is_name_like(&self) -> bool {
        matches!(self.kind, TokenKind::Word | TokenKind::Number(_))
    }
}

/// Characters that end a bare word.
const DELIMITERS: &[u8] = b"(),={}\"'";

/// Splits a logical card into tokens.
///
/// # Errors
///
/// Returns [`SpiceError::Parse`] for an unterminated `{` expression, an
/// unmatched `}` or an unterminated quoted string.
pub fn tokenize(line: &LogicalLine) -> SpiceResult<Vec<Token>> {
    tokenize_bounded(line, &mut SourceBudget::new(SourceLimits::default()))
}

pub(super) fn tokenize_bounded(
    line: &LogicalLine,
    budget: &mut SourceBudget,
) -> SpiceResult<Vec<Token>> {
    budget.card_size(line.text.len(), &line.location)?;
    let text = line.text.as_str();
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        let location = line.location_at(index);
        budget.tokens(1, &location)?;
        budget.allocation(tokens.try_reserve(1), &location)?;
        let (kind, end) = scan_token(text, index, budget, &location)?;
        let spelling = copy_text(&text[index..end], budget, &location)?;
        tokens.push(Token::new(kind, spelling, line, index, location));
        index = end;
    }
    Ok(tokens)
}

fn scan_token(
    text: &str,
    start: usize,
    budget: &mut SourceBudget,
    location: &SourceLoc,
) -> SpiceResult<(TokenKind, usize)> {
    let bytes = text.as_bytes();
    let kind = match bytes[start] {
        b'(' => TokenKind::LParen,
        b')' => TokenKind::RParen,
        b',' => TokenKind::Comma,
        b'=' => TokenKind::Equals,
        b'{' => {
            let end = find_closing_brace(bytes, start).ok_or_else(|| {
                SpiceError::parse(location.clone(), "unterminated '{' expression on this card")
            })?;
            let inner = copy_text(text[start + 1..end].trim(), budget, location)?;
            return Ok((TokenKind::Expression(inner), end + 1));
        }
        b'}' => {
            return Err(SpiceError::parse(
                location.clone(),
                "unmatched '}' without an opening '{' on this card",
            ));
        }
        quote @ (b'"' | b'\'') => {
            return scan_quoted(text, start, char::from(quote), budget, location);
        }
        _ => {
            let mut end = start + 1;
            while end < bytes.len()
                && !bytes[end].is_ascii_whitespace()
                && !DELIMITERS.contains(&bytes[end])
            {
                end += 1;
            }
            let word = &text[start..end];
            let kind = parse_spice_number(word).map_or(TokenKind::Word, TokenKind::Number);
            return Ok((kind, end));
        }
    };
    Ok((kind, start + 1))
}

fn scan_quoted(
    text: &str,
    start: usize,
    quote: char,
    budget: &mut SourceBudget,
    location: &SourceLoc,
) -> SpiceResult<(TokenKind, usize)> {
    let content_start = start + 1;
    let mut characters = text[content_start..].char_indices();
    let mut value = String::new();
    while let Some((offset, character)) = characters.next() {
        let decoded = if character == '\\' {
            characters.next().map_or(character, |(_, escaped)| escaped)
        } else if character == quote {
            return Ok((
                TokenKind::Quoted(value),
                content_start + offset + character.len_utf8(),
            ));
        } else {
            character
        };
        budget.allocation(value.try_reserve(decoded.len_utf8()), location)?;
        value.push(decoded);
    }
    Err(SpiceError::parse(
        location.clone(),
        format!("unterminated {quote} quoted string on this card"),
    ))
}

/// Finds the `}` matching the `{` at `open`, accounting for nesting.
fn find_closing_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{TokenKind, tokenize};
    use crate::netlist::source::parse_deck_text;
    use std::path::Path;

    fn tokens(text: &str) -> Vec<super::Token> {
        let deck = parse_deck_text(Path::new("test.cir"), &format!("title\n{text}\n"));
        tokenize(&deck.lines[0]).expect("card tokenizes")
    }

    fn kinds(text: &str) -> Vec<TokenKind> {
        tokens(text).into_iter().map(|token| token.kind).collect()
    }

    #[test]
    fn splits_on_whitespace() {
        assert_eq!(
            kinds("r1 in out 1k"),
            vec![
                TokenKind::Word,
                TokenKind::Word,
                TokenKind::Word,
                TokenKind::Number(1000.0),
            ]
        );
    }

    #[test]
    fn numbers_keep_their_spelling() {
        let tokens = tokens("r1 in out 1kohm");
        assert_eq!(tokens[3].kind, TokenKind::Number(1000.0));
        assert_eq!(tokens[3].text, "1kohm");
    }

    #[test]
    fn numeric_node_names_become_number_tokens() {
        // `0` is the ground node; device letters and node names that look like
        // numbers must still be usable by the parser.
        assert_eq!(
            kinds("v1 1 0 dc 5"),
            vec![
                TokenKind::Word,
                TokenKind::Number(1.0),
                TokenKind::Number(0.0),
                TokenKind::Word,
                TokenKind::Number(5.0),
            ]
        );
    }

    #[test]
    fn part_numbers_stay_words() {
        // `2N2222` is not `2e-9` followed by junk, so it is not a number.
        assert_eq!(
            kinds("q1 c b 0 2N2222"),
            vec![
                TokenKind::Word,
                TokenKind::Word,
                TokenKind::Word,
                TokenKind::Number(0.0),
                TokenKind::Word,
            ]
        );
    }

    #[test]
    fn punctuation_is_split_out() {
        assert_eq!(
            kinds(".model d d(is=1e-14)"),
            vec![
                TokenKind::Word,
                TokenKind::Word,
                TokenKind::Word,
                TokenKind::LParen,
                TokenKind::Word,
                TokenKind::Equals,
                TokenKind::Number(1e-14),
                TokenKind::RParen,
            ]
        );
    }

    #[test]
    fn braced_expressions_are_one_token() {
        let tokens = tokens("r1 a b {r2*2}");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[3].kind, TokenKind::Expression("r2*2".to_owned()));
        assert_eq!(tokens[3].text, "{r2*2}");
    }

    #[test]
    fn nested_braces_are_matched() {
        let tokens = tokens("r1 a b {max(1,{x})}");
        assert_eq!(
            tokens[3].kind,
            TokenKind::Expression("max(1,{x})".to_owned())
        );
    }

    #[test]
    fn quoted_strings_are_unwrapped() {
        let tokens = tokens(r#"title "a b" 'c d'"#);
        assert_eq!(tokens[1].kind, TokenKind::Quoted("a b".to_owned()));
        assert_eq!(tokens[1].text, "\"a b\"");
        assert_eq!(tokens[2].kind, TokenKind::Quoted("c d".to_owned()));
    }

    #[test]
    fn escaped_quotes_survive() {
        let tokens = tokens(r#""a \" b""#);
        assert_eq!(tokens[0].kind, TokenKind::Quoted("a \" b".to_owned()));
    }

    #[test]
    fn quoted_unicode_escape_preserves_following_token() {
        let tokens = tokens(r#""α \β \" γ" next"#);
        assert_eq!(tokens[0].kind, TokenKind::Quoted("α β \" γ".to_owned()));
        assert_eq!(tokens[1].text, "next");
    }

    #[test]
    fn continued_quoted_tokens_keep_spelling_and_physical_positions() {
        let tokens = tokens("  .title \"α \\\\β\n* between\n   + γ\" next");
        assert_eq!(tokens[1].text, "\"α \\\\β γ\"");
        assert_eq!(tokens[1].kind, TokenKind::Quoted("α \\β γ".into()));
        assert_eq!(
            (tokens[1].location.line, tokens[1].location.column),
            (2, 10)
        );
        assert_eq!(
            (tokens[2].location.line, tokens[2].location.column),
            (4, 10)
        );
        let end = tokens[1]
            .position()
            .advance(tokens[1].text.len())
            .location();
        assert_eq!((end.line, end.column), (4, 9));
        assert_eq!(tokens[2].raw_offset, ".title \"α \\\\β γ\" ".len());
    }

    #[test]
    fn columns_are_one_based() {
        let tokens = tokens("r1 in out 1k");
        assert_eq!(tokens[0].location.column, 1);
        assert_eq!(tokens[1].location.column, 4);
        assert_eq!(tokens[3].location.column, 11);
        assert_eq!(tokens[0].location.line, 2);
    }

    #[test]
    fn unterminated_constructs_are_errors() {
        assert!(tokenize(&card("{r2*2")).is_err());
        assert!(tokenize(&card("\"oops")).is_err());
        let stray = tokenize(&card("r1 a 0 }")).unwrap_err().to_string();
        assert!(stray.contains(":2:8"), "{stray}");
        assert!(stray.contains("unmatched '}'"), "{stray}");
    }

    fn card(text: &str) -> crate::netlist::source::LogicalLine {
        let deck = parse_deck_text(Path::new("test.cir"), &format!("title\n{text}\n"));
        deck.lines[0].clone()
    }

    #[test]
    fn keyword_comparison_is_case_insensitive() {
        let tokens = tokens("R1 in out 1k");
        assert!(tokens[0].is_keyword("r1"));
        assert!(tokens[0].is_name_like());
        assert!(!tokens[3].is_keyword("r1"));
    }
}
