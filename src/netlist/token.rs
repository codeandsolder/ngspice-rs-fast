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

use std::borrow::Cow;
use std::fmt;

use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number};

use crate::netlist::source::budget::{SourceBudget, SourceLimits, copy_text};
use crate::netlist::source::{LogicalLine, SourceMap, SourcePosition};

/// What kind of token this is.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind<S = String> {
    /// An identifier, node name, model name or keyword.
    Word,
    /// A numeric literal. [`Token::text`] keeps the original spelling, which may
    /// carry a scale suffix (`1k`) and a trailing unit name (`5V`).
    Number(Real),
    /// A `{ ... }` numparam expression, kept verbatim. The tokenizer only
    /// matches braces; grammar-level consumers parse the text (see
    /// [`crate::netlist::expr`]).
    Expression(S),
    /// A quoted string, with its quotes removed and escapes resolved.
    Quoted(S),
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `,`
    Comma,
    /// `=`
    Equals,
}

impl<S> TokenKind<S> {
    /// A short human-readable name, for diagnostics and the CLI's token dump.
    #[must_use = "the returned name identifies the token category in diagnostics"]
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

impl<S> fmt::Display for TokenKind<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl TokenKind<Cow<'_, str>> {
    /// Own the payload only when storing a token in the parsed deck. Source
    /// allocations remain charged to the parser's bounded source budget.
    fn into_owned(self, budget: &mut SourceBudget, location: &SourceLoc) -> SpiceResult<TokenKind> {
        let owned = |value: Cow<'_, str>, budget: &mut SourceBudget| -> SpiceResult<String> {
            match value {
                Cow::Borrowed(text) => copy_text(text, budget, location),
                Cow::Owned(text) => Ok(text),
            }
        };
        Ok(match self {
            Self::Word => TokenKind::Word,
            Self::Number(number) => TokenKind::Number(number),
            Self::Expression(value) => TokenKind::Expression(owned(value, budget)?),
            Self::Quoted(value) => TokenKind::Quoted(owned(value, budget)?),
            Self::LParen => TokenKind::LParen,
            Self::RParen => TokenKind::RParen,
            Self::Comma => TokenKind::Comma,
            Self::Equals => TokenKind::Equals,
        })
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
    #[must_use = "the optional value distinguishes parsed numeric tokens from other kinds"]
    pub const fn number(&self) -> Option<Real> {
        match self.kind {
            TokenKind::Number(value) => Some(value),
            _ => None,
        }
    }

    /// Compares the token's spelling to a keyword, case-insensitively.
    #[must_use = "the predicate determines whether the token matches the requested keyword"]
    pub fn is_keyword(&self, keyword: &str) -> bool {
        self.text.eq_ignore_ascii_case(keyword)
    }

    /// True when the token is a word or a number, i.e. something that can be
    /// used as a name.
    #[must_use = "the predicate determines whether the token can serve as a name"]
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
    let code_card = text.trim_start().starts_with(['a', 'A']);
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
        let (kind, end) = scan_token(text, index, budget, &location, code_card)?;
        let spelling = copy_text(&text[index..end], budget, &location)?;
        let kind = kind.into_owned(budget, &location)?;
        tokens.push(Token::new(kind, spelling, line, index, location));
        index = end;
    }
    Ok(tokens)
}

/// A borrowed token spelling and its production lexical classification.
/// This keeps scalar validation and whole-card tokenization on one scanner.
pub(super) struct Lexeme<'a> {
    pub(super) kind: TokenKind<Cow<'a, str>>,
    pub(super) text: &'a str,
}

/// Scan the next token without allocating a `LogicalLine` or token vector.
/// Textual tokens retain the same decoding and diagnostics as deck parsing.
pub(super) fn scan_next<'a>(
    text: &'a str,
    cursor: &mut usize,
    location: impl FnOnce(usize) -> SourceLoc,
) -> SpiceResult<Option<Lexeme<'a>>> {
    let bytes = text.as_bytes();
    while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
        *cursor += 1;
    }
    let start = *cursor;
    if start == bytes.len() {
        return Ok(None);
    }
    let position = location(start);
    let mut budget = SourceBudget::new(SourceLimits::default());
    budget.card_size(text.len(), &position)?;
    budget.tokens(1, &position)?;
    let (kind, end) = scan_token(text, start, &mut budget, &position, false)?;
    *cursor = end;
    Ok(Some(Lexeme {
        kind,
        text: &text[start..end],
    }))
}

fn scan_token<'a>(
    text: &'a str,
    start: usize,
    budget: &mut SourceBudget,
    location: &SourceLoc,
    code_card: bool,
) -> SpiceResult<(TokenKind<Cow<'a, str>>, usize)> {
    let bytes = text.as_bytes();
    let kind = match bytes[start] {
        b'[' | b']' | b'~' if code_card => TokenKind::Word,
        b'(' => TokenKind::LParen,
        b')' => TokenKind::RParen,
        b',' => TokenKind::Comma,
        b'=' => TokenKind::Equals,
        b'{' => {
            let end = find_closing_brace(bytes, start).ok_or_else(|| {
                SpiceError::parse(location.clone(), "unterminated '{' expression on this card")
            })?;
            let inner = text[start + 1..end].trim();
            return Ok((TokenKind::Expression(Cow::Borrowed(inner)), end + 1));
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
                && !(code_card && b"[]~".contains(&bytes[end]))
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

fn scan_quoted<'a>(
    text: &'a str,
    start: usize,
    quote: char,
    budget: &mut SourceBudget,
    location: &SourceLoc,
) -> SpiceResult<(TokenKind<Cow<'a, str>>, usize)> {
    let content_start = start + 1;
    let source = &text[content_start..];
    let mut characters = source.char_indices();
    let mut decoded: Option<String> = None;
    while let Some((offset, character)) = characters.next() {
        if character == quote {
            let value = decoded.map_or_else(|| Cow::Borrowed(&source[..offset]), Cow::Owned);
            return Ok((
                TokenKind::Quoted(value),
                content_start + offset + character.len_utf8(),
            ));
        }
        let character = if character == '\\' {
            characters.next().map_or(character, |(_, escaped)| escaped)
        } else {
            character
        };
        if decoded.is_none() && source.as_bytes()[offset] == b'\\' {
            let mut value = String::new();
            let capacity = offset.checked_add(character.len_utf8()).ok_or_else(|| {
                SpiceError::parse(location.clone(), "quoted token allocation size overflow")
            })?;
            budget.allocation(value.try_reserve(capacity), location)?;
            value.push_str(&source[..offset]);
            value.push(character);
            decoded = Some(value);
        } else if let Some(value) = &mut decoded {
            budget.allocation(value.try_reserve(character.len_utf8()), location)?;
            value.push(character);
        }
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
    use super::{TokenKind, scan_next, tokenize};
    use crate::netlist::source::parse_deck_text;
    use std::path::Path;

    fn tokens(text: &str) -> Vec<super::Token> {
        let deck = parse_deck_text(Path::new("test.cir"), &format!("title\n{text}\n"));
        tokenize(&deck.lines[0]).expect("card tokenizes")
    }

    #[test]
    fn borrowed_scanner_and_bounded_deck_tokenizer_share_exact_lexemes() {
        use crate::netlist::source::parse_deck_text;
        use std::path::Path;
        for text in [
            "r1 a 0 1k tc1={foo + 2}",
            "v1 a 0 pwl(0,1,2,3) ac 5 90",
            r#"x1 a b "a quoted model" p='a + b'"#,
            r#".include "α \" β.sp""#,
            ".param x={max(1,{a})} y=-2.0e-12",
        ] {
            let deck = parse_deck_text(Path::new("lex.cir"), &format!("t\n{text}\n"));
            let line = &deck.lines[0];
            let tokens = tokenize(line).unwrap();
            let mut cursor = 0;
            for token in tokens {
                let borrowed = scan_next(&line.text, &mut cursor, |_| token.location.clone())
                    .unwrap()
                    .unwrap();
                assert_eq!(borrowed.text, token.text);
                assert_eq!(borrowed.kind.name(), token.kind.name());
                match (borrowed.kind, token.kind) {
                    (TokenKind::Number(left), TokenKind::Number(right)) => {
                        assert_eq!(left.to_bits(), right.to_bits())
                    }
                    (TokenKind::Quoted(left), TokenKind::Quoted(right))
                    | (TokenKind::Expression(left), TokenKind::Expression(right)) => {
                        assert_eq!(left.as_ref(), right)
                    }
                    _ => {}
                }
            }
            assert!(
                scan_next(&line.text, &mut cursor, |_| line.location.clone())
                    .unwrap()
                    .is_none()
            );
        }
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
