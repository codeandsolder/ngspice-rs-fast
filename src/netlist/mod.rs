//! The netlist front end: deck text in, [`ast::Netlist`] out.
//!
//! The front end is split so that it can be ported in stages:
//!
//! | Module | Job | State |
//! | --- | --- | --- |
//! | [`source`] | physical lines → logical cards: title, `+` continuations, comments | **ported** |
//! | [`token`] | logical card → token stream | **ported** |
//! | [`card`] | first token → [`card::CardKind`] classification | **ported** |
//! | [`expr`] | unevaluated parameter-expression syntax tree (bounded numparam subset) | **#14** |
//! | [`bexpr`] | behavioural-source (B, E/G VALUE/TABLE/POLY) expression syntax tree | **#79** |
//! | [`behavioural`] | E/G/F/H nonlinear-form lowering and B-expression parameter/`.func` resolution | **#79** |
//! | [`eval`] | bounded `.param` scope resolution and expression evaluation | **#15** |
//! | [`elaborate`] | literalized netlist copy with evaluated numeric sites | **#15** |
//! | [`ast`] | the semantic netlist model | bounded devices/models, ordered subcircuit scopes and source provenance |
//! | [`writer`], [`semantic`] | normalized deck serialization and location-free comparison | **#20** |
//! | [`dump`], [`snapshot`] | versioned token/AST text dumps and snapshot generation | **#21** |
//! | [`parser`] | winnow token stream → [`ast::Netlist`] and the deck's `.save`/`.print`/`.measure` cards | **M1a/M1b + scoped subcircuits/X, bounded source resolution and output/measurement cards; no flattening** |
//!
//! The C equivalent is spread over `src/frontend/inp.c`,
//! `src/frontend/inpcom.c` and `src/spicelib/parser/`. Deck dispatch is in
//! `inppas2.c`/`inp2dot.c` and per-device functions (`inp2r.c`, `inp2c.c`, …).
//! `.save`/`.print` output cards are collected by `ft_dotsaves()`/
//! `ft_savedotargs()` (`src/frontend/dotcards.c`; see
//! `docs/port/OUTPUT_SELECTION.md`) and `.measure` cards by `inp_spsource()`
//! (`src/frontend/inp.c`; see `docs/port/MEASURE.md`).
//! `src/frontend/parse-bison.y` is a separate expression grammar, not a deck
//! grammar; `.param` evaluation lives in `src/frontend/numparam/`.

pub mod ast;
pub mod behavioural;
pub mod bexpr;
pub mod card;
pub mod dump;
pub mod elaborate;
pub mod eval;
pub mod expr;
pub mod parser;
pub mod prepared_initialization;
pub mod semantic;
pub mod snapshot;
pub mod source;
pub mod token;
pub mod writer;

pub use card::{CardKind, DEVICE_DESIGNATORS, DotCommand, RawCard};
pub use expr::{Expr, ExprKind, ParameterExpression, SourceSpan};
pub use parser::{
    ParsedDeck, Parser, SourceLimits, classify_deck, load_classified, parse_parameter_expression,
};
pub use semantic::{semantic_diff, semantic_eq};
pub use source::admission::SourceAdmission;
pub use source::{Deck, LogicalLine, PhysicalLine, load};
pub use token::{Token, TokenKind, tokenize};
pub use writer::{WriteLimits, write_netlist, write_netlist_to};

/// The C reference for the whole front end, used in `NotYetPorted` errors.
pub const C_REFERENCE: &str =
    "src/frontend/inp.c, src/frontend/inpcom.c, src/spicelib/parser/inp*.c";

/// Explicit physical network setters shared by the normalized native grammar
/// and writer. No source-dialect defaults are selected here.
pub(crate) fn network_parameter(designator: char, name: &str) -> bool {
    if matches!(designator, 'c' | 'l' | 'v')
        && matches!(
            name,
            "pn_control"
                | "pn_report_dc"
                | "pn_report_ac"
                | "pn_report_tran"
                | "pn_report_dc_g"
                | "pn_report_ac_g"
                | "pn_report_tran_g"
        )
    {
        return true;
    }
    match designator {
        'c' | 'l' => matches!(
            name,
            "pn_report_dc_rparallel_factor"
                | "pn_report_ac_rparallel_factor"
                | "pn_report_tran_rparallel_factor"
                | "pn_rseries"
                | "pn_lseries"
                | "pn_rseries_shunt"
                | "pn_rstorage_shunt"
                | "pn_rparallel"
                | "pn_rparallel_noise_abs"
                | "pn_noiseless"
                | "pn_cparallel"
                | "pn_bias_op"
                | "pn_bias_dc"
                | "pn_bias_ac"
                | "pn_bias_tran"
                | "pn_terminal_bias_op"
                | "pn_terminal_bias_dc"
                | "pn_terminal_bias_ac"
                | "pn_terminal_bias_tran"
                | "pn_exact_ic"
                | "pn_parallel_ic"
                | "pn_gseries_dc"
                | "pn_gseries_ac"
                | "pn_gseries_tran"
                | "pn_gseries_dc_gmin"
                | "pn_gseries_ac_gmin"
                | "pn_gseries_tran_gmin"
                | "pn_gseries_noise"
                | "pn_gstorage_dc"
                | "pn_gstorage_ac"
                | "pn_gstorage_tran"
                | "pn_gstorage_dc_gmin"
                | "pn_gstorage_ac_gmin"
                | "pn_gstorage_tran_gmin"
                | "pn_gstorage_noise"
                | "pn_gterminal_dc"
                | "pn_gterminal_ac"
                | "pn_gterminal_tran"
                | "pn_gterminal_dc_gmin"
                | "pn_gterminal_ac_gmin"
                | "pn_gterminal_tran_gmin"
                | "pn_gterminal_noise"
        ),
        'v' => matches!(
            name,
            "pn_rseries" | "pn_cparallel" | "pn_cseries_shunt" | "pn_cconditional" | "pn_temp"
        ),
        _ => false,
    }
}
