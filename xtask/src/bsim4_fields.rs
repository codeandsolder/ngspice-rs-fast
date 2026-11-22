//! Extract BSIM4 scalar field declarations.
//!
//! Extract the numeric field declarations of BSIM4's three C structs
//! (`sBSIM4model`, `bsim4SizeDependParam`, `sBSIM4instance`) as Rust struct
//! definitions: field names and scalar types only. The BSIM4 port keeps C's
//! field names (without the `BSIM4` prefix) so that `b4set.c`, `b4temp.c` and
//! `b4ld.c` can be followed statement by statement. No formula, default or
//! comment text is copied; pointers, arrays and matrix bindings are skipped.
use std::collections::BTreeSet;
use std::fmt::Write as _;
use syn::{
    BinOp, Expr, Member,
    visit::{self, Visit},
};

/// One scalar field of a C struct.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Field {
    /// The C member name, e.g. `BSIM4vth0`.
    c_name: String,
    /// `f64`, `i32`, `bool` (a `:1` given bit) or `String` (`char *`).
    kind: &'static str,
}

/// Members of the struct whose body starts at `opening`, in declaration order.
fn members(header: &str, opening: &str, closing: &str) -> Result<Vec<Field>, String> {
    let start = header
        .find(opening)
        .ok_or_else(|| format!("bsim4def.h: missing `{opening}`"))?;
    let body = &header[start + opening.len()..];
    let end = body
        .find(closing)
        .ok_or_else(|| format!("bsim4def.h: unterminated `{opening}`"))?;
    let mut fields = Vec::new();
    for line in body[..end].lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("struct ") {
            continue;
        }
        let words: Vec<&str> = line.trim_end_matches(';').split_whitespace().collect();
        let field = match words.as_slice() {
            ["double", name] if !name.contains(['*', '[']) => Field {
                c_name: (*name).into(),
                kind: "f64",
            },
            ["int", name] | ["const", "int", name] if !name.contains(['*', '[']) => Field {
                c_name: (*name).into(),
                kind: "i32",
            },
            ["unsigned", name, ":1"] | ["unsigned", name, ":", "1"] => Field {
                c_name: (*name).into(),
                kind: "bool",
            },
            ["char", name] if name.starts_with('*') && !name[1..].contains(['*', '[']) => Field {
                c_name: name[1..].into(),
                kind: "String",
            },
            // Pointers, matrix bindings, noise arrays and OpenMP/KLU blocks.
            ["double" | "BindElement" | "char", ..] => continue,
            _ => return Err(format!("bsim4def.h: unrecognised member `{line}`")),
        };
        if !field
            .c_name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(format!("bsim4def.h: unexpected member name `{line}`"));
        }
        fields.push(field);
    }
    Ok(fields)
}

/// The Rust spelling of a C member: the `BSIM4` prefix removed, a raw
/// identifier for Rust keywords.
fn rust_name(c_name: &str) -> String {
    let name = c_name.strip_prefix("BSIM4").unwrap_or(c_name);
    if matches!(
        name,
        "type" | "pub" | "mod" | "ref" | "in" | "fn" | "use" | "as"
    ) {
        format!("r#{name}")
    } else {
        name.into()
    }
}

fn emit_struct(
    out: &mut String,
    name: &str,
    c_struct: &str,
    fields: &[Field],
) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for field in fields {
        if !seen.insert(rust_name(&field.c_name)) {
            return Err(format!("{c_struct}: duplicate member {}", field.c_name));
        }
    }
    writeln!(
        out,
        "\n/// `{c_struct}`'s scalar members ({} of them), in declaration order.\n\
         #[derive(Clone, Default)]\npub(crate) struct {name} {{",
        fields.len()
    )
    .map_err(|e| e.to_string())?;
    for field in fields {
        writeln!(
            out,
            "    pub(crate) {}: {},",
            rust_name(&field.c_name),
            field.kind
        )
        .map_err(|e| e.to_string())?;
    }
    out.push_str("}\n");
    writeln!(out, "\n#[rustfmt::skip]\nimpl {name} {{").map_err(|e| e.to_string())?;
    for (kind, accessor) in [("f64", "real"), ("i32", "int"), ("bool", "flag")] {
        if !fields.iter().any(|f| f.kind == kind) {
            continue;
        }
        writeln!(
            out,
            "    /// The `{kind}` member named `c_name` in C, if there is one.\n    \
             #[allow(dead_code, reason = \"uniform generated schema accessors may be unused by a particular device path\")]\n    pub(crate) fn {accessor}_mut(&mut self, c_name: &str) -> Option<&mut {kind}> {{\n        \
             Some(match c_name {{"
        )
        .map_err(|e| e.to_string())?;
        for field in fields.iter().filter(|f| f.kind == kind) {
            writeln!(
                out,
                "            \"{}\" => &mut self.{},",
                field.c_name,
                rust_name(&field.c_name)
            )
            .map_err(|e| e.to_string())?;
        }
        out.push_str("            _ => return None,\n        })\n    }\n");
    }
    // Positional access over the scalar (non-string) members, for records
    // that hold the whole struct as numbers (`.sens`).
    let scalars: Vec<&Field> = fields.iter().filter(|f| f.kind != "String").collect();
    out.push_str("    /// C member names of the scalar members, in declaration order.\n    #[allow(dead_code, reason = \"uniform generated schema accessors may be unused by a particular device path\")]\n    pub(crate) const MEMBERS: &[&str] = &[\n");
    for field in &scalars {
        writeln!(out, "        \"{}\",", field.c_name).map_err(|e| e.to_string())?;
    }
    out.push_str("    ];\n\n");
    out.push_str("    /// The scalar members as numbers, in [`Self::MEMBERS`] order (`i32` exactly,\n    /// a given bit as 0 or 1).\n    #[allow(dead_code, reason = \"uniform generated schema accessors may be unused by a particular device path\")]\n    pub(crate) fn to_values(&self) -> Vec<f64> {\n        vec![\n");
    for field in &scalars {
        let name = rust_name(&field.c_name);
        let expression = match field.kind {
            "f64" => format!("self.{name}"),
            "i32" => format!("f64::from(self.{name})"),
            _ => format!("f64::from(u8::from(self.{name}))"),
        };
        writeln!(out, "            {expression},").map_err(|e| e.to_string())?;
    }
    out.push_str("        ]\n    }\n\n");
    out.push_str("    /// Stores numbers produced by [`Self::to_values`]; `values` must have\n    /// `Self::MEMBERS.len()` entries.\n    #[allow(dead_code, clippy::cast_possible_truncation, reason = \"generated numeric snapshots round-trip exact i32 fields and may be unused for some record types\")]\n    pub(crate) const fn set_values(&mut self, values: &[f64]) {\n");
    for (index, field) in scalars.iter().enumerate() {
        let name = rust_name(&field.c_name);
        let expression = match field.kind {
            "f64" => format!("values[{index}]"),
            "i32" => format!("values[{index}] as i32"),
            _ => format!("values[{index}] != 0.0"),
        };
        writeln!(out, "        self.{name} = {expression};").map_err(|e| e.to_string())?;
    }
    out.push_str("    }\n");
    out.push_str("}\n");
    Ok(())
}

/// `(C parameter ID, member)` for every `case` of an ask routine that reads
/// one real member (`value->rValue = <prefix>X;`), in source order.
fn real_asks(source: &str, prefix: &str) -> Result<Vec<(String, String)>, String> {
    let source = crate::ifparm::strip_comments(source)?;
    let mut asks: Vec<(String, String)> = Vec::new();
    let mut pending: Vec<&str> = Vec::new();
    // The asks recorded for the last case group, dropped again when C goes
    // on to scale the value (`value->rValue *= ...`).
    let mut last = 0..0;
    for line in source.lines() {
        let line = line.trim().trim_start_matches('{').trim();
        if line.starts_with("value->rValue *=") {
            asks.drain(last.clone());
            last = 0..0;
            continue;
        }
        if let Some(rest) = line.strip_prefix("case ") {
            let id = rest.trim_end_matches(':').trim();
            if !id.starts_with("BSIM4_") {
                return Err(format!("ask: unexpected case `{line}`"));
            }
            pending.push(id);
        } else if let Some(rest) = line.strip_prefix("value->rValue = ") {
            if let Some(member) = rest
                .strip_suffix(';')
                .map(str::trim)
                .and_then(|r| r.strip_prefix(prefix))
                .filter(|m| m.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
            {
                let start = asks.len();
                for &id in &pending {
                    asks.push((id.into(), member.into()));
                }
                last = start..asks.len();
            }
            pending.clear();
        } else if line.starts_with("value->") || line.starts_with("return") {
            pending.clear();
        }
    }
    Ok(asks)
}

/// The generated `asks.rs` text: the real asks of `b4mask.c` and `b4ask.c`
/// that read a member directly (`sens_getp()` succeeds only for these).
pub(crate) fn asks(mask: &str, ask: &str, revision: &str) -> Result<String, String> {
    let model = real_asks(mask, "model->")?;
    let instance = real_asks(ask, "here->")?;
    if model.len() != 869 || instance.len() != 40 {
        return Err(format!(
            "b4mask.c/b4ask.c: {} model and {} instance plain real asks; review them",
            model.len(),
            instance.len()
        ));
    }
    let mut out = format!(
        "// Generated ask facts (C parameter IDs and the member each real ask\n\
         // reads), not C implementation code. Source: ngspice {revision},\n\
         // bsim4/{{b4mask.c,b4ask.c}}.\n\
         // Regenerate: cargo xtask bsim4-schema --source /path/to/ngspice\n\n\
         /// `b4mask.c`: `(model parameter ID, member)` asks that read one real member.\n\
         #[rustfmt::skip]\npub(super) const MODEL: &[(&str, &str)] = &[\n"
    );
    for (id, member) in &model {
        writeln!(out, "    (\"{id}\", \"{member}\"),").map_err(|e| e.to_string())?;
    }
    out.push_str("];\n\n/// `b4ask.c`: `(instance parameter ID, member)` asks that read one real member.\n#[rustfmt::skip]\npub(super) const INSTANCE: &[(&str, &str)] = &[\n");
    for (id, member) in &instance {
        writeln!(out, "    (\"{id}\", \"{member}\"),").map_err(|e| e.to_string())?;
    }
    out.push_str("];\n");
    Ok(out)
}

/// Direct scalar writes in the statement-by-statement translation. The
/// routines use a named instance argument, with no mutable field aliases
/// or writes hidden in macros. Setup, temperature, validation and load
/// are visited separately, including their conditional branches.
struct InstanceWrites<'a> {
    owner: &'a str,
    fields: BTreeSet<String>,
}

impl InstanceWrites<'_> {
    fn record(&mut self, expression: &Expr) {
        match expression {
            Expr::Field(field) => {
                if let (Expr::Path(base), Member::Named(member)) = (&*field.base, &field.member)
                    && base.path.is_ident(self.owner)
                {
                    self.fields.insert(member.to_string());
                }
            }
            Expr::Paren(paren) => self.record(&paren.expr),
            Expr::Tuple(tuple) => {
                for element in &tuple.elems {
                    self.record(element);
                }
            }
            _ => {}
        }
    }
}

impl<'ast> Visit<'ast> for InstanceWrites<'_> {
    fn visit_expr_assign(&mut self, expression: &'ast syn::ExprAssign) {
        self.record(&expression.left);
        visit::visit_expr_assign(self, expression);
    }

    fn visit_expr_binary(&mut self, expression: &'ast syn::ExprBinary) {
        if matches!(
            expression.op,
            BinOp::AddAssign(_)
                | BinOp::SubAssign(_)
                | BinOp::MulAssign(_)
                | BinOp::DivAssign(_)
                | BinOp::RemAssign(_)
                | BinOp::BitXorAssign(_)
                | BinOp::BitAndAssign(_)
                | BinOp::BitOrAssign(_)
                | BinOp::ShlAssign(_)
                | BinOp::ShrAssign(_)
        ) {
            self.record(&expression.left);
        }
        visit::visit_expr_binary(self, expression);
    }

    fn visit_expr_reference(&mut self, expression: &'ast syn::ExprReference) {
        if expression.mutability.is_some() {
            self.record(&expression.expr);
        }
        visit::visit_expr_reference(self, expression);
    }
}

fn instance_writes(source: &str, owner: &str) -> Result<BTreeSet<String>, String> {
    let syntax = syn::parse_file(source).map_err(|error| error.to_string())?;
    let mut visitor = InstanceWrites {
        owner,
        fields: BTreeSet::new(),
    };
    // Only the production routines participate in the projection; nested
    // test/verification modules construct their own unrelated instances.
    for item in &syntax.items {
        if let syn::Item::Fn(function) = item {
            visitor.visit_item_fn(function);
        }
    }
    Ok(visitor.fields)
}

fn emit_instance_ownership(out: &mut String, fields: &[Field]) -> Result<(), String> {
    let phases = [
        (
            "SETUP_WRITES",
            instance_writes(include_str!("../../src/devices/bsim4/set.rs"), "h")?,
        ),
        (
            "TEMPERATURE_WRITES",
            instance_writes(include_str!("../../src/devices/bsim4/temp.rs"), "h")?,
        ),
        (
            "CHECK_WRITES",
            instance_writes(include_str!("../../src/devices/bsim4/check.rs"), "h")?,
        ),
        (
            "LOAD_WRITES",
            instance_writes(include_str!("../../src/devices/bsim4/load.rs"), "here")?,
        ),
    ];
    let known: BTreeSet<_> = fields
        .iter()
        .map(|field| rust_name(&field.c_name))
        .collect();
    for (stage, writes) in &phases {
        if let Some(unknown) = writes.difference(&known).next() {
            return Err(format!("{stage}: unknown instance member {unknown}"));
        }
    }
    // b4temp.c assigns both overlap capacitances unconditionally. A new
    // overlap needs review of whether preparation actually resets it on
    // every path; a conditional write is not enough to discard live state.
    let overlap: BTreeSet<_> = phases[3]
        .1
        .iter()
        .filter(|name| phases[..3].iter().any(|(_, writes)| writes.contains(*name)))
        .map(String::as_str)
        .collect();
    if overlap != BTreeSet::from(["cgdo", "cgso"]) {
        return Err(format!(
            "BSIM4 load/preparation overlap changed: {overlap:?}; review reset ownership"
        ));
    }
    let persistent: BTreeSet<_> = phases[3]
        .1
        .iter()
        .filter(|name| !phases[..3].iter().any(|(_, writes)| writes.contains(*name)))
        .cloned()
        .collect();
    out.push_str("\n#[rustfmt::skip]\nimpl Instance {\n");
    for (stage, writes) in &phases {
        writeln!(out, "    /// Scalar write set derived from the translated routine.\n    #[cfg(test)]\n    pub(crate) const {stage}: &[&str] = &[").map_err(|e| e.to_string())?;
        for field in fields
            .iter()
            .filter(|field| writes.contains(&rust_name(&field.c_name)))
        {
            writeln!(out, "        \"{}\",", field.c_name).map_err(|e| e.to_string())?;
        }
        out.push_str("    ];\n");
    }
    out.push_str("    /// Rebase persistent load state onto a fresh setup/temperature snapshot.\n    /// Fields also written by preparation belong to that fresh snapshot.\n    pub(crate) const fn with_previous_load_state(&self, previous: &Self) -> Self {\n        Self {\n");
    for field in fields {
        let name = rust_name(&field.c_name);
        if field.kind == "String" {
            return Err(format!(
                "instance member {} needs explicit ownership review",
                field.c_name
            ));
        }
        let owner = if persistent.contains(&name) {
            "previous"
        } else {
            "self"
        };
        writeln!(out, "            {name}: {owner}.{name},").map_err(|e| e.to_string())?;
    }
    out.push_str("        }\n    }\n}\n");
    Ok(())
}

/// The generated `fields.rs` text.
pub(crate) fn fields(header: &str, revision: &str) -> Result<String, String> {
    let header = crate::ifparm::strip_comments(header)?;
    let instance = members(
        &header,
        "typedef struct sBSIM4instance\n{",
        "} BSIM4instance",
    )?;
    let size = members(&header, "struct bsim4SizeDependParam\n{", "};")?;
    let model = members(&header, "typedef struct sBSIM4model\n{", "} BSIM4model")?;
    for (label, fields, expected) in [
        ("instance", &instance, 370),
        ("size", &size, 224),
        ("model", &model, 1824),
    ] {
        if fields.len() != expected {
            return Err(format!(
                "bsim4def.h: {} {label} members, reviewed {expected}; review the header",
                fields.len()
            ));
        }
    }
    let mut out = format!(
        "// @generated by cargo xtask bsim4-schema; do not edit directly.\n\
         // Generated field facts (C member names and scalar types), not C\n\
         // implementation code. Source: ngspice {revision}, bsim4/bsim4def.h.\n\
         // Regenerate: cargo xtask bsim4-schema --source /path/to/ngspice\n\
         "
    );
    // Only the generated schema may retain these C-field spelling and
    // monolithic accessor patterns. The source policy verifies provenance.
    out.push_str(
        "#![allow(\n    non_snake_case,\n    clippy::struct_excessive_bools,\n    clippy::used_underscore_binding,\n    clippy::too_many_lines,\n    clippy::large_stack_arrays,\n    clippy::redundant_pub_crate,\n    reason = \"generated C schema preserves member spelling, layout and uniform positional accessors\"\n)]\n",
    );
    emit_struct(&mut out, "Model", "struct sBSIM4model", &model)?;
    emit_struct(&mut out, "SizeParam", "struct bsim4SizeDependParam", &size)?;
    emit_struct(&mut out, "Instance", "struct sBSIM4instance", &instance)?;
    emit_instance_ownership(&mut out, &instance)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{instance_writes, real_asks};

    #[test]
    fn ownership_includes_conditional_and_compound_writes_but_not_reads() {
        let source = r#"fn f() { if h.read == 1 { h.a = 0; h.b += 1; f(&mut h.c); } /* h.fake = 0; */ let _ = "h.fake = 0"; (h.d, h.e) = (1, 2); }"#;
        let writes = instance_writes(source, "h").unwrap();
        assert_eq!(
            writes.into_iter().collect::<Vec<_>>(),
            ["a", "b", "c", "d", "e"]
        );
    }

    #[test]
    fn real_asks_keeps_case_aliases_and_discards_scaled_values() {
        let source = "
            case BSIM4_A:
            case BSIM4_ALIAS:
                value->rValue = model->BSIM4a;
                return OK;
            case BSIM4_SCALED:
                value->rValue = model->BSIM4b;
                value->rValue *= 2;
                return OK;
            case BSIM4_C:
                value->rValue = model->BSIM4c;
                return OK;
        ";
        assert_eq!(
            real_asks(source, "model->").unwrap(),
            [
                ("BSIM4_A".to_owned(), "BSIM4a".to_owned()),
                ("BSIM4_ALIAS".to_owned(), "BSIM4a".to_owned()),
                ("BSIM4_C".to_owned(), "BSIM4c".to_owned()),
            ]
        );
    }
}
