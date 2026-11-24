// Development-only two-stage source generator for the circuit-LTO experiment.
// Run from the repository root. The first stage adds model exporters and a
// generic stamp dispatch; the second stage is emitted by the simulator itself.
import fs from "node:fs";
const edit = (path, fn) => {
    const old = fs.readFileSync(path, "utf8");
    const next = fn(old);
    if (next === old) throw new Error("no change: " + path);
    fs.writeFileSync(path, next);
};
const fields = fs.readFileSync("src/devices/bsim4/fields.rs", "utf8");
function emitter(type, obj) {
    const body = fields.match(new RegExp("struct " + type + " \\{([\\s\\S]*?)\\n\\}"))[1];
    return body.split("\n").map(line => line.match(/pub\(crate\) (\S+): (\w+),/)).filter(Boolean).map(([,name,ty]) => {
        if (ty === "String") return 'writeln!(out, "' + name + ': String::new(),")?;';
        if (ty === "f64") return 'writeln!(out, "' + name + ': f64::from_bits(0x{:016x}),", ' + obj + '.' + name + '.to_bits())?;';
        return 'writeln!(out, "' + name + ': {},", ' + obj + '.' + name + ')?;';
    }).join("\n");
}
edit("src/devices/bsim4/mod.rs", s => s.replace("mod device;", "mod device;\npub(super) use device::Bsim4;"));
edit("src/devices/bsim4/device.rs", s => {
    const start = s.indexOf("    fn stamp(&self, context: &mut StampContext<'_>)");
    const end = s.indexOf("\n    /// `BSIM4convTest`", start);
    const original = s.slice(start,end);
    const body = original.slice(original.indexOf("\n")).replace("load::load(&prepared.model, h, &prepared.size, &mut host)?;", "let (model, size) = baked.unwrap_or((&prepared.model, &prepared.size));\n        load::load(model, h, size, &mut host)?;");
    const helper = `
impl Bsim4 {
    #[inline(always)]
    pub(in crate::devices) fn stamp_with_model(&self, context: &mut StampContext<'_>, baked: Option<(&Model, &SizeParam)>) -> SpiceResult<()> {${body}
    pub(in crate::devices) fn emit_model(&self, context: &ModelContext) -> SpiceResult<String> {
        use std::fmt::Write;
        let prepared = self.ensure(context.temperature, context.nominal_temperature)?;
        let mut out = String::new();
        // version text is only setup/diagnostic data; the load uses v48intVersion.
        let emit = || -> std::fmt::Result {
            let mut out = String::new();
            writeln!(out, "static MODEL: crate::devices::bsim4::fields::Model = crate::devices::bsim4::fields::Model {{")?;
            ${emitter("Model","prepared.model")}
            writeln!(out, "}};")?;
            writeln!(out, "static SIZE: crate::devices::bsim4::fields::SizeParam = crate::devices::bsim4::fields::SizeParam {{")?;
            ${emitter("SizeParam","prepared.size")}
            writeln!(out, "}};")?;
            Ok(())
        };
        // String writes cannot fail. Return the generated Rust rather than a debug dump.
        let _ = emit;
        let result = (|| -> std::fmt::Result {
            writeln!(out, "static MODEL: crate::devices::bsim4::fields::Model = crate::devices::bsim4::fields::Model {{")?;
            ${emitter("Model","prepared.model")}
            writeln!(out, "}};")?;
            writeln!(out, "static SIZE: crate::devices::bsim4::fields::SizeParam = crate::devices::bsim4::fields::SizeParam {{")?;
            ${emitter("SizeParam","prepared.size")}
            writeln!(out, "}};")
        })();
        result.map_err(|_| SpiceError::circuit("cannot emit prepared model"))?;
        Ok(out)
    }
}
`;
    // Remove the duplicate temporary emitter (keeps only the actual closure).
    const clean = helper.replace(/        let emit = \|\| -> std::fmt::Result \{[\s\S]*?        let _ = emit;\n/, "");
    return s.slice(0,start) + "    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {\n        self.stamp_with_model(context, None)\n    }\n" + s.slice(end) + clean;
});
edit("src/devices/bsim4/load.rs", s => s.replace("pub(super) fn load(", "#[inline(always)]\npub(super) fn load("));
for (const [file, names] of Object.entries({bias:["select","commit"],channel:["electrostatics","transport"],junction:["current","charge"],charge:["intrinsic","companion"],leakage:["evaluate"],stamp:["apply"]})) {
    edit("src/devices/bsim4/load/"+file+".rs", s => {
        for (const name of names) s = s.replace("pub(super) fn "+name+"(", "#[inline(always)]\npub(super) fn "+name+"(");
        return s;
    });
}
edit("src/devices/circuit.rs", s => {
    s = s.replace("    fixed_stamps: std::cell::RefCell<crate::maths::FixedStamps>,", "    fixed_stamps: std::cell::RefCell<crate::maths::FixedStamps>,\n    specialization: Option<(usize, bool)>,");
    const start = s.indexOf("        for (position, index) in self.load_order().enumerate() {", s.indexOf("    fn stamp_devices("));
    const cstart = s.indexOf("            let states = request", start);
    const cend = s.indexOf("            })?;", cstart)+ "            })?;".length;
    const context = s.slice(cstart,cend).replace("            device.stamp(&mut StampContext {","            stamp(&mut StampContext {").replace("            })?;", "            })");
    s = s.slice(0,start) + `        if replacements.is_empty() && let Some((case, bake)) = self.specialization {
            if specialization::dispatch(case, bake, self, request, matrix, rhs, trial, mutual, voltage_source_value)? {
                return Ok(());
            }
        }
` + s.slice(start,cstart) + `            self.stamp_one(position, index, request, matrix, rhs, trial, mutual, voltage_source_value, |context| device.stamp(context))?;` + s.slice(cend);
    const insert = s.indexOf("    /// A zeroed MNA matrix", start);
    s = s.slice(0,insert) + `
    #[inline(always)]
    #[expect(clippy::too_many_arguments, reason = "experiment retains the exact load context")]
    fn stamp_one(
        &self, position: usize, index: usize, request: &LoadRequest<'_>,
        matrix: &mut SparseMatrix, rhs: &mut Vector, trial: &mut TrialState,
        mutual: &[Vec<MutualTerm>], voltage_source_value: &mut Real,
        stamp: impl FnOnce(&mut StampContext<'_>) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
${context}
    }

    /// Enable a generated schedule after verifying the circuit inventory.
    /// This research API is for the embedded benchmark fixtures only.
    pub fn enable_specialization(&mut self, case: usize, bake: bool, model: &ModelContext) -> SpiceResult<()> {
        if !specialization::matches(case, self, model)? {
            return Err(SpiceError::circuit("specialization does not match circuit/model"));
        }
        self.specialization = Some((case, bake));
        Ok(())
    }

    /// Emit source from the actual elaborated circuit and prepared models.
    pub fn emit_specialization(&self, case: usize, model: &ModelContext) -> SpiceResult<String> {
        specialization::emit(self, case, model)
    }

` + s.slice(insert);
    return s + "\nmod specialization;\n";
});
