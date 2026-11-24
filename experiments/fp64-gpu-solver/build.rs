use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    for shader in ["tridiagonal", "sparse_fixed_pivot"] {
        let source = format!("shaders/{shader}.comp");
        let output = out.join(format!("{shader}.spv"));
        println!("cargo:rerun-if-changed={source}");
        let status = Command::new("glslc")
            .args(["-fshader-stage=comp", "--target-env=vulkan1.2", "-O"])
            .arg(&source)
            .arg("-o")
            .arg(&output)
            .status()
            .expect("glslc is required to compile FP64 GLSL for Vulkan");
        assert!(status.success(), "glslc failed for {source}");
        let status = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&output)
            .status()
            .expect("spirv-val is required to verify FP64 SPIR-V");
        assert!(status.success(), "SPIR-V validation failed for {source}");
    }
}
