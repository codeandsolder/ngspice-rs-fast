//! Development-only C reference tooling, independent of the simulator.
//!
//! Fixture discovery, reference selection and child-process mechanics need
//! only the standard library. Analysis scheduling and typed rawfile comparison
//! belong to their simulator-aware callers.

pub mod fixtures;
pub mod ngspice;

#[cfg(test)]
struct Temp(std::path::PathBuf);

#[cfg(test)]
impl Temp {
    fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ID: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "ngspice-conformance-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("creating {}: {error}", path.display()),
            }
        }
    }
}

#[cfg(test)]
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
