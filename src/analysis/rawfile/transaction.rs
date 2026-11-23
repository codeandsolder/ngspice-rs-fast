//! Atomic rawfile publication. A temporary path is owned only after exclusive
//! creation succeeds; no destructor is registered for a collision.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, ErrorKind, Read, Seek, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::primitives::{Complex, SpiceError, SpiceResult};

use super::{
    BinaryByteOrder, FinishedPlot, RawFileWriter, RawFormat, RawPlot, TextSpool, unsupported,
};

const ATTEMPTS: usize = 128;
static NEXT_NAME: AtomicU64 = AtomicU64::new(0);

/// Cleanup responsibility for a path acquired by exclusive file creation.
/// On Unix, compare the opened file identity before removing or renaming the
/// path, so an already-replaced name is not accidentally deleted. As with
/// rename-based transactions generally, the directory must be trusted:
/// another process that can arbitrarily change its entries can race a rename.
#[derive(Debug)]
pub(super) struct OwnedTemporary {
    path: PathBuf,
    published: bool,
    #[cfg(unix)]
    identity: (u64, u64),
}

impl OwnedTemporary {
    pub(super) fn open(path: PathBuf) -> io::Result<(File, Self)> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        #[cfg(unix)]
        let identity = {
            use std::os::unix::fs::MetadataExt as _;
            let metadata = file.metadata()?;
            (metadata.dev(), metadata.ino())
        };
        let owned = Self {
            path,
            published: false,
            #[cfg(unix)]
            identity,
        };
        Ok((file, owned))
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    fn verify(&self) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let metadata = fs::symlink_metadata(&self.path)?;
            if !metadata.is_file() || (metadata.dev(), metadata.ino()) != self.identity {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "rawfile temporary path no longer names the exclusively created file",
                ));
            }
        }
        Ok(())
    }

    fn publish(mut self, output: &Path) -> SpiceResult<()> {
        self.verify()
            .map_err(|error| SpiceError::io(&self.path, &error))?;
        fs::rename(&self.path, output).map_err(|error| SpiceError::io(output, &error))?;
        self.published = true;
        Ok(())
    }
}

impl Drop for OwnedTemporary {
    fn drop(&mut self) {
        if !self.published && self.verify().is_ok() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn names(kind: &'static str) -> impl Iterator<Item = OsString> {
    std::iter::repeat_with(move || {
        let id = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
        format!(".spice-rs-{}-{id:016x}-{kind}.tmp", std::process::id()).into()
    })
}

/// The destination has already been anchored to an existing directory.
/// Each candidate is a single generated file name. Exclude the destination
/// itself even when it does not exist yet.
fn create_beside(
    output: &Path,
    candidates: impl Iterator<Item = OsString>,
) -> SpiceResult<(File, OwnedTemporary)> {
    for name in candidates.take(ATTEMPTS) {
        if output.file_name() == Some(name.as_os_str()) {
            continue;
        }
        let path = output.with_file_name(name);
        match OwnedTemporary::open(path.clone()) {
            Ok(created) => return Ok(created),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
            Err(error) => return Err(SpiceError::io(&path, &error)),
        }
    }
    Err(SpiceError::io(
        output,
        &io::Error::new(
            ErrorKind::AlreadyExists,
            "rawfile temporary-name attempts exhausted",
        ),
    ))
}

fn anchor(output: &Path) -> SpiceResult<PathBuf> {
    let name = output.file_name().ok_or_else(|| {
        SpiceError::io(
            output,
            &io::Error::new(ErrorKind::InvalidInput, "the output path names no file"),
        )
    })?;
    let directory = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory_error = |error: io::Error| {
        SpiceError::io(
            output,
            &io::Error::new(
                error.kind(),
                format!("output directory {}: {error}", directory.display()),
            ),
        )
    };
    let parent = directory.canonicalize().map_err(directory_error)?;
    if !parent.is_dir() {
        return Err(directory_error(io::Error::new(
            ErrorKind::NotADirectory,
            "not a directory",
        )));
    }
    Ok(parent.join(name))
}

impl TextSpool {
    pub(crate) fn create(output: &Path) -> SpiceResult<Self> {
        let (file, temporary) = create_beside(&anchor(output)?, names("report"))?;
        Ok(Self {
            file: BufWriter::new(file),
            temporary,
        })
    }

    pub(crate) fn append(&mut self, text: &str) -> SpiceResult<()> {
        self.file
            .write_all(text.as_bytes())
            .map_err(|error| SpiceError::io(self.temporary.path(), &error))
    }

    pub(crate) fn position(&mut self) -> SpiceResult<u64> {
        self.file
            .stream_position()
            .map_err(|error| SpiceError::io(self.temporary.path(), &error))
    }

    pub(crate) fn flush(&mut self) -> SpiceResult<()> {
        self.file
            .flush()
            .map_err(|error| SpiceError::io(self.temporary.path(), &error))
    }

    pub(crate) fn copy_range(&mut self, range: Range<u64>, to: &mut impl Write) -> SpiceResult<()> {
        let length = range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| unsupported("invalid report spool range"))?;
        let copy = (|| -> io::Result<()> {
            self.file.seek(io::SeekFrom::Start(range.start))?;
            let count = io::copy(&mut self.file.get_mut().take(length), to)?;
            if count != length {
                return Err(io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "incomplete report spool",
                ));
            }
            Ok(())
        })();
        copy.map_err(|error| SpiceError::io(self.temporary.path(), &error))
    }
}

/// An atomic output transaction for one complete rawfile.
///
/// Complete plots and one active streamed plot share the same destination.
/// All temporary files are exclusively created beside that destination.
/// A failed operation poisons the transaction; publishing is only possible
/// after all plots have completed and the output writer has flushed. Dropping
/// an unpublished transaction closes its files before removing them.
#[derive(Debug)]
pub struct RawFileTransaction {
    writer: RawFileWriter,
    temporary: OwnedTemporary,
    output: PathBuf,
    failed: bool,
}

impl RawFileTransaction {
    /// Create an exclusive temporary output beside the final destination.
    ///
    /// # Errors
    /// Invalid destination, missing directory, exhausted name candidates or I/O failure.
    pub fn create(
        output: impl AsRef<Path>,
        format: RawFormat,
        order: BinaryByteOrder,
    ) -> SpiceResult<Self> {
        Self::create_with(output.as_ref(), format, order, names("raw"))
    }

    fn create_with(
        output: &Path,
        format: RawFormat,
        order: BinaryByteOrder,
        candidates: impl Iterator<Item = OsString>,
    ) -> SpiceResult<Self> {
        let output = anchor(output)?;
        let (file, temporary) = create_beside(&output, candidates)?;
        let writer = RawFileWriter::from_file(file, temporary.path().to_path_buf(), format, order);
        Ok(Self {
            writer,
            temporary,
            output,
            failed: false,
        })
    }

    fn execute<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> SpiceResult<T>,
    ) -> SpiceResult<T> {
        if self.failed {
            return Err(unsupported("rawfile output transaction has failed"));
        }
        let result = operation(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Append a complete retained plot to the same owned output.
    ///
    /// # Errors
    /// Previous failure, active streamed plot, invalid rows or I/O failure.
    pub fn write_plot(&mut self, plot: &RawPlot) -> SpiceResult<()> {
        self.execute(|this| this.writer.write_plot(plot))
    }

    /// Start streaming directly to the exclusively owned destination.
    /// There is no auxiliary point spool or second temporary file.
    ///
    /// # Errors
    /// Previous failure, active plot, invalid header, seek or I/O failure.
    pub fn begin_plot(&mut self, header: RawPlot) -> SpiceResult<()> {
        self.execute(|this| this.writer.begin_plot(header))
    }

    /// Add one point directly to the active streamed plot.
    ///
    /// # Errors
    /// Previous failure, no active plot, invalid row or write failure.
    pub fn push_point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        self.execute(|this| this.writer.push_point(values))
    }

    /// Patch the point count and complete the streamed plot.
    ///
    /// # Errors
    /// Previous failure, no active plot, seek/flush/write failure.
    pub fn finish_plot(&mut self) -> SpiceResult<FinishedPlot> {
        self.execute(|this| this.writer.finish_plot())
    }

    /// Flush and atomically replace the final output. A failed or incomplete
    /// transaction cannot publish; its temporary files are cleaned up.
    ///
    /// # Errors
    /// Previous failure, an unfinished streamed plot, write/flush or rename failure.
    pub fn publish(self) -> SpiceResult<()> {
        if self.failed || self.writer.has_active() {
            return Err(unsupported(
                "cannot publish a failed or unfinished rawfile transaction",
            ));
        }
        let Self {
            writer,
            temporary,
            output,
            ..
        } = self;
        writer.finish()?;
        temporary.publish(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{Plot, PlotFlags, Variable};
    use ngspice_conformance::ngspice::Scratch;

    fn header() -> RawPlot {
        RawPlot {
            title: "transaction".into(),
            date: String::new(),
            command: String::new(),
            plot: Plot::new(
                "test",
                "Operating Point",
                PlotFlags::Real,
                vec![Variable::new("v(out)", "voltage")],
            ),
        }
    }

    fn candidates<'a>(names: &'a [&str]) -> impl Iterator<Item = OsString> + 'a {
        names.iter().map(OsString::from)
    }

    fn entries(path: &Path) -> Vec<String> {
        let mut entries: Vec<_> = fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        entries.sort_unstable();
        entries
    }

    #[test]
    fn report_spool_replays_ranges_and_removes_owned_path_after_sink_failure() {
        struct FailedWriter;

        impl Write for FailedWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(ErrorKind::BrokenPipe, "closed report sink"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let dir = Scratch::new("report-spool-ranges").unwrap();
        let mut spool = TextSpool::create(&dir.path().join("final.raw")).unwrap();
        let path = spool.temporary.path().to_path_buf();
        let start = spool.position().unwrap();
        spool.append("first plot\n").unwrap();
        let boundary = spool.position().unwrap();
        spool.append("second plot\n").unwrap();
        let end = spool.position().unwrap();
        spool.flush().unwrap();
        let mut replay = Vec::new();
        spool.copy_range(boundary..end, &mut replay).unwrap();
        spool.copy_range(start..boundary, &mut replay).unwrap();
        assert_eq!(replay, b"second plot\nfirst plot\n");
        assert!(spool.copy_range(start..end, &mut FailedWriter).is_err());
        drop(spool);
        assert!(!path.exists());
        assert!(entries(dir.path()).is_empty());
    }

    #[test]
    fn exclusive_candidates_retry_without_truncating_or_removing_existing_paths() {
        let dir = Scratch::new("rawfile-transaction-collision").unwrap();
        let output = dir.path().join("final.raw");
        let sentinel = dir.path().join("sentinel");
        fs::write(&sentinel, "KEEP").unwrap();
        let mut choices = vec!["final.raw", "sentinel"];
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&sentinel, dir.path().join("link")).unwrap();
            std::os::unix::fs::symlink(dir.path().join("missing"), dir.path().join("dangling"))
                .unwrap();
            choices.extend(["link", "dangling"]);
        }
        choices.push("owned");
        let transaction = RawFileTransaction::create_with(
            &output,
            RawFormat::Ascii,
            BinaryByteOrder::LittleEndian,
            candidates(&choices),
        )
        .unwrap();
        assert_eq!(transaction.temporary.path(), dir.path().join("owned"));
        // The final output did not exist and must not have become a temp file.
        assert!(!output.exists());
        drop(transaction);
        assert!(!dir.path().join("owned").exists());
        assert_eq!(fs::read_to_string(&sentinel).unwrap(), "KEEP");
        #[cfg(unix)]
        {
            assert!(dir.path().join("link").is_symlink());
            assert!(dir.path().join("dangling").is_symlink());
            assert!(!dir.path().join("missing").exists());
        }
        let before = entries(dir.path());
        let error = RawFileTransaction::create_with(
            &output,
            RawFormat::Ascii,
            BinaryByteOrder::LittleEndian,
            std::iter::repeat(OsString::from("sentinel")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("attempts exhausted"));
        assert_eq!(entries(dir.path()), before);
        assert_eq!(fs::read_to_string(&sentinel).unwrap(), "KEEP");
    }

    #[test]
    fn stream_failure_and_unfinished_transaction_cannot_publish() {
        let dir = Scratch::new("rawfile-transaction-incomplete").unwrap();
        let output = dir.path().join("final.raw");
        let sentinel = dir.path().join("sentinel");
        fs::write(&output, "PREVIOUS").unwrap();
        fs::write(&sentinel, "KEEP").unwrap();
        for failure in [
            "unfinished",
            "ragged",
            "duplicate",
            "header_write",
            "write",
            "patch",
            "rename",
        ] {
            let mut transaction = RawFileTransaction::create(
                &output,
                RawFormat::Ascii,
                BinaryByteOrder::LittleEndian,
            )
            .unwrap();
            match failure {
                "unfinished" => {
                    transaction.begin_plot(header()).unwrap();
                    transaction.push_point(&[Complex::real(1.)]).unwrap();
                }
                "ragged" => {
                    transaction.begin_plot(header()).unwrap();
                    assert!(transaction.push_point(&[]).is_err());
                    assert!(transaction.finish_plot().is_err());
                }
                "duplicate" => {
                    transaction.begin_plot(header()).unwrap();
                    assert!(transaction.begin_plot(header()).is_err());
                }
                "header_write" => {
                    // BufWriter::stream_position flushes the header prefix;
                    // a read-only descriptor rejects it before any row.
                    transaction.writer.out =
                        std::io::BufWriter::new(File::open(transaction.temporary.path()).unwrap());
                    assert!(transaction.begin_plot(header()).is_err());
                }
                "write" => {
                    // An open read-only descriptor of the owned file injects
                    // a deterministic buffered write failure at publish.
                    transaction.writer.out =
                        std::io::BufWriter::new(File::open(transaction.temporary.path()).unwrap());
                    transaction.write_plot(&header()).unwrap();
                }
                "patch" => {
                    transaction.begin_plot(header()).unwrap();
                    transaction.push_point(&[Complex::real(1.)]).unwrap();
                    transaction.writer.out =
                        std::io::BufWriter::new(File::open(transaction.temporary.path()).unwrap());
                    assert!(transaction.finish_plot().is_err());
                }
                "rename" => {
                    // The writer can flush, but the destination cannot be
                    // replaced by a file (a nonempty directory).
                    let blocked = dir.path().join("blocked");
                    fs::create_dir(&blocked).unwrap();
                    fs::write(blocked.join("keep"), "KEEP").unwrap();
                    transaction.output = blocked.clone();
                    transaction.write_plot(&header()).unwrap();
                    assert!(transaction.publish().is_err());
                    assert_eq!(fs::read_to_string(blocked.join("keep")).unwrap(), "KEEP");
                    fs::remove_dir_all(blocked).unwrap();
                    assert_eq!(entries(dir.path()), ["final.raw", "sentinel"]);
                    continue;
                }
                other => panic!("unknown injected failure {other}"),
            }
            assert!(transaction.publish().is_err(), "{failure}");
            assert_eq!(
                fs::read_to_string(&output).unwrap(),
                "PREVIOUS",
                "{failure}"
            );
            assert_eq!(fs::read_to_string(&sentinel).unwrap(), "KEEP", "{failure}");
            assert_eq!(entries(dir.path()), ["final.raw", "sentinel"], "{failure}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn replaced_temporary_name_is_not_published_or_removed() {
        let dir = Scratch::new("rawfile-transaction-replaced").unwrap();
        let output = dir.path().join("final.raw");
        fs::write(&output, "PREVIOUS").unwrap();
        let mut transaction = RawFileTransaction::create_with(
            &output,
            RawFormat::Ascii,
            BinaryByteOrder::LittleEndian,
            candidates(&["owned"]),
        )
        .unwrap();
        let owned = transaction.temporary.path().to_path_buf();
        // Keep the original inode allocated so the replacement cannot reuse it.
        fs::rename(&owned, dir.path().join("moved")).unwrap();
        fs::write(&owned, "REPLACEMENT").unwrap();
        transaction.write_plot(&header()).unwrap();
        assert!(transaction.publish().is_err());
        assert_eq!(fs::read_to_string(&output).unwrap(), "PREVIOUS");
        assert_eq!(fs::read_to_string(&owned).unwrap(), "REPLACEMENT");
        assert_eq!(entries(dir.path()), ["final.raw", "moved", "owned"]);
    }

    #[test]
    fn direct_stream_successfully_replaces_without_an_auxiliary_file() {
        let dir = Scratch::new("rawfile-transaction-success").unwrap();
        let output = dir.path().join("final.raw");
        let sentinel = dir.path().join("sentinel");
        fs::write(&output, "PREVIOUS").unwrap();
        fs::write(&sentinel, "KEEP").unwrap();
        let mut transaction = RawFileTransaction::create_with(
            &output,
            RawFormat::Binary,
            BinaryByteOrder::BigEndian,
            candidates(&["owned"]),
        )
        .unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&sentinel, dir.path().join("link")).unwrap();
        transaction.begin_plot(header()).unwrap();
        let temporary_name = transaction
            .temporary
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(entries(dir.path()).len(), if cfg!(unix) { 4 } else { 3 });
        assert!(entries(dir.path()).contains(&temporary_name));
        transaction.push_point(&[Complex::real(12.)]).unwrap();
        let finished = transaction.finish_plot().unwrap();
        assert_eq!(finished.point_count, 1);
        assert_eq!(fs::read_to_string(&output).unwrap(), "PREVIOUS");
        transaction.publish().unwrap();
        assert!(!dir.path().join("owned").exists());
        assert_eq!(fs::read_to_string(&sentinel).unwrap(), "KEEP");
        let bytes = fs::read(&output).unwrap();
        let raw = super::super::RawFileReader::new(bytes.as_slice())
            .with_byte_order(BinaryByteOrder::BigEndian)
            .read()
            .unwrap();
        assert_eq!(
            raw.single_plot().unwrap().value("v(out)", 0),
            Some(Complex::real(12.))
        );
        #[cfg(unix)]
        assert!(dir.path().join("link").is_symlink());
    }
}
