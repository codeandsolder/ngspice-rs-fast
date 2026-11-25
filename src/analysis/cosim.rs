//! Versioned, bounded, checkpointed HDL process transport.
//!
//! A simulator executable is selected by the application, never by evaluating
//! a shell command in a netlist. See docs/port/COSIM.md for the wire contract.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::devices::digital_bridge::{
    Bit, Operation, Reply, Request, Schema, Simulator, TimeScale,
};
use crate::primitives::{SpiceError, SpiceResult};

const MAGIC: [u8; 4] = *b"NSH1";
const REPLAY_MAGIC: [u8; 4] = *b"NSR1";
const ABSENT_TICK: u64 = u64::MAX;
const ABSENT_SNAPSHOT: u32 = u32::MAX;
const REQUEST_HEADER: usize = 36;
const REPLY_HEADER: usize = 32;
const MAX_DIAGNOSTIC: usize = 4096;

fn failure(message: impl std::fmt::Display) -> SpiceError {
    SpiceError::circuit(format!("HDL co-simulation: {message}"))
}

fn width(value: usize) -> SpiceResult<u32> {
    u32::try_from(value).map_err(failure)
}

fn encode(magic: [u8; 4], schema: Schema, request: &Request<'_, [u8]>) -> SpiceResult<Vec<u8>> {
    if request.inputs.len() != schema.inputs
        || request
            .snapshot
            .is_some_and(|state| state.len() > schema.snapshot_bytes)
    {
        return Err(failure("request exceeds port/checkpoint schema"));
    }
    TimeScale::new(request.precision)?.physical(request.tick)?;
    let snapshot = request.snapshot.unwrap_or_default();
    let size = request
        .snapshot
        .map_or(Ok(ABSENT_SNAPSHOT), |state| width(state.len()))?;
    let mut bytes = Vec::with_capacity(REQUEST_HEADER + request.inputs.len() + snapshot.len());
    bytes.extend(magic);
    bytes.extend(request.tick.to_le_bytes());
    bytes.extend(request.precision.to_le_bytes());
    bytes.extend((request.operation as u32).to_le_bytes());
    bytes.extend(width(request.delta)?.to_le_bytes());
    bytes.extend(width(schema.inputs)?.to_le_bytes());
    bytes.extend(width(schema.outputs)?.to_le_bytes());
    bytes.extend(size.to_le_bytes());
    bytes.extend(request.inputs.iter().map(|bit| *bit as u8));
    bytes.extend(snapshot);
    Ok(bytes)
}

fn take<const N: usize>(bytes: &mut &[u8]) -> SpiceResult<[u8; N]> {
    let (field, rest) = bytes
        .split_first_chunk::<N>()
        .ok_or_else(|| failure("truncated frame"))?;
    *bytes = rest;
    Ok(*field)
}

fn decode(magic: [u8; 4], schema: Schema, mut bytes: &[u8]) -> SpiceResult<Reply<Arc<[u8]>>> {
    if take::<4>(&mut bytes)? != magic {
        return Err(failure("incompatible protocol version"));
    }
    let status = u32::from_le_bytes(take(&mut bytes)?);
    if status != 0 {
        if status != 1 || bytes.len() > MAX_DIAGNOSTIC {
            return Err(failure("invalid simulator error response"));
        }
        let diagnostic = std::str::from_utf8(bytes).map_err(failure)?;
        return Err(failure(format!(
            "simulator rejected evaluation: {diagnostic}"
        )));
    }
    let reached_tick = u64::from_le_bytes(take(&mut bytes)?);
    let next = u64::from_le_bytes(take(&mut bytes)?);
    let outputs = usize::try_from(u32::from_le_bytes(take(&mut bytes)?)).map_err(failure)?;
    let snapshot = usize::try_from(u32::from_le_bytes(take(&mut bytes)?)).map_err(failure)?;
    if outputs != schema.outputs
        || snapshot > schema.snapshot_bytes
        || bytes.len() != outputs.saturating_add(snapshot)
    {
        return Err(failure(
            "response exceeds port/checkpoint schema or has trailing bytes",
        ));
    }
    let (bits, state) = bytes.split_at(outputs);
    Ok(Reply {
        reached_tick,
        outputs: bits
            .iter()
            .copied()
            .map(Bit::decode)
            .collect::<SpiceResult<_>>()?,
        next_tick: (next != ABSENT_TICK).then_some(next),
        snapshot: Arc::from(state),
    })
}

struct Exchange {
    payload: Vec<u8>,
    response_limit: usize,
    result: mpsc::SyncSender<io::Result<Vec<u8>>>,
}

fn exchange_io(
    stdin: &mut impl Write,
    stdout: &mut impl Read,
    request: &Exchange,
) -> io::Result<Vec<u8>> {
    let size = u32::try_from(request.payload.len()).map_err(io::Error::other)?;
    stdin.write_all(&size.to_le_bytes())?;
    stdin.write_all(&request.payload)?;
    stdin.flush()?;
    let mut prefix = [0; 4];
    stdout.read_exact(&mut prefix)?;
    let size = usize::try_from(u32::from_le_bytes(prefix)).map_err(io::Error::other)?;
    if size > request.response_limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "oversized HDL response",
        ));
    }
    let mut bytes = vec![0; size];
    stdout.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[derive(Debug)]
struct Session {
    child: Child,
    requests: mpsc::SyncSender<Exchange>,
}

impl Session {
    fn start(program: &Path, args: &[OsString], directory: &Path) -> SpiceResult<Self> {
        let mut child = Command::new(program)
            .args(args)
            .current_dir(directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(failure)?;
        let result = Self::worker(&mut child);
        match result {
            Ok(requests) => Ok(Self { child, requests }),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(error)
            }
        }
    }

    fn worker(child: &mut Child) -> SpiceResult<mpsc::SyncSender<Exchange>> {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| failure("missing simulator stdin"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| failure("missing simulator stdout"))?;
        let (tx, rx) = mpsc::sync_channel::<Exchange>(1);
        std::thread::Builder::new()
            .name("hdl-bridge-io".into())
            .spawn(move || {
                while let Ok(request) = rx.recv() {
                    let result = exchange_io(&mut stdin, &mut stdout, &request);
                    let failed = result.is_err();
                    if request.result.send(result).is_err() || failed {
                        break;
                    }
                }
            })
            .map_err(failure)?;
        Ok(tx)
    }

    fn exchange(
        &self,
        payload: Vec<u8>,
        response_limit: usize,
        timeout: Duration,
    ) -> SpiceResult<Vec<u8>> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.requests
            .try_send(Exchange {
                payload,
                response_limit,
                result: tx,
            })
            .map_err(failure)?;
        rx.recv_timeout(timeout).map_err(failure)?.map_err(failure)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // The I/O worker owns both pipes; terminating the direct executable
        // interrupts its blocking read/write. The channel closes on drop.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A direct, persistent simulator process.
///
/// Every request carries a complete
/// checkpoint, including on retry; the process is never authoritative state.
/// A malformed response or timeout kills the session, allowing a later trial
/// to restart the executable from its accepted snapshot.
#[derive(Debug)]
pub struct ProcessSimulator {
    program: PathBuf,
    args: Vec<OsString>,
    directory: PathBuf,
    timeout: Duration,
    session: Mutex<Option<Session>>,
}

impl ProcessSimulator {
    /// Select an explicitly trusted simulator executable and working directory.
    /// The protocol timeout covers pipe I/O and simulator evaluation.
    ///
    /// # Errors
    /// Relative executable/directory or zero timeout.
    pub fn new(
        program: PathBuf,
        args: Vec<OsString>,
        directory: PathBuf,
        timeout: Duration,
    ) -> SpiceResult<Self> {
        if !program.is_absolute() || !directory.is_absolute() || timeout.is_zero() {
            return Err(failure(
                "absolute paths and a positive timeout are required",
            ));
        }
        Ok(Self {
            program,
            args,
            directory,
            timeout,
            session: Mutex::new(None),
        })
    }
}

impl Simulator for ProcessSimulator {
    type State = Arc<[u8]>;

    fn evaluate(
        &self,
        schema: Schema,
        request: Request<'_, Self::State>,
    ) -> SpiceResult<Reply<Self::State>> {
        let schema = schema.validate()?;
        if request.operation != Operation::ApplyInputs {
            return Err(failure(
                "checkpoint transport requires directly scheduled wakeups",
            ));
        }
        let payload = encode(
            MAGIC,
            schema,
            &Request {
                tick: request.tick,
                precision: request.precision,
                operation: request.operation,
                delta: request.delta,
                inputs: request.inputs,
                snapshot: request.snapshot.map(AsRef::as_ref),
            },
        )?;
        let limit = (REPLY_HEADER + schema.outputs + schema.snapshot_bytes).max(MAX_DIAGNOSTIC + 8);
        let mut slot = self.session.lock().map_err(failure)?;
        if slot.is_none() {
            *slot = Some(Session::start(&self.program, &self.args, &self.directory)?);
        }
        let result = slot
            .as_ref()
            .ok_or_else(|| failure("missing simulator session"))?
            .exchange(payload, limit, self.timeout)
            .and_then(|bytes| decode(MAGIC, schema, &bytes));
        if result.is_err() {
            *slot = None;
        }
        result
    }
}

#[derive(Debug)]
struct Record {
    tick: u64,
    precision: i32,
    operation: Operation,
    delta: usize,
    inputs: Vec<Bit>,
    reached_tick: u64,
    outputs: Vec<Bit>,
    next_tick: Option<u64>,
}

struct ReplayNode {
    previous: Option<Arc<Self>>,
    record: Record,
    bytes: usize,
    steps: usize,
}

impl std::fmt::Debug for ReplayNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplayNode")
            .field("bytes", &self.bytes)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

impl Drop for ReplayNode {
    fn drop(&mut self) {
        // A long, uniquely owned chain must not recursively consume the stack.
        let mut previous = self.previous.take();
        while let Some(link) = previous {
            let Ok(mut node) = Arc::try_unwrap(link) else {
                break;
            };
            previous = node.previous.take();
        }
    }
}

/// Immutable shared input/output history for a simulator without serialization.
#[derive(Clone, Debug)]
pub struct ReplayState(Arc<ReplayNode>);

#[derive(Debug, Default)]
struct ReplaySession {
    process: Option<Session>,
    resident: Option<ReplayState>,
}

impl ReplaySession {
    fn reset(&mut self) {
        self.process = None;
        self.resident = None;
    }

    fn matches(&self, checkpoint: Option<&ReplayState>) -> bool {
        match (&self.resident, checkpoint) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(&a.0, &b.0),
            _ => false,
        }
    }
}

/// How an external simulator discovers autonomous events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayMode {
    /// Simulator reports its next time slot after each evaluation.
    Scheduled,
    /// Simulator advances through a requested horizon to discover outputs.
    Lookahead,
}

/// Persistent process with deterministic replay checkpoints for simulators
/// that cannot serialize their event scheduler (e.g. timed Verilator).
///
/// Forward evaluations append an immutable record in constant work; rollback
/// restarts the process and replays only the requested checkpoint history.
#[derive(Debug)]
pub struct ReplaySimulator {
    program: PathBuf,
    args: Vec<OsString>,
    directory: PathBuf,
    timeout: Duration,
    max_steps: usize,
    mode: ReplayMode,
    session: Mutex<ReplaySession>,
}

impl ReplaySimulator {
    /// Construct a replay backend. The history budget is additionally bounded
    /// by the bridge schema's checkpoint-byte limit.
    ///
    /// # Errors
    /// Invalid paths, timeout or replay limit.
    pub fn new(
        program: PathBuf,
        args: Vec<OsString>,
        directory: PathBuf,
        timeout: Duration,
        max_steps: usize,
        mode: ReplayMode,
    ) -> SpiceResult<Self> {
        if !program.is_absolute() || !directory.is_absolute() || timeout.is_zero() || max_steps == 0
        {
            return Err(failure(
                "absolute paths, positive timeout and replay limit are required",
            ));
        }
        Ok(Self {
            program,
            args,
            directory,
            timeout,
            max_steps,
            mode,
            session: Mutex::new(ReplaySession::default()),
        })
    }

    fn step(
        &self,
        process: &Session,
        schema: Schema,
        request: &Request<'_, [u8]>,
    ) -> SpiceResult<Reply<Arc<[u8]>>> {
        let payload = encode(REPLAY_MAGIC, schema, request)?;
        let bytes = process.exchange(
            payload,
            (REPLY_HEADER + schema.outputs).max(MAX_DIAGNOSTIC + 8),
            self.timeout,
        )?;
        let result = decode(REPLAY_MAGIC, schema, &bytes)?;
        if !result.snapshot.is_empty() {
            return Err(failure("replay simulator returned an unexpected snapshot"));
        }
        Ok(result)
    }

    fn restore(
        &self,
        slot: &mut ReplaySession,
        schema: Schema,
        checkpoint: Option<&ReplayState>,
    ) -> SpiceResult<()> {
        if slot.process.is_some() && slot.matches(checkpoint) {
            return Ok(());
        }
        slot.reset();
        let process = Session::start(&self.program, &self.args, &self.directory)?;
        let mut history = Vec::new();
        let mut cursor = checkpoint.map(|state| state.0.as_ref());
        while let Some(node) = cursor {
            history.push(&node.record);
            cursor = node.previous.as_deref();
        }
        for record in history.into_iter().rev() {
            let result = self.step(
                &process,
                schema,
                &Request {
                    tick: record.tick,
                    precision: record.precision,
                    operation: record.operation,
                    delta: record.delta,
                    inputs: &record.inputs,
                    snapshot: None,
                },
            )?;
            if result.outputs != record.outputs
                || result.next_tick != record.next_tick
                || result.reached_tick != record.reached_tick
            {
                return Err(failure(
                    "non-deterministic HDL simulator during checkpoint replay",
                ));
            }
        }
        slot.process = Some(process);
        slot.resident = checkpoint.cloned();
        Ok(())
    }

    fn evaluate_locked(
        &self,
        slot: &mut ReplaySession,
        schema: Schema,
        request: &Request<'_, ReplayState>,
    ) -> SpiceResult<Reply<ReplayState>> {
        let previous = request.snapshot.map(|state| &state.0);
        let steps = previous.map_or(0, |node| node.steps).saturating_add(1);
        let bytes = previous
            .map_or(0, |node| node.bytes)
            .saturating_add(std::mem::size_of::<ReplayNode>())
            .saturating_add(schema.inputs)
            .saturating_add(schema.outputs);
        if steps > self.max_steps || bytes > schema.snapshot_bytes {
            return Err(failure("HDL replay checkpoint budget exceeded"));
        }
        self.restore(slot, schema, request.snapshot)?;
        let process = slot
            .process
            .as_ref()
            .ok_or_else(|| failure("missing replay session"))?;
        let result = self.step(
            process,
            schema,
            &Request {
                tick: request.tick,
                precision: request.precision,
                operation: request.operation,
                delta: request.delta,
                inputs: request.inputs,
                snapshot: None,
            },
        )?;
        let snapshot = ReplayState(Arc::new(ReplayNode {
            previous: previous.cloned(),
            record: Record {
                tick: request.tick,
                precision: request.precision,
                operation: request.operation,
                delta: request.delta,
                inputs: request.inputs.to_vec(),
                reached_tick: result.reached_tick,
                outputs: result.outputs.clone(),
                next_tick: result.next_tick,
            },
            bytes,
            steps,
        }));
        slot.resident = Some(snapshot.clone());
        Ok(Reply {
            reached_tick: result.reached_tick,
            outputs: result.outputs,
            next_tick: result.next_tick,
            snapshot,
        })
    }
}

impl Simulator for ReplaySimulator {
    type State = ReplayState;

    fn requires_lookahead(&self) -> bool {
        self.mode == ReplayMode::Lookahead
    }

    fn evaluate(
        &self,
        schema: Schema,
        request: Request<'_, Self::State>,
    ) -> SpiceResult<Reply<Self::State>> {
        let schema = schema.validate()?;
        if request.operation == Operation::AdvanceHeld && self.mode != ReplayMode::Lookahead {
            return Err(failure("simulator does not support horizon lookahead"));
        }
        let mut slot = self.session.lock().map_err(failure)?;
        let result = self.evaluate_locked(&mut slot, schema, &request);
        if result.is_err() {
            slot.reset();
        }
        result
    }
}
