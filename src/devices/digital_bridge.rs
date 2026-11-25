//! Typed external HDL model boundary. All mutable simulator state belongs to
//! the trial snapshot; a transport must restore it before every evaluation.

use std::sync::Arc;

use crate::devices::code_model::{
    AutonomousAdvance, CodeModel, DelayMode, Delays, Drive, EventContext, Logic, ModelSchema,
    Output, Strength,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Four-state HDL bit. Weak VHDL states must be normalized by the adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Bit {
    /// Logic zero.
    Zero = 0,
    /// Logic one.
    One = 1,
    /// Unknown.
    X = 2,
    /// High impedance.
    Z = 3,
}

impl Bit {
    /// Decode the version-one wire representation.
    ///
    /// # Errors
    /// Unknown code.
    pub fn decode(value: u8) -> SpiceResult<Self> {
        match value {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            2 => Ok(Self::X),
            3 => Ok(Self::Z),
            _ => Err(SpiceError::circuit("invalid HDL logic encoding")),
        }
    }

    /// Convert a resolved SPICE drive to an HDL bit.
    #[must_use]
    pub const fn from_drive(drive: Drive) -> Self {
        if matches!(drive.strength, Strength::HighImpedance) {
            Self::Z
        } else {
            match drive.logic {
                Logic::Zero => Self::Zero,
                Logic::One => Self::One,
                Logic::Unknown => Self::X,
            }
        }
    }

    /// Convert a four-state output to a SPICE drive.
    #[must_use]
    pub const fn drive(self) -> Drive {
        match self {
            Self::Zero => Drive::strong(Logic::Zero),
            Self::One => Drive::strong(Logic::One),
            Self::X => Drive::strong(Logic::Unknown),
            Self::Z => Drive {
                logic: Logic::Unknown,
                strength: Strength::HighImpedance,
            },
        }
    }
}

/// Decimal HDL timescale. Physical events are quantized down to the latest
/// representable HDL tick; output changes are never delivered in the past.
#[derive(Clone, Copy, Debug)]
pub struct TimeScale {
    seconds: Real,
    exponent: i32,
}

impl TimeScale {
    /// Construct a decimal tick in seconds (1 fs through 1 s).
    ///
    /// # Errors
    /// Unsupported exponent.
    pub fn new(exponent: i32) -> SpiceResult<Self> {
        if !(-15..=0).contains(&exponent) {
            return Err(SpiceError::circuit(
                "HDL timescale exponent must be -15 through 0",
            ));
        }
        Ok(Self {
            seconds: 10_f64.powi(exponent),
            exponent,
        })
    }

    /// Largest tick index that keeps adjacent decimal ticks distinguishable
    /// in binary64 physical time for all supported precisions.
    pub const MAX_TICK: u64 = (1_u64 << 52) - 1;

    /// Translate integer simulator time into physical seconds.
    ///
    /// # Errors
    /// Tick index outside the supported range.
    #[expect(
        clippy::cast_precision_loss,
        reason = "tick is bounded below 2^52 before conversion"
    )]
    pub fn physical(self, tick: u64) -> SpiceResult<Real> {
        if tick > Self::MAX_TICK {
            return Err(SpiceError::circuit("HDL tick exceeds exact integer range"));
        }
        Ok(tick as Real * self.seconds)
    }

    /// Find the latest tick not after this physical time. Correct floating
    /// division near boundaries against the canonical physical conversion.
    ///
    /// # Errors
    /// Negative, nonfinite or out-of-range time.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "finite quotient is explicitly bounded before truncation"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "physical time and quotient are nonnegative"
    )]
    pub fn tick(self, physical: Real) -> SpiceResult<u64> {
        if !physical.is_finite() || physical < 0. || physical > self.physical(Self::MAX_TICK)? {
            return Err(SpiceError::circuit("physical time outside HDL tick range"));
        }
        let mut tick = (physical / self.seconds) as u64;
        tick = tick.min(Self::MAX_TICK);
        if self.physical(tick)? > physical {
            tick -= 1;
        } else if tick < Self::MAX_TICK && self.physical(tick + 1)? <= physical {
            tick += 1;
        }
        Ok(tick)
    }
}

/// Flattened port dimensions and maximum checkpoint size.
#[derive(Clone, Copy, Debug)]
pub struct Schema {
    /// Number of input bits.
    pub inputs: usize,
    /// Number of output bits.
    pub outputs: usize,
    /// Upper bound on opaque simulator checkpoint bytes.
    pub snapshot_bytes: usize,
}

impl Schema {
    /// Maximum bits in either direction.
    pub const MAX_BITS: usize = 65_536;
    /// Maximum opaque checkpoint size.
    pub const MAX_SNAPSHOT: usize = 16 * 1024 * 1024;

    /// Validate all resource bounds.
    ///
    /// # Errors
    /// Empty output, oversized signal or checkpoint.
    pub fn validate(self) -> SpiceResult<Self> {
        if self.inputs > Self::MAX_BITS
            || self.outputs == 0
            || self.outputs > Self::MAX_BITS
            || self.snapshot_bytes == 0
            || self.snapshot_bytes > Self::MAX_SNAPSHOT
        {
            return Err(SpiceError::circuit("invalid HDL bridge resource bounds"));
        }
        Ok(self)
    }
}

/// HDL transaction kind. Both operations begin at the supplied checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Operation {
    /// Apply the supplied inputs at the requested time and settle.
    ApplyInputs = 0,
    /// Hold the existing inputs and advance to the first settled output
    /// change or the requested time horizon, whichever comes first.
    AdvanceHeld = 1,
}

/// One deterministic evaluation, including the previously accepted or trial
/// simulator checkpoint. None represents a fresh, uninitialized model.
#[derive(Debug)]
pub struct Request<'a, C: ?Sized> {
    /// Absolute HDL tick.
    pub tick: u64,
    /// Decimal tick exponent in seconds, matching the simulator precision.
    pub precision: i32,
    /// Operation for this transaction.
    pub operation: Operation,
    /// SPICE delta cycle at the physical event time.
    pub delta: usize,
    /// Flattened input bits.
    pub inputs: &'a [Bit],
    /// Opaque simulator state.
    pub snapshot: Option<&'a C>,
}

/// Complete settled outputs and simulator checkpoint after one evaluation.
#[derive(Debug)]
pub struct Reply<C> {
    /// Actual simulator tick reached by this transaction.
    pub reached_tick: u64,
    /// Flattened output bits.
    pub outputs: Vec<Bit>,
    /// Next absolute HDL time slot (equal tick means another delta cycle).
    pub next_tick: Option<u64>,
    /// Complete simulator checkpoint.
    pub snapshot: C,
}

/// Simulator transport. Evaluating from a checkpoint must produce the same
/// result independently of previously attempted requests or process state.
pub trait Simulator: std::fmt::Debug + Send + Sync + 'static {
    /// Immutable or independently cloneable checkpoint representation.
    type State: Clone + std::fmt::Debug + Send + Sync + 'static;

    /// True for a simulator that requires `AdvanceHeld` operations instead
    /// of returning the next autonomous tick directly.
    fn requires_lookahead(&self) -> bool {
        false
    }

    /// Execute one checkpointed evaluation.
    ///
    /// # Errors
    /// Transport, simulator or protocol failure.
    fn evaluate(
        &self,
        schema: Schema,
        request: Request<'_, Self::State>,
    ) -> SpiceResult<Reply<Self::State>>;
}

/// Cloneable simulator checkpoint with a pending physical wakeup.
#[derive(Clone, Debug)]
pub struct Checkpoint<C> {
    snapshot: Option<C>,
    wake: Option<Real>,
    tick: u64,
    held: Arc<[Bit]>,
    outputs: Arc<[Bit]>,
}

/// External simulator exposed as a native digital code model.
#[derive(Debug)]
pub struct Bridge<S> {
    simulator: S,
    schema: Schema,
    scale: TimeScale,
    delays: Delays,
}

impl<S: Simulator> Bridge<S> {
    /// Bind a simulator and validate its port/checkpoint limits.
    ///
    /// # Errors
    /// Invalid schema or delay.
    pub fn new(simulator: S, schema: Schema, scale: TimeScale) -> SpiceResult<Self> {
        Ok(Self {
            simulator,
            schema: schema.validate()?,
            scale,
            delays: Delays::new(0., 0.)?,
        })
    }

    fn outputs(&self, bits: &[Bit]) -> Vec<Output> {
        bits.iter()
            .map(|bit| Output {
                drive: bit.drive(),
                delays: self.delays,
                mode: DelayMode::Transport,
            })
            .collect()
    }

    fn check_reply(&self, reply: &Reply<S::State>) -> SpiceResult<()> {
        if reply.outputs.len() != self.schema.outputs {
            return Err(SpiceError::circuit("HDL response exceeds declared schema"));
        }
        if self.simulator.requires_lookahead() && reply.next_tick.is_some() {
            return Err(SpiceError::circuit(
                "lookahead simulator must not schedule direct wakeups",
            ));
        }
        Ok(())
    }
}

impl<S: Simulator> CodeModel for Bridge<S> {
    type State = Checkpoint<S::State>;

    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(self.schema.inputs, self.schema.outputs)
    }

    fn initial_state(&self) -> Self::State {
        Checkpoint {
            snapshot: None,
            wake: None,
            tick: 0,
            held: Arc::from([]),
            outputs: Arc::from([]),
        }
    }

    fn next_wakeup(&self, state: &Self::State) -> Option<Real> {
        state.wake
    }

    fn requires_lookahead(&self) -> bool {
        self.simulator.requires_lookahead()
    }

    fn advance_autonomous(
        &self,
        now: Real,
        horizon: Real,
        state: &mut Self::State,
    ) -> SpiceResult<AutonomousAdvance> {
        let tick = self.scale.tick(horizon)?;
        if tick < state.tick {
            return Err(SpiceError::circuit(
                "HDL lookahead horizon precedes checkpoint",
            ));
        }
        if tick == state.tick {
            return Ok(AutonomousAdvance::Quiet);
        }
        let reply = self.simulator.evaluate(
            self.schema,
            Request {
                tick,
                precision: self.scale.exponent,
                operation: Operation::AdvanceHeld,
                delta: 0,
                inputs: &state.held,
                snapshot: state.snapshot.as_ref(),
            },
        )?;
        self.check_reply(&reply)?;
        if reply.reached_tick <= state.tick || reply.reached_tick > tick {
            return Err(SpiceError::circuit(
                "HDL lookahead did not advance causally",
            ));
        }
        let changed = reply.outputs.as_slice() != state.outputs.as_ref();
        let result = if changed {
            let time = self.scale.physical(reply.reached_tick)?;
            if time <= now || time > horizon {
                return Err(SpiceError::circuit(
                    "HDL lookahead discovered an event outside the trial horizon",
                ));
            }
            AutonomousAdvance::Event {
                time,
                outputs: self.outputs(&reply.outputs),
            }
        } else {
            if reply.reached_tick != tick {
                return Err(SpiceError::circuit(
                    "HDL lookahead stopped without an output event",
                ));
            }
            AutonomousAdvance::Quiet
        };
        state.tick = reply.reached_tick;
        state.outputs = reply.outputs.into();
        state.snapshot = Some(reply.snapshot);
        Ok(result)
    }

    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut Self::State,
    ) -> SpiceResult<Vec<Output>> {
        if inputs.len() != self.schema.inputs {
            return Err(SpiceError::circuit("HDL input width mismatch"));
        }
        let tick = self.scale.tick(context.time)?;
        let bits: Vec<_> = inputs.iter().copied().map(Bit::from_drive).collect();
        let reply = self.simulator.evaluate(
            self.schema,
            Request {
                tick,
                precision: self.scale.exponent,
                operation: Operation::ApplyInputs,
                delta: context.delta,
                inputs: &bits,
                snapshot: state.snapshot.as_ref(),
            },
        )?;
        self.check_reply(&reply)?;
        if reply.reached_tick != tick {
            return Err(SpiceError::circuit(
                "HDL input evaluation did not reach its requested tick",
            ));
        }
        let wake = match reply.next_tick {
            Some(next) if next < tick => {
                return Err(SpiceError::circuit("HDL requested a past time slot"));
            }
            Some(next) if next == tick => Some(context.time),
            Some(next) => {
                let time = self.scale.physical(next)?;
                if time <= context.time {
                    return Err(SpiceError::circuit(
                        "HDL wakeup below physical time resolution",
                    ));
                }
                Some(time)
            }
            None => None,
        };
        let outputs = self.outputs(&reply.outputs);
        state.snapshot = Some(reply.snapshot);
        state.wake = wake;
        state.tick = tick;
        state.held = bits.into();
        state.outputs = reply.outputs.into();
        Ok(outputs)
    }
}
