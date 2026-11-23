//! Independent candidate simulation and bounded ensemble scheduling.
//!
//! Prepared parameter and objective plans are shared read-only. Each worker
//! owns a full `CircuitCandidate`, and each candidate starts with independent
//! runtime history. Parallel dispatch bounds both in-flight input and
//! out-of-order output; no full plot batch is retained.

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::mpsc;

use crate::analysis::optimization::derivatives::{self, DerivativeRequest, InputDerivative};
use crate::analysis::optimization::objectives::{PlotObjectives, ScenarioFailure, ScenarioPlan};
use crate::analysis::optimization::parameters::{
    CandidateTopology, CircuitCandidate, CircuitParameterPlan,
};
use crate::analysis::{AnalysisInput, AnalysisPlan};
use crate::netlist::ast::AnalysisCard;
use crate::netlist::eval::{EvalBudget, EvalLimits};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Caller-defined stable identity and owned independent numeric coordinates.
pub struct CandidateJob {
    /// Identity returned unchanged with this candidate's result.
    pub id: u64,
    /// Full independent parameter vector in plan input order.
    pub values: Vec<Real>,
}

/// Explicit per-scenario simulation state policy.
///
/// The first scenario always starts independently; later scenarios may
/// deliberately continue the previous scenario within the same candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioStart {
    /// Reset accepted state, histories, and cached analysis revisions.
    Independent,
    /// Deliberately retain the preceding scenario's state in this candidate.
    Continue,
}

/// Analyses whose numeric requests are fixed may be precompiled once;
/// candidate-dependent options and initial conditions must be resolved
/// against the candidate's `RunConfig` instead.
pub enum ScenarioAnalyses<'a> {
    /// Already validated analysis plans, without repeated parsing.
    Prepared(&'a [AnalysisPlan]),
    /// Re-resolve against this candidate's numeric options and hints.
    Candidate(&'a [AnalysisInput]),
    /// Literalize original netlist cards against this candidate's parameters,
    /// including braced time/frequency arguments, and resolve their drivers.
    Deck(&'a [AnalysisCard]),
}

/// One requested run and its matching pre-bound objective-only output schema.
pub struct CandidateScenario<'a> {
    /// Independent or within-candidate continued initial state.
    pub start: ScenarioStart,
    /// Analysis plans or inputs in intended execution order.
    pub analyses: ScenarioAnalyses<'a>,
    /// Expected plot schemas and requested observations.
    pub objectives: &'a ScenarioPlan,
}

/// The typed reason this candidate could not complete all its scenarios.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateFailure {
    /// A parameter expression, device binding, or topology rebuild failed.
    Preparation(SpiceError),
    /// Resolving candidate-specific analysis arguments failed.
    Planning {
        /// Scenario ordinal.
        scenario: usize,
        /// Underlying numeric or input error.
        error: SpiceError,
    },
    /// A simulation or objective sink failed.
    Simulation {
        /// Scenario ordinal.
        scenario: usize,
        /// Original analysis and source error.
        error: ScenarioFailure,
    },
}

/// One completed candidate, never a partial successful prefix.
///
/// The scenario list contains only requested observations and measurements;
/// unrequested plot rows are streamed and discarded by `ScenarioPlan`.
pub struct CandidateResult {
    /// Exact caller-defined ID.
    pub id: u64,
    /// None if preparing the candidate failed; otherwise final topology class.
    pub topology: Option<CandidateTopology>,
    /// Complete scenario results, or the failure with no partial successes.
    pub outcome: Result<Vec<Vec<PlotObjectives>>, CandidateFailure>,
}

/// One candidate's independently computed physical parameter gradient,
/// without per-parameter circuit resimulation.
pub struct CandidateDerivativeResult {
    /// Original caller-defined candidate identity.
    pub id: u64,
    /// Validated topology, or None on failed parameter preparation.
    pub topology: Option<CandidateTopology>,
    /// Gradient statuses in root parameter input order.
    pub outcome: Result<Vec<InputDerivative>, CandidateDerivativeFailure>,
}

/// Only preparation or common primal/adjoint execution failures abort the
/// derivative result; unsupported individual coordinates stay in the vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateDerivativeFailure {
    /// Candidate coordinates could not be applied transactionally.
    Preparation(SpiceError),
    /// The shared physical sensitivity objective failed to execute.
    Evaluation(SpiceError),
}

/// Counters for a bounded stream, including each failed candidate.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EnsembleStatistics {
    /// Number of candidates whose results have been emitted.
    pub candidates: usize,
    /// Successfully evaluated candidate count.
    pub succeeded: usize,
    /// Candidates requiring rebuilt topology (successful only).
    pub rebuilt: usize,
    /// Candidates that failed preparation, planning, or execution.
    pub failed: usize,
}

impl EnsembleStatistics {
    fn record(&mut self, result: &CandidateResult) {
        self.candidates += 1;
        if result.outcome.is_ok() {
            self.succeeded += 1;
            self.rebuilt += usize::from(result.topology == Some(CandidateTopology::Rebuilt));
        } else {
            self.failed += 1;
        }
    }
}

/// One persistent, exclusively owned candidate simulator. A failed candidate
/// cannot poison a later job, and repeated identical inputs still restart.
pub struct CandidateWorker<'a> {
    session: CircuitCandidate<'a>,
    limits: EvalLimits,
}

impl<'a> CandidateWorker<'a> {
    /// Prepare a worker from immutable parameter bindings.
    ///
    /// # Errors
    /// Baseline circuit/session construction failures.
    pub fn new(plan: &'a CircuitParameterPlan, limits: EvalLimits) -> SpiceResult<Self> {
        Ok(Self {
            session: plan.session()?,
            limits,
        })
    }

    /// Evaluate a single candidate with fresh expression budget and explicit
    /// independent state. Once any stage fails, no partial results are exposed.
    pub fn evaluate(
        &mut self,
        job: &CandidateJob,
        scenarios: &[CandidateScenario<'_>],
    ) -> CandidateResult {
        let mut budget = EvalBudget::new(self.limits);
        if let Err(error) = self.session.update(&job.values, &mut budget) {
            return CandidateResult {
                id: job.id,
                topology: None,
                outcome: Err(CandidateFailure::Preparation(error)),
            };
        }
        let topology = self.session.topology();
        let result = self.run_scenarios(scenarios);
        CandidateResult {
            id: job.id,
            topology: Some(topology),
            outcome: result,
        }
    }

    /// Evaluate one direct smooth DC/AC objective's derivatives after
    /// transactional parameter preparation. This uses the same independent
    /// candidate worker state as black-box objective evaluation, but one
    /// sensitivity adjoint instead of one simulation per tunable parameter.
    ///
    /// Unsupported input parameters have typed statuses in the successful
    /// result; only common parameter preparation and solver failures abort.
    pub fn evaluate_gradient(
        &mut self,
        job: &CandidateJob,
        request: &DerivativeRequest,
    ) -> CandidateDerivativeResult {
        let mut budget = EvalBudget::new(self.limits);
        if let Err(error) = self.session.update(&job.values, &mut budget) {
            return CandidateDerivativeResult {
                id: job.id,
                topology: None,
                outcome: Err(CandidateDerivativeFailure::Preparation(error)),
            };
        }
        let topology = self.session.topology();
        self.session.circuit_mut().clear_candidate_analysis_state();
        CandidateDerivativeResult {
            id: job.id,
            topology: Some(topology),
            outcome: derivatives::evaluate(&mut self.session, request, self.limits)
                .map_err(CandidateDerivativeFailure::Evaluation),
        }
    }

    fn run_scenarios(
        &mut self,
        scenarios: &[CandidateScenario<'_>],
    ) -> Result<Vec<Vec<PlotObjectives>>, CandidateFailure> {
        let mut output = Vec::with_capacity(scenarios.len());
        for (index, scenario) in scenarios.iter().enumerate() {
            if scenario.start == ScenarioStart::Independent || index == 0 {
                // Clear even for an A/A repeat, when parameter update is a no-op.
                self.session.circuit_mut().clear_candidate_analysis_state();
            }
            // Config clone shares its resolved ParamScope through Arc. Holding
            // it separately permits borrowing the mutable circuit for the run.
            let config = self.session.config().clone();
            let context = self.session.analysis_context();
            let plans: Cow<'_, [AnalysisPlan]> = match scenario.analyses {
                ScenarioAnalyses::Prepared(plans) => Cow::Borrowed(plans),
                ScenarioAnalyses::Candidate(inputs) => Cow::Owned(
                    inputs
                        .iter()
                        .cloned()
                        .map(|input| config.plan(input))
                        .collect::<SpiceResult<Vec<_>>>()
                        .map_err(|error| CandidateFailure::Planning {
                            scenario: index,
                            error,
                        })?,
                ),
                ScenarioAnalyses::Deck(cards) => {
                    Cow::Owned(config.plans_for(cards).map_err(|error| {
                        CandidateFailure::Planning {
                            scenario: index,
                            error,
                        }
                    })?)
                }
            };
            let params = config.params().ok_or_else(|| CandidateFailure::Planning {
                scenario: index,
                error: SpiceError::circuit("candidate configuration has no parameter scope"),
            })?;
            let result = scenario
                .objectives
                .run(&plans, self.session.circuit_mut(), &context, params)
                .map_err(|error| CandidateFailure::Simulation {
                    scenario: index,
                    error,
                })?;
            output.push(result);
        }
        Ok(output)
    }

    /// Stream a bounded number of independent candidates in input order.
    /// The result callback runs synchronously, so memory is independent of
    /// total candidate count. If a non-sized source exceeds `max_candidates`,
    /// the already emitted prefix is retained and an error is returned.
    ///
    /// # Errors
    /// Invalid scenario policy or stream exceeding its declared work limit.
    pub fn run_bounded(
        &mut self,
        jobs: impl IntoIterator<Item = CandidateJob>,
        scenarios: &[CandidateScenario<'_>],
        max_candidates: NonZeroUsize,
        mut emit: impl FnMut(CandidateResult),
    ) -> SpiceResult<EnsembleStatistics> {
        validate_scenarios(scenarios)?;
        let mut stats = EnsembleStatistics::default();
        let iter = jobs.into_iter();
        if iter.size_hint().0 > max_candidates.get() {
            return Err(SpiceError::circuit(
                "candidate batch exceeds declared limit",
            ));
        }
        for job in iter {
            if stats.candidates == max_candidates.get() {
                return Err(SpiceError::circuit(
                    "candidate batch exceeds declared limit",
                ));
            }
            let result = self.evaluate(&job, scenarios);
            stats.record(&result);
            emit(result);
        }
        Ok(stats)
    }
}

fn validate_scenarios(scenarios: &[CandidateScenario<'_>]) -> SpiceResult<()> {
    if scenarios.is_empty() {
        return Err(SpiceError::circuit(
            "candidate batch requires at least one scenario",
        ));
    }
    if scenarios[0].start != ScenarioStart::Independent {
        return Err(SpiceError::circuit(
            "first scenario must start from independent simulation state",
        ));
    }
    Ok(())
}

/// Run worker-owned candidates concurrently and emit results in input order.
///
/// Live workers and the reorder window are bounded by worker count. Callbacks
/// only execute on the dispatcher thread; no unsafe thread traits are declared.
/// More workers than the candidate limit are not spawned. An
/// unbounded iterator that exceeds `max_candidates` reports an error after
/// emitting its allowed prefix; caller-provided sized streams are checked
/// before dispatch.
///
/// # Errors
/// Invalid scenario start, failed worker initialization, worker communication,
/// or the caller's declared candidate count limit.
pub fn run_parallel(
    plan: &CircuitParameterPlan,
    scenarios: &[CandidateScenario<'_>],
    jobs: impl IntoIterator<Item = CandidateJob>,
    workers: NonZeroUsize,
    max_candidates: NonZeroUsize,
    limits: EvalLimits,
    mut emit: impl FnMut(CandidateResult),
) -> SpiceResult<EnsembleStatistics> {
    validate_scenarios(scenarios)?;
    let mut iter = jobs.into_iter();
    if iter.size_hint().0 > max_candidates.get() {
        return Err(SpiceError::circuit(
            "candidate batch exceeds declared limit",
        ));
    }
    let count = workers.get().min(max_candidates.get());
    std::thread::scope(|threads| {
        let (sent, received) = mpsc::sync_channel::<(usize, usize, CandidateResult)>(count);
        let (initialized, startup) = mpsc::sync_channel::<SpiceResult<()>>(count);
        let mut senders = Vec::with_capacity(count);
        let mut idle = VecDeque::with_capacity(count);
        for worker_id in 0..count {
            let (tx, rx) = mpsc::sync_channel::<(usize, CandidateJob)>(1);
            let sender = sent.clone();
            let ready = initialized.clone();
            threads.spawn(move || {
                // A finalized Circuit contains non-Send device trait objects.
                // Construct, operate and destroy it exclusively on this thread.
                let Ok(mut worker) = CandidateWorker::new(plan, limits).inspect_err(|error| {
                    let _ = ready.send(Err(error.clone()));
                }) else {
                    return;
                };
                if ready.send(Ok(())).is_err() {
                    return;
                }
                while let Ok((ordinal, job)) = rx.recv() {
                    let result = worker.evaluate(&job, scenarios);
                    if sender.send((worker_id, ordinal, result)).is_err() {
                        break;
                    }
                }
            });
            senders.push(tx);
            idle.push_back(worker_id);
        }
        drop(sent);
        drop(initialized);
        for _ in 0..count {
            startup.recv().map_err(|_| {
                SpiceError::circuit("candidate worker failed during initialization")
            })??;
        }
        let mut results = BTreeMap::new();
        let mut stats = EnsembleStatistics::default();
        let mut dispatched = 0_usize;
        let mut next_result = 0_usize;
        let mut running = 0;
        let mut ended = false;
        let mut exceeded = false;
        let window = count.saturating_mul(2);
        loop {
            while !ended && !idle.is_empty() && dispatched.saturating_sub(next_result) < window {
                if dispatched == max_candidates.get() {
                    ended = true;
                    exceeded = iter.next().is_some();
                    break;
                }
                let Some(job) = iter.next() else {
                    ended = true;
                    break;
                };
                let Some(worker_id) = idle.pop_front() else {
                    return Err(SpiceError::circuit(
                        "candidate dispatcher has no idle worker",
                    ));
                };
                senders[worker_id]
                    .send((dispatched, job))
                    .map_err(|_| SpiceError::circuit("candidate worker stopped unexpectedly"))?;
                dispatched += 1;
                running += 1;
            }
            if running == 0 {
                break;
            }
            let (worker_id, ordinal, result) = received
                .recv()
                .map_err(|_| SpiceError::circuit("candidate worker lost its result"))?;
            running -= 1;
            idle.push_back(worker_id);
            results.insert(ordinal, result);
            while let Some(result) = results.remove(&next_result) {
                stats.record(&result);
                emit(result);
                next_result += 1;
            }
        }
        drop(senders);
        if exceeded {
            Err(SpiceError::circuit(
                "candidate batch exceeds declared limit",
            ))
        } else {
            Ok(stats)
        }
    })
}
