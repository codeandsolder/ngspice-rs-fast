# Native prepared model integration

The shared `model-kernel-api` crate defines descriptors, immutable artifacts,
prepared numerical contexts, reusable point buffers and homogeneous tiles.
The solver consumes the API directly. Model producers own physical equations
and analytic derivatives; the solver owns topology, matrix assembly, integration,
Newton iteration and accepted state.

* Descriptor equation order defines the binding. Explicit terminals are supplied
  by the caller and hidden nodes are allocated transactionally.
* Thermal node voltages use Celsius at the circuit interface and Kelvin at the
  artifact boundary. The Newton affine term uses circuit coordinates.
* Computed storage uses two history slots per equation for charge and derivative.
  Unavailable storage fails explicitly when required by an analysis.
* The prepared run applies numeric replacements and prepares changed contexts
  outside the Newton load. Instance ambient temperature is either a fixed value
  or a circuit-relative offset; the compiled artifact is shared.
* Optional coordinate-domain Newton limiting uses prior trial or accepted
  coordinates and retries physical domain failures along that segment.
  Limited trials report nonconvergence and do not alter the exact model law.
* Noise sources can inject coherently into arbitrary bound rows with complex
  coefficients. Frequency-dependent densities refresh without rebuilding the
  circuit topology. The RF noise path requires its own source classification;
  generalized sources cannot silently claim this classification.

VDMOS is the first native producer bound through this interface. Its parameter
compilation, thermal physics, hidden equations and recovery derivatives belong
to the producer. The solver's VDMOS source binding selects immutable artifacts
by declaration, nominal temperature and thermal topology.

The direct homogeneous tile API is available for numerical producer tests.
Circuit-level dispatch currently evaluates individual prepared devices; grouping
the circuit's hot path by artifact remains a separate integration step.
