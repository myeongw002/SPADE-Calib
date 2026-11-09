# Configurations (planned)

Configuration-driven pipeline execution has not been implemented. The example below documents the intended fields only. Source-code defaults and algorithm semantics must be audited before a parser is committed.

- `examples/d1.yaml`: a **non-executable** configuration draft with placeholder input paths.
- Sensor calibration data should be stored separately from experiment policy.
- Scene IDs, input/output paths, ground hypotheses, model versions and random seeds must not be embedded in library code.
