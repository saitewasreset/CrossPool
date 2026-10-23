# Raw Artifact Parsing and Causal Reconstruction

## Goal

Provide a Python library for strict raw parsing, provenance-preserving reconstruction,
Routing attachment validation, quality reporting, and offline legacy projections.

## Accepted Changes

- Add `xpool.observability`; use the existing complete installation environment.
- Accept explicit chunk, metadata, and quality paths plus typed attachment indices
  and their root directories. Directory discovery and CLI policy are outside
  the offline reconstruction library.
- Validate chunks one at a time. Reconstruction retains evidence in memory and
  grows with input size; runtime collection budgets do not bound offline memory.
- Keep complete Run-scoped identities, source paths/offsets, actual Producers,
  endpoint provenance, raw domains, absent facts, and all duplicate locations.
- Reject malformed input and identity/event conflicts using structured artifact errors.
  Represent missing valid evidence and reversed boundaries as quality findings.
- Reconstruct Graph templates/executions, Transport operations, Fabric role
  observations, participants, Lane/Lease identities, and explicitly evidenced edges.
  Preserve Graph one-to-many and TP/DP many-to-one relationships.
- Keep Graph owners scoped to their actual executing process. Retain explicit
  Fabric-entry associations independently of publication, execution return and
  result availability, including one Invocation with multiple Graph owners.
- Validate published Routing tensors and preserve independent event/attachment
  availability. Keep calibration samples as raw evidence for clock calibration.
- Provide earliest-N legacy projections and separate quality/provenance without
  changing existing runtime writers or getters.

## Interfaces

`parse_chunk`, `encode_chunk`, `read_artifacts`, `reconstruct`, and
`project_legacy_snapshots` are the supported facade. Strict Pydantic input models
own JSON validation. Immutable identity and source values key reconstruction.
Errors carry a stable code, path, offset, Producer/event identity when known, and
the original exception when applicable. No recovery branches inspect messages.
Exact signatures and supported result types belong to source declarations.

Snapshot operation-attempt totals are explicit per-domain inputs: events cannot
reveal entirely unobserved operations, and process-wide quality counters cannot
invent per-Endpoint totals. Missing totals leave a projection unavailable.

## Implementation and Validation

1. Implement schemas, independent fixed-layout codec, metadata/reference checks,
   and structured errors. Compare all valid bytes and every malformed vector.
2. Implement deduplication, fact merging, reconstruction, quality checkpoints,
   Graph/protocol edges and intervals. Exercise partial evidence, identity reuse,
   Endpoint recreation, Replay/Eager, TP/DP and zero-valued present facts.
3. Implement artifact I/O, Safetensors validation and legacy projections. Compare
   fixed snapshot values and retention behavior; test actual files separately.

Each layer receives focused unit tests and managed Ruff/ty checks. File and
cross-module tests belong to Integration; run the relevant tests through xtest.
Use the repository's pinned Ruff and ty rather than adding another checker.
Existing fixed fixtures remain independent oracles; do not regenerate their expected outputs.
Verify fixture byte lengths and checksums. Exercise non-returning and failed
execution, multiple Graph owners, and unsupported versions through public
parsing/reconstruction boundaries and published-file round trips.
Update the owning Devkit design and user documentation after validation.

## Out of Scope

Runtime collectors, concurrent slot/Context/token mechanisms, CUDA changes,
clock fitting, Perfetto, CLI, external sorting, and disk-backed reconstruction.
Synthetic CPU agreement does not establish real serving or GPU acceptance.
