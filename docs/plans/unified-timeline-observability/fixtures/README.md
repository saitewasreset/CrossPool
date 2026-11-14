# Reference Fixtures v6

These immutable inputs and expected outputs implement the
[shared contract](../shared-contract.md) and [calibration algorithm](../calibration.md).
They are synthetic contract evidence, not captured serving or clock measurements.
Fixture readers use exact integers and reduced rational values. Each scenario
starts from a fresh Run state; identical IDs across scenarios never join them.

## Files

| File | Contract surface |
| --- | --- |
| [layout.json](layout.json) | Complete 256-byte field table, presence groups, event IDs, and FFN enums |
| [metadata.json](metadata.json) | Immutable metadata revision used by valid raw encoding; SHA-256 is over its exact bytes |
| [metadata-atnagent.json](metadata-atnagent.json) | The same registered identities with an AtnAgent Producer envelope for anomalous raw chunks |
| [identity-lifecycle.json](identity-lifecycle.json) | Allocation/policy and diagnostic preconditions, Generation-scoped PE membership, and observed-process registration boundary |
| [encoding.json](encoding.json) | Full one-event chunk hex and malformed variants with exact structured errors |
| [causality.json](causality.json) | Complete/partial events, explicit associations, Replay/Eager, identity isolation, Context, tokens, quota, and tail cases |
| [quality.json](quality.json) | Full quality checkpoint and conservation, export failure, unknown tail, and counter saturation cases |
| [routing.json](routing.json) | Publication/attachment outcomes and independent event/tensor availability |
| [routing.safetensors](routing.safetensors) | Literal published attachment, live_rows=2, effective_topk=2 |
| [calibration.json](calibration.json) | Exact blocks, feasible vertices, mappings, held-out failures, drift, gaps, Epochs, and durations |
| [checksums.json](checksums.json) | Byte lengths and SHA-256 for the data files; excludes this README and itself |

These fixtures are independent of generators, runtime collectors, and offline readers.
Implementations consume these files directly; they must not regenerate their own expected
outputs using the implementation under test. An intentional contract change
updates version, bytes, semantic expectations, and checksums together.

## Case Interpretation

JSON top-level contract_version is 6. Each `cases` entry has a unique `id`,
`input`, and `expected`. Expected dictionaries are assertions on the named
properties, not a license to discard other quality evidence. The complete
`normal_dense` case specifies full ordinary behavior.
Small counter/Context/quota cases deliberately describe that owning interface
rather than manufacturing a complete protocol trace for it.

Derived causal inputs use base_case=normal_dense. Copy that input, remove events
by `[producer,event_sequence]`, replace listed events by the same identity, then
append the listed events in order. Replace other explicitly supplied input
properties; unspecified properties are inherited. Remove/replace must match
exactly one base event. Appends may intentionally duplicate an identity to test
conflicts. This fixed expansion convention avoids repeating the complete trace
in every case and is separate from the production wire format.

For event cases, `events` entries contain Producer plus a full canonical wire
record: every absent field is explicitly zero. Use layout presence bits to
distinguish known zero from absent. Producer/Run come from the surrounding
context; `metadata` references a fixture-relative file. `metadata_overrides`
defines a fresh scenario's metadata revision: endpoints_add appends Endpoint
definitions, and model_topologies replaces the Generation's topology. IDs in
these lists are composite keys, not tensor-slot indices or array positions.
producers_add, clock_domains_add, and graph_templates_add append registered
definitions; generations_replace replaces the matching Generation by ID. Scenario capability
overrides replace the named capabilities for that scenario.
Independent scenario IDs are not serialized into the production format.

Encoding vectors use the top-level metadata file unless a vector supplies its
own `metadata` filename. Each chunk's Producer, process and clock must agree
with that exact source-specific document, including its SHA-256 prefix.
TransportAnomaly uses kind 150 and RawRequest bit 18; its data0 identifies the
original boundary and data1 through data4 preserve raw request values. Bit 18
does not control payload words belonging to other Event families.

Membership cases in identity-lifecycle.json use `membership_metadata` to name
their base metadata file. Their `metadata_overrides` also permit
`generations_add`, `producers_replace`, and `clock_domains_replace`, keyed by
registered `id`; `revision` replaces the document revision. Replacements must
match one base definition, and additions must have fresh IDs. `prior_metadata`
names an earlier immutable revision for cross-revision membership validation.
`membership_queries` contains `{generation, producer}` lookups; expected `pes`
is the ordered result, with null for a nonmember or absent Generation evidence.
An unresolved positive Generation reference is an error, not a null result.
`events` checks recording-source membership and, when present, the Event's PE.
Malformed metadata errors precede event validation. The raw vectors reject
an unsupported header version and complete v5/v4/v3 chunks without compatibility decoding.
Daemon registration
expectations describe launcher behavior, not an additional serialized Event role.

`normal_dense` retains one Instance operation, its AtnAgent observation,
Fabric entry, three Fabric role observations, one SGLang Replay, and one native
FFN activity. The Host return at 350 is distinct from the FFN compute completion
at raw Device 240; clocks in this example have no implicit cross-domain mapping.
Same-domain intervals remain calculable without calibration. Producer 9 is the
registered AtnAgent Host source; Producer 5 owns both Coordinator and FfnAgent
evidence, distinguished by role and protocol boundary rather than local trace ID.
Dense DirectPartial has no PartialReadyPublished/PeerPartialsReadyObserved Hook.
A disabled Routing class alone does not establish that a layer is Dense.

Event bytes 136–143 are reserved zero. Presence bit 8 is forbidden; all active
presence groups retain their offsets. Semantic duplicates and conflicts use
Operation for Transport and Invocation for Fabric, scoped by local Producer,
role/site and boundary. TransportAnomaly uses its original boundary, not kind
150 alone. Event identity conflicts remain independently checked.

For Routing, capabilities.routing=enabled changes the base Dense fixture to a
synthetic MoE publication scenario. Published Routing data is:

```text
topk_ids:     [[0, 1], [1, 0]]       int32
topk_weights: [[0.75, 0.25], [0.5, 0.5]]  float32
```

Its final attachment index uses the fixture-relative routing.safetensors path.
The event's artifact_id is zero because runtime file publication occurs later;
the exact attachment identity resolves the final index. Not-collected, pool
capacity, payload oversize, I/O failure, missing file, and unknown crashed tail
are different expected outcomes. Safetensors metadata contains contract_version,
run_id, producer_id and records, with per-attachment identities and facts. Tensor
values and keys retain the fixed values above.

The identity-lifecycle cases describe preparation and diagnostic interfaces:
allocation is required by Unified Timeline, Transport Observer, or an explicitly
identity-dependent consumer, rather than by Graph/Routing enablement alone.
Diagnostic availability is a precondition result; equal Context generations do
not validate live mutable storage. The stale graph field in the live-peek case
is deliberately invalid diagnostic input, not part of the version 6 IPC layout.
The causal cases execution_never_returns and execution_returns_failure retain
the entry edge without inventing FFN completion. invocation_multiple_graph_owners
preserves separate FfnAgent executions under one Invocation. Attributed SGLang
edges originate in Instance evidence; AtnAgent Transport events carry no Graph
identity obtained from another process.

## Anomalous Transport Examples

Raw encoding vectors cover every original Transport boundary, each invalid
request enum and combined invalid values, actual Shutdown, and recorded zero
time. Invalid masks, presence, boundary/site and result combinations fail strict
validation. These bytes and expected semantic values are independent of runtime
encoders and readers. The quality lower-bound case distinguishes retained
anomalous Operations from Event attempts and loss.

## Numerical Oracles

Calibration input declares its attempted range, provenance, raw Epoch, load
phase, samples, and query Device times. Positions 4 and 9 are held out. Each
valid expected model includes reduced rational vertices (a,c), a and b,
support, fitting/validation sequences, and exact/rounded mapping results.
Unavailable model entries assert their reason; their public numerical values
are null. Sample failure retains null G/H_after, never a fabricated Device read.

known_offset has exactly a=1, b=100. known_rate has a=2, b=50.
asymmetric_windows has a=1, b=105, with feasible slope range [8/9,10/9]; its
estimate is conditional, not independent truth. ties_to_even checks half-integer
estimates at G=3 and G=5. held_out_failure changes a validation sample only,
so it must not silently change the fitted mapping to pass. model_conflict
changes a fitting window and must not be repaired by deleting that sample.

Series cases define consecutive blocks or distinct Epochs. The drift case has
rates 1 and 101/100, separated support, and resolved slope difference 10000 ppm.
between_blocks remains unavailable. Duration fixtures explicitly preserve
negative bounds and protocol edges when temporal ordering is indeterminate.
unresolved_drift keeps a rate difference interval containing zero. Duplicate,
conflicting, and invalid-bracket samples test the input boundary before fitting.
wire_sample_cases connect the fixed runtime Event payload to the public Sample
shape, including timeout with unknown Device time and retained resource ownership.

## Required Downstream Checks

- Encode every field of the valid vector and compare all 384 bytes. Decode it
  independently and compare the complete expected record, Producer, and CRCs.
  Validate metadata hashing before payload semantic decoding.
- Reject every malformed vector at the specified error stage. Recomputed CRCs
  on unknown kinds/flags/presence ensure semantic rejection is actually tested.
- Reconstruct protocol identities and causal edges from retained Events;
  compare semantic duplicate/conflict handling with the new observation keys.
  Assert no guessed edge in missing_association and no timestamp fabricated
  from repeated scalar facts.
- Cover out-of-order commits, completed-copy-only slot reuse, stale generation,
  Context exhaustion and retirement, reserved-zero fields, and identity
  exhaustion without modifying production outcomes.
- Read the literal Safetensors attachment; verify checksum, metadata identity,
  dtype, shape, values, and tensor keys. Event loss does not erase a tensor;
  a tensor alone never creates a missing event.
- Recompute feasible regions using exact rational arithmetic; compare sorted
  vertices, representative mapping, validation, support, bounds, rounding, and
  duration classifications. Test duplicate/conflicting sample inputs and
  invalid bracketing windows as structured input errors.

CPU fixture agreement enables parallel implementation. It proves neither
concurrent CUDA correctness nor actual Host/Device accuracy. Those acceptance
boundaries remain in the [task plan](../README.md).
