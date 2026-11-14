# Shared Observability Contract v6

This is the target contract linked from the [task plan](README.md). Version 6
supersedes versions 1 through 5 as one coordinated specification/fixture revision.
Readers and participants accept only version 6. Artifacts comprise immutable
Events, metadata, quality checkpoints and attachments. Native ABI versions
remain independently managed. Event size remains 256 bytes; bytes 136–143
are reserved zero and presence bit 8 is forbidden. Other field offsets, IPC
field layout and calibration mathematics are unchanged.

All integers on disk are little-endian except opaque UUID bytes. Native struct padding,
`size_t`, pointers, CUDA handles, and enum ordinals inferred from C++ layout are
never wire representations. Fixture JSON uses exact integer tokens; consumers
must not round them through IEEE-754 doubles.

## Identity and Metadata

| Identity           | Type and scope                                           | Allocation and isolation                                                                                                                                                                                                                                                                                                  |
| ------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run                | 16-byte UUID                                             | New launch Attempt; never PID or wall time alone                                                                                                                                                                                                                                                                          |
| Process creation   | 16-byte UUID                                             | New OS process creation, even after PID reuse                                                                                                                                                                                                                                                                             |
| Producer           | Positive u64 within Run                                  | Launcher registers one Host source per observed Instance Rank, AtnAgent, or FfnAgent process, and one Device source per such process/Device; retries never reuse IDs                                                                                                                                                      |
| Generation         | Positive u64 metadata reference within Run               | Resolves to the exact production FabricGenerationId string                                                                                                                                                                                                                                                                |
| Endpoint           | `(owner_producer, endpoint_sequence)`                    | `owner_producer` is the AtnAgent process's Device Producer; the AtnAgent Device source allocates a positive sequence for each new IPC observation region. One region is exported per Transport Arena pairing, matching the existing one-Arena-per-Instance-rank export granularity; it is never shared across TP/DP ranks |
| Operation          | `(Run, Endpoint, operation_sequence)`                    | Positive Device counter after successful Mailbox Claim; independent of event reservations                                                                                                                                                                                                                                 |
| Graph template     | `(Run, graph_owner, template_sequence)`                  | Host source registers a fresh template when capturing/installing; actual CUDA handles are not identity                                                                                                                                                                                                                    |
| Graph execution    | `(Run, graph_owner, execution_sequence)`                 | Fresh positive sequence per actual invocation of the execution path; Eager has no template                                                                                                                                                                                                                                |
| Invocation         | `(Run, Generation, instance_index, invocation_sequence)` | Uses production InvocationKey; index may be zero, sequence is positive                                                                                                                                                                                                                                                    |
| Lane/Lease         | `(Run, Generation, lane_index, lease_sequence)`          | Uses production assignment; Lane zero is valid, lease is positive                                                                                                                                                                                                                                                         |
| Routing attachment | `(Run, attachment_owner, attachment_sequence)`           | Device publication allocates sequence independently of event/tensor capacity                                                                                                                                                                                                                                              |
| Clock domain       | Positive u64 within Run                                  | Host CLOCK_MONOTONIC reference; Device domain additionally names process, physical UUID, and raw-clock Epoch                                                                                                                                                                                                              |
| Event              | `(Run, Producer, event_sequence)`                        | Positive attempted-event counter; never a buffer slot or operation sequence                                                                                                                                                                                                                                               |

Producer IDs and metadata references are registered before Capture and copied
as immutable observation configuration. Endpoint sequences are not local arena
indices. The Endpoint owner is shared by the Instance and AtnAgent, although
their event Producers differ. Each Producer has exactly one clock domain;
one process can have several Producers. Physical UUIDs, process creation,
logical-device mapping, model IDs, Instance topology, scheduler policy,
capabilities, and artifact references live in metadata.

The Daemon allocates and distributes observation identities and brokers IPC
descriptors; it is not an event source in this contract. It registers no Host
Producer for itself and owns no observation Collector/Writer or associated
collection budget. An absent Daemon Producer artifact is not an evidence gap.
Run-level missing-process checks range only over registered observation sources.
`coordinator` and `collector` are Event roles, not additional process roles:
the Coordinator executes in the first FfnAgent PE, and calibration collection
uses the observed process's Host Producer. Future Daemon control-plane events
require a separately specified contract extension.

The launcher distributes one Run and the registered Producer/Generation/clock
IDs through the existing preparation/bootstrap path. IPC exports add optional
`observation_region` with exactly version:u32=5, owner_producer:u64,
endpoint_sequence:u64, generation:u64, allocation_bytes:u64, and handle.
Handle uses the existing canonical hexadecimal CUDA IPC handle representation;
it is never persisted in raw timeline metadata. Disabled related observation
exports null and allocates no region. Version or preparation-policy mismatch
fails startup; all participants use the same Native ABI, with no old-worker
fallback.

### Endpoint allocation, export, and attachment

One `observation_region` is allocated per Transport Arena pairing, in the same
`cudaMalloc` call sequence that creates that Arena (the existing
`Arena::create()` allocation path). The AtnAgent process's Device Producer
assigns a fresh positive `endpoint_sequence` immediately before this
allocation; sequences are never reused within a process lifetime. The region's
IPC handle is exported through `cudaIpcGetMemHandle` and encoded with the
existing `xpool::utils::hex::HexValue<cudaIpcMemHandle_t>` convention,
identical to the Transport Arena handle encoding (`Arena::handle()`). It is
published to the Daemon broker as additional sibling fields on the same
descriptor record already used for the Arena handle (the existing
`TransportBroker.publish()` path); no new RPC is introduced. The Instance
fetches and attaches it with `cudaIpcOpenMemHandle` at the same call site
where it already attaches the Transport Arena handle. Preparation derives one
immutable `transport_identity_required` policy for
both participants: it is true when Unified Timeline or Transport Observer is
enabled, or another enabled observation consumer explicitly requires shared
Endpoint/Operation identity. Only when it is false is `observation_region` null
and no region allocated. A required region missing from a descriptor, or a
participant policy/version mismatch, fails startup. Graph-only and Routing-only
observation do not require this region merely because they are enabled;
process-local Context and sampler storage are separate resources. Independent
mode accounts for the region in its observation allocation budget as well.

### Endpoint recreation

A new region and a new `endpoint_sequence` are allocated whenever the
underlying Transport Arena is itself recreated: an AtnAgent process restart
(new process-creation identity), or an explicit Arena re-pairing following a
Fabric Generation rollover. `generation` in the descriptor is the Generation
reference current at the moment of this allocation. The prior region's CUDA
allocation and IPC mapping retire only after the Instance has detached
(`cudaIpcCloseMemHandle`) and any Collector work
still reading the prior generation's events has drained, following the
general rule above that mappings retire after IPC detach and in-flight work.
The hot descriptor itself carries no "superseded" field; Endpoint lineage
across recreation is a metadata concern (see Metadata revision below), not a
per-export wire concern.

### Operation word and write/read timing

The versioned public prefix contains only `operation_sequence:u64` at offset
zero, naturally eight-byte aligned. Its size is eight bytes; allocation-alignment
rounding is accounted separately. There is no Graph identity or Graph relay in
this region. Graph relationships are immutable events recorded by the actual
execution owners, not a last-resolved Graph tuple on an Endpoint.

The Instance writes the fresh positive sequence after successful Claim
(`try_begin_staging()` returns true), before `publish_request()`. The AtnAgent
reads it after `observe()` returns `Published`. Visibility reuses the Mailbox's
existing release-acquire pair, which already protects `payload_rows` and
`request`. No new atomic primitive is required for this protocol-owned access.
A subsequent Claim can overwrite the word only after the previous cycle's
`acknowledge()` changes Evaluated to Idle. The AtnAgent must copy the identity
into every event that needs it before `publish_result()` permits reuse; the
background Collector never rereads this mutable word to complete an event.

The successful claimant increments the operation counter independently of
event reservations. Exhaustion follows the identity-exhaustion policy below;
it must not wrap or abort production. Endpoint recreation creates a new region
and sequence. Observation state stays out of production RequestMetadata and
the production Arena allocation. Mappings retire after IPC detach and in-flight
work.

### Diagnostic reads

Online diagnostics consume immutable committed events through the Collector.
Direct inspection of `observation_region` or Dynamic Context storage is valid
only after the relevant writers and readers are quiesced or explicitly
synchronized. A diagnostic request without that precondition reports
`unavailable`; it must not accept a live raw-memory sample as a consistent
snapshot. If execution is stuck and synchronization cannot complete, diagnostics
can show already collected evidence and unknown tails, not a fabricated current
snapshot.

`context_generation` identifies Context slot reuse; it is not a seqlock or a
snapshot version. Equal reads do not exclude an Operation update, an unfinished
Context write, or a mixed-field sample. This version defines no live IPC
snapshot API. Adding one requires a separately specified publication protocol
covering all writers, atomic access and memory ordering, cross-process
visibility, bounded retries, and slot lifetime during reads. A version check or
double buffer alone is insufficient.

### Dynamic Context storage: process-wide, not per-Endpoint

Context identity `(graph_owner, execution_sequence, template_sequence,
context_generation)` is scoped to the actual Graph-owning process's Host Producer
(`graph_owner`): the Instance Rank for SGLang execution, and the FfnAgent for
native FFN execution. AtnAgent does not own either Graph family. This scope is
independent of any single Endpoint: a `lane_service` execution can
serve multiple Invocations over its lifetime, and TP/DP activity under one
Invocation is not attributable to exactly one Endpoint. The canonical bounded
Context slot array is therefore a separate, process-wide Device allocation
owned by the Graph Observer, sized once per `graph_owner`, independent of how
many Endpoints that process currently has open. Its recommended default
layout:

| Offset | Field                | Type | Purpose                                                                                     |
| ------ | -------------------- | ---- | ------------------------------------------------------------------------------------------- |
| 0      | `slot_generation`    | u64  | Incremented on every reuse; the generation a reader must match                              |
| 8      | `execution_sequence` | u64  | Owning Graph execution identity                                                             |
| 16     | `template_sequence`  | u64  | Zero for Eager                                                                              |
| 24     | `stream_ordinal`     | u64  | Process-local monotonic registration ID of the owning Stream; never a raw `CUstream` handle |
| 32     | `state`              | u32  | `Free=0`, `Reserved=1`, `Committed=2`                                                       |
| 36     | (padding)            | u32  | Zero                                                                                        |

Slot size is 40 bytes. Default capacity (`context_slot_capacity`) is **64**
per process, sized for the worst-case concurrent-Stream count across all
observed execution paths in the owning Instance Rank or FfnAgent process.
`context_slots_bytes = 64 * 40 = 2560` bytes. Concurrent Streams use distinct live slots; slot reuse
follows the existing completion-boundary rule. Context overflow increments
`context_unavailable` and leaves association unavailable; it never overwrites
an in-flight slot. Disabled Graph observation allocates no Context array.

An alternative of folding this array into each `observation_region` (one full
Context array per Endpoint) was evaluated and rejected: it requires either
duplicating `lane_service` Context across every Endpoint it serves or picking
one arbitrary "owning" Endpoint for it, both of which conflict with the
Context identity's process-wide scope. FfnAgents have no Transport Endpoint in which to store such Context.

### Calibration sampler storage: per-process Device Producer, not inside observation_region

When calibration is enabled, a sampler identified by
`(Run, Host Producer, Device clock_domain_id)` must exist for every process
with a Device Producer, including FfnAgent
processes that never create an `observation_region` at all (FfnAgent has no
Transport Endpoint). Sampler storage is therefore a separate, non-IPC,
per-process Device allocation, parallel to `observation_region` rather than
inside it:

| Field      | Type | Purpose                                                                                                                                 |
| ---------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `raw_g_ns` | u64  | Written by the short sampling Kernel on the dedicated Stream; read back by the Host via `cudaMemcpyAsync` after completion confirmation |

Default `sampler_slot_capacity` is **1** per process (one clock domain per
Device Producer in the common case), configurable up to a small bound for a
process observing more than one Device. `sampler_bytes =
sampler_slot_capacity * 8`. All other sampler bookkeeping (`in_flight` flag,
last `attempt_sequence`, stashed `H_before`, current Epoch id) is ordinary
Host-process memory, not a Device allocation, and is accounted under the
existing Host collection budget, not the Device budget.

### Device collection budget composition

`debug.timeline.device_bytes` remains 8,388,608 bytes (8 MiB) by default per
process/Device. Before observation starts, admit the complete actual allocation
requirement of the active configuration:

```text
A = distinct simultaneously live Device allocations required by the configuration
device_bytes_required = sum(requested_bytes_including_alignment(a) for a in A)
device_bytes_required <= debug.timeline.device_bytes
```

Include event payload and slot control, packing buffers, ReadToken storage,
necessary source/transfer/Hook state, and enabled IPC regions, Graph Context,
and sampler storage. Use each owner's actual allocation layout and alignment.
Count each allocation once: if a frame already contains its token or control
storage, do not add those fields again. Additional scratch and state are real
allocation requirements, not alignment padding. Pinned frames and other Host
allocations count toward the Host budget instead.

The following logical capacity example describes the listed storage categories;
it does not replace complete allocation admission:

```text
logical_device_bytes =
    device_event_slot_capacity * (256 + 16)  # event ring payload + control
  + endpoint_capacity * 8                   # Operation IPC region(s)
  + context_slot_capacity * 40              # process-wide Dynamic Context
  + sampler_slot_capacity * 8               # calibration raw-G buffers
  + 256                                    # fixed Collector control
```

Recommended defaults: `device_event_slot_capacity=16384`, `endpoint_capacity=8`,
`context_slot_capacity=64`, `sampler_slot_capacity=1`. Substituting these
values:

```text
16384 * 272 = 4,456,448
     8 * 8  =        64
      2,560 =     2,560
          8 =         8
        256 =       256
total        = 4,459,336 bytes
```

This logical byte total is below the default, but does not prove that the
complete runtime allocation requirement fits. Use actual requested allocation
sizes for admission, including implementation storage absent from the example.
Do not add the example's control allowance again when the same storage is
already included in an actual layout. Reject startup if the complete requirement
exceeds `device_bytes`, its arithmetic overflows, or a required size is unknown;
never silently enlarge the budget. Admission includes only active categories.
FfnAgents have no Endpoint IPC regions. Context is
charged to its actual Graph owner, while sampler storage exists only when
calibration is enabled. Independent observers include their required identity
allocations even when Unified Timeline is disabled.

Every identity/counter exhaustion is explicit: stop observation in the affected
identity domain, set `identity_exhausted`, and preserve production results.
Saturate quality counters with `counter_exhausted`; never wrap or abort a
production kernel because an observation counter exhausted. Exhausted Context
capacity reports unavailable Graph association; it never overwrites an in-flight
Context or blocks the production path waiting for observation storage.

### Metadata revision

Each immutable `metadata.<producer>.<revision>.json` contains exactly:
`contract_version`, `run_id`, `producer_id`, `revision`, `process`,
`clock_domain_id`, `producers`, `generations`, `clock_domains`, `endpoints`,
`graph_templates`, `artifacts`, `capabilities`, and `configuration`.
UUIDs use lowercase canonical text. Process contains `creation_id`, `pid`, and
`runtime_role`, restricted to `instance`, `atnagent`, or `ffnagent`.
Producer entries include their complete Process, source kind,
and clock domain so readers can resolve remote evidence. Array
entries use positive integer `id` keys where applicable;
Endpoint and template entries use their composite identities. References are
immutable across revisions. Import metadata for every referenced remote owner
and identity before publishing a referencing chunk. Graph executions and
operation sequences are dynamic identities, not per-operation dictionary entries.
Endpoint entries optionally carry `supersedes_endpoint_sequence` naming the
prior Endpoint this one replaces under the same `owner_producer`, letting
offline reconstruction stitch Endpoint-recreation lineage without widening the
hot IPC descriptor.

Revision starts at one and increases. Hash the exact UTF-8 file bytes, including
its final newline; chunk headers contain the first 16 bytes of SHA-256.
Metadata is published before its chunks. Revision updates are bounded by Host
and disk budgets. If metadata cannot be published, do not publish undecodable
chunks; report `metadata_unavailable` and stop the affected collection source.
An entirely absent registered observation-source process artifact is a run-level
evidence gap; the Daemon is excluded. A present chunk
with missing/mismatched required metadata is a structured parse error.

### Fabric membership

Every Generation entry has a required `pe_members` array in ascending PE order.
Each member contains exactly `pe:i32`, `host_producer:u64`, and
`device_producer:u64`. PE is nonnegative; both Producer references are positive.
The array covers exactly `0 .. atnagent_count + ffnagent_count - 1`, without
duplicates or gaps. Members before `atnagent_count` belong to AtnAgent processes;
the remaining members belong to FfnAgent processes. The Coordinator executes on
PE `atnagent_count`, the first FfnAgent PE; it has no separate Producer or PE.

Member references resolve to registered Host and Device sources, respectively,
with identical complete Process identity. The Device clock domain belongs to
that same process creation. A Producer may appear in only one member of a
Generation. Instance sources never directly own Fabric PEs: an Endpoint's
AtnAgent peer is not the Instance's own PE. Membership identifies process owners,
not physical Device UUIDs or process-local Device indices.

Preparation derives membership from the Daemon's authoritative Fabric Plan and
the registered process owners, never from observed event order. Publish the
complete member table and imported Producer/clock definitions before any chunk
references that Generation. The table is immutable across metadata revisions
for an existing Generation. A new Generation may assign a different PE to the
same Producer, without changing that Producer's process/source/clock identity;
lookup is always `(Run, Generation, Producer)`, never Producer alone.

Generation/PE/topology references resolve through this metadata. An Event's present PE
must agree with the member containing its recording Producer in its Generation.
Fabric AtnAgent/FfnAgent roles require the corresponding process role;
Coordinator events require the first FfnAgent member. Graph and other events
without PE may resolve membership only when their own Generation and recording
Producer identify a member; an Instance event remains a nonmember. No missing
Generation or membership is repaired by timestamps, logical Device numbers,
other events, or an associated remote owner.

Missing required membership, an uncovered PE, out-of-range PE, unresolved
references, wrong source kinds, mismatched process/clock ownership, or invalid
role placement fail with `invalid_event`. Duplicate PE or Producer bindings,
redefinition of an existing Generation's table, and a present Event PE conflicting
with membership fail with `identity_conflict`. Validate table shape/coverage,
references/ownership, then Event membership, in that order; duplicate bindings
are checked before coverage. Errors include the metadata field path and
Generation, Producer, and PE when available, in addition to ordinary parse
context. These checks are part of metadata semantic validation after identity/hash
verification and before payload semantic decoding; Event checks follow ordinary
width/presence/enum/reference validation. Missing membership is malformed
metadata, not missing normal evidence.

## Event Vocabulary and Boundaries

These IDs are stable wire values. Hook enum values are mapped explicitly.
The name arrays in [the layout fixture](fixtures/layout.json) are normative.

| IDs     | Source and ordered names                                                                                                                                                                                        | Meaning                                                                                              |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 100–110 | Transport: RequestStagingStarted, RequestStagingCompleted, RequestPublished, RequestObserved, ExecutionStarted, ExecutionCompleted, ResultPublished, ResultObserved, OutputCopied, ResultAcknowledged, Closed   | Existing `TransportProtocolEventKind` boundaries, unchanged                                          |
| 150     | TransportAnomaly                                                                                                                                                                                                | One actual Transport boundary with raw invalid request enums; data0 identifies the original boundary |
| 200–205 | AtnAgent Fabric: SubmissionPrepared, SubmissionPublished, AdmissionObserved, InputReadyPublished, OutputCommitObserved, OutputAcknowledgementPublished                                                          | Local preparation/publication/observation, not a remote acknowledgement inferred from time           |
| 300–307 | Coordinator: Enqueued, Scheduled, AdmissionPublished, LaneExecutionPublished, FfnAgentCompletionsObserved, OutputCommitPublished, OutputAcknowledgementsObserved, LaneReleased                                  | Existing aggregate conditions; they do not contain individual participant observation times          |
| 400–408 | FfnAgent: LaneExecutionObserved, InputReadyObserved, RoutingMetadataPublished, RoutingMetadataObserved, ComputeStarted, ComputeCompleted, PartialReadyPublished, PeerPartialsReadyObserved, CompletionPublished | Observed protocol and compute facts of this PE                                                       |
| 450     | TransportFabricAssociated                                                                                                                                                                                       | Actual entry into Submission preparation with the real key; does not imply SubmissionPublished       |
| 500–503 | GraphCaptureBegin, GraphCaptureEnd, GraphCaptureError, GraphTemplateInstalled                                                                                                                                   | Template lifecycle; Capture is construction activity                                                 |
| 504–506 | GraphHostCallBegin, GraphHostCallEnd, GraphHostCallError                                                                                                                                                        | Host call boundaries; return does not prove Device completion                                        |
| 507–508 | GraphDeviceActivityBegin, GraphDeviceActivityEnd                                                                                                                                                                | Device record points installed in the actual execution path; only the measured activity is enclosed  |
| 509–512 | GraphTemplateRetired, GraphInstallBegin, GraphInstallEnd, GraphInstallError                                                                                                                                     | Installation and retirement lifecycle                                                                |
| 550     | RoutingPublished                                                                                                                                                                                                | Same real RoutingMetadataPublished boundary, enriched with attachment identity                       |
| 600–601 | CalibrationSampleSucceeded, CalibrationSampleFailed                                                                                                                                                             | Host-published asynchronous sampling outcome, not production-stream synchronization                  |

Transport events retain their site. Before Fabric entry, rejection, shutdown, or
timeout can produce a Transport terminal event without an association. Missing
association evidence remains `association_missing`; matching metadata values
or nearby timestamps never repairs it. A retained association proves entry;
a missing publication event makes publication unknown. TP/DP contribution
operations remain separate and can share one Invocation.

`TransportFabricAssociated` is committed at actual Submission preparation:
after production validation succeeds and the real InvocationKey and Submission
facts are available, before Submission publication or any Admission/output
wait. The existing SubmissionPrepared Hook is the concrete boundary. Constructing
a candidate key or calling the outer Fabric function alone does not prove entry.
The association contains no guessed Graph identity. SubmissionPublished separately
proves local publication. After `fabric::atnagent::execute()` returns, Transport
records ExecutionCompleted and its result publication; these are completion
facts, not a second association or the first opportunity to establish one.
A failed or non-returning execution can therefore retain its entry association
without a completion event. Early event commit does not guarantee recovery
from Device/process failure: capacity losses and untransferred tails remain
explicitly missing or unknown.

Graph Host events have execution mode `replay=1`, `eager=2`, or
`lane_service=3`, and phase `decode=1`, `prefill=2`, or `native_ffn=3`.
Capture events have a template identity; calls and Device activities have a
fresh execution identity. Eager carries no template identity. Native Primary
parameterization maps to GraphTemplateInstalled; Lane build registers a
separate `lane_service` template referencing its selectable compute templates.
Existing installation lifecycle Hooks map to GraphInstallBegin/End and retirement.
New dynamic boundaries are installed by their production execution owners.
`graph_owner` is the registered Host Producer of the process that owns the
Graph lifecycle, including Device-issued activities. The recording `Producer`
and its clock domain still identify the actual Host/Device event source;
Device Graph events must resolve to a Host owner in the same process creation.
Instance events associate SGLang execution with each Operation at its actual
Transport execution boundary. AtnAgent associates that Operation with Invocation
at Submission preparation. FfnAgent events associate the actual selected Graph
activity with Invocation/Lease at Compute Branch selection. No owner infers
another process's Graph mode or selected template from Fabric return.
These explicit edges preserve one Graph execution to many Operations and one
Invocation to multiple participating Graph activities. The registered Graph
owner has disjoint execution-sequence ranges: Host-issued
sequences are 1 through 2^63-1; Device-issued activities use 2^63 plus a local
counter in 1 through 2^63-1. Both local counters stop before overflow. This
permits native activity allocation on Device while templates remain Host-owned.
Compute Branch selection records the actual chosen template, Invocation, and
Lane/Lease at GraphDeviceActivityBegin; its End encloses that activity only.
A long-lived Lane service launch is a separate execution, not one launch per
Invocation. Repeated templates do not imply repeated Capture.

Dynamic Context is `(graph_owner, execution_sequence, template_sequence,
context_generation)` in bounded Device storage. Stream-ordered installation
precedes the actual execution, and slot reuse follows its completion boundary.
Capture warmup uses construction-only Context with no production execution ID;
Replay resolves fresh Context from Device storage. Concurrent Streams use
distinct live slots. A Context overflow leaves association unavailable and
increments quality; no stale Capture ID or guessed replacement is allowed.
This storage is process-wide per `graph_owner`, not per `observation_region`;
see "Dynamic Context storage: process-wide, not per-Endpoint" above for its
default layout and capacity.

## Anomalous Transport Evidence

TransportAnomaly is the required representation for an actual Transport Hook
whose received request contains an invalid forward mode, output requirement,
or DP row layout. Ordinary Events retain strict enum validation.

### Payload and intrinsic validation

Kind **150**, `TransportAnomaly`, uses presence bit **18**, `RawRequest`.
Its required presence mask is `0x50007`: Generation, Endpoint, Operation,
Instance Rank and RawRequest. Only Result bit 15 may be added.
Ordinary request/invocation bits 9/10 and their fields are absent and zero.
RawRequest is forbidden on every other kind. Unlisted presence bits are zero.

| Offset | Word      | Meaning                                                  |
| ------ | --------- | -------------------------------------------------------- |
| 216    | data0:u64 | Original Transport boundary number, 100 through 110      |
| 224    | data1:u64 | Forward mode in low u32; output requirement in high u32  |
| 232    | data2:u64 | DP row layout in low u32; invalid-field mask in high u32 |
| 240    | data3:u64 | Received layer ordinal                                   |
| 248    | data4:u64 | Received physical payload rows                           |

The invalid-field mask is nonzero and equals exactly the invalid enum fields:
bit 0 for forward mode outside `{1,2,4}`, bit 1 for output requirement outside
`{1,2}`, and bit 2 for DP row layout outside `{0,1,2}`. Higher mask bits are zero.
Each raw enum preserves its exact received u32 bit pattern. data0 is a boundary
identifier, not a second Event kind or a replacement timestamp.

Role Instance (1) permits original boundaries 100–102 and 107–110;
AtnAgent (2) permits 103–106. Result is absent before boundary 106 and present
at boundaries 106–110, with the actual local result code 0–3. Instance Closed
repeats its previously observed effective result when known; otherwise it
records the actual close result. The Event raw_time_ns is the original
boundary timestamp, including zero. Illegal masks, presence, site/boundary
combinations or result presence fail with `invalid_event`.

### Scope and required facts

AtnAgent observes a Published request before validating its RequestMetadata.
An unknown forward mode, output requirement, or DP row layout can therefore
reach RequestObserved and a later ResultPublished boundary. Requiring legal
request enums on every ordinary Transport Event leaves these observed facts
unrepresentable as ordinary Events, including a separately valid terminal result.

Anomalous Transport Events preserve the invalid request fields and each actual local
terminal boundary. An invalid enum alone must not cause intentional Event
discard, replacement with a legal default, or classification solely as an
unobserved operation. Production validation, result selection, and Hook
boundaries remain unchanged. Requests with legal enums but invalid row counts
or topology continue to use ordinary Transport evidence where representable.

Use a dedicated anomaly Event representation in the same bounded collection
and export path. An enabled Hook selects the ordinary or anomaly representation
for a single submission to its Source; existing Stopped and identity-exhaustion
admission rules still apply. The anomaly representation must retain:

- Registered Run, Producer and clock provenance, Generation, Endpoint,
  Operation, site and Instance Rank, under their existing
  ownership and validation rules. Missing identities are never reconstructed
  from counters, timestamps, another site, or mutable IPC storage.
- The actual original protocol boundary and its raw timestamp, including a
  recorded zero. Observation of an invalid field is distinct from publication
  or observation of a result.
- The received request scalars, exact raw integer values of all three request
  enums, and which fields are invalid. Raw invalid values have dedicated
  semantics; they are not legal values of the ordinary request-facts group.
- A result only when it becomes known at that site under the existing
  observed/published/closed rule. RequestObserved cannot prefill a rejection.
  An actual Shutdown result during drain must remain Shutdown, even when the
  request also contains invalid fields.

Each later boundary repeats the request facts already known for that
operation. A retained terminal Event must remain independently interpretable
when the opening Event is lost. Repetition recovers scalar facts, never the
timestamp of another boundary. A retained opening Event alone does not prove
that a result was published, execution started, or the operation terminated.
An actual local result does not prove peer observation or acknowledgement.

Semantic duplicate detection uses the Transport observation key and
original boundary defined below, rather than kind 150 alone. Ordinary/anomalous
request
representations for the same operation conflict; repeated request and result
facts must agree. Existing Event identity conflicts and
conflicting repeated operation facts remain explicit errors; two anomaly
Events for different boundaries are not duplicates merely because they share
one dedicated kind. This representation does not make malformed observation
artifacts valid or authorize bypassing strict enum checks on ordinary Events.

### Bounded collection

Anomaly evidence shares Source sequencing, reserve/commit ownership,
Collector, Writer, memory and disk admission, and retirement rules. It adds no
unbounded emergency queue or synchronous Hook I/O. Required preservation
means that invalid request values remain expressible and are submitted for
collection; it does not promise zero loss under Full, transfer or export
failure, or process crash. Such losses and unknown tails remain explicit.
An invalid field is not itself a Transfer or I/O failure.

Operation identity allocation remains independent of Event capacity.
One operation can produce several anomaly Events. Event attempts, abandoned
reservations, and dropped Events are not counts of rejected requests. A count
derived only from retained distinct anomalous Operations is an observed lower bound;
it is not the exact number of anomalous requests. Producer-wide loss cannot
identify the missing operation or its failure reason, and absence of anomaly
evidence cannot prove absence of anomalies.

### Acceptance

Native collection/reconstruction and offline parsing/reconstruction must
independently agree on the versioned reference bytes and semantic expectations.
Unsupported participant or reader versions are rejected; unknown kinds are
errors rather than silently skipped records.

Acceptance examples must cover each invalid enum and combinations, legal-enum
geometry rejection, opening-only, terminal-only and complete Event loss,
actual Shutdown versus ProtocolMismatch, recorded zero timestamps, duplicate
and conflicting evidence, resource admission and bounded loss, and unsupported-version
rejection. For example,
an operation with forward_mode=99 observed at t1 and a ProtocolMismatch result
published at t2 must retain the raw value and both boundaries when both Events
survive. With only t2 retained, the value and actual result remain known while
t1 is unknown; with only t1 retained, result publication remains unknown.

## Fixed Event Layout

Every record is 256 bytes. The Producer and Run are supplied by the chunk.
All unlisted presence bits and all absent fields are zero. Zero is a valid
timestamp; presence distinguishes observed zero from an absent field.

| Offset                  | Field                                                                                 | Type          |
| ----------------------- | ------------------------------------------------------------------------------------- | ------------- |
| 0, 8, 16                | event_sequence, raw_time_ns, clock_domain_id                                          | u64 each      |
| 24, 28                  | kind, presence                                                                        | u32 each      |
| 32                      | generation                                                                            | u64           |
| 40, 48, 56              | endpoint_owner, endpoint_sequence, operation_sequence                                 | u64 each      |
| 64, 72, 80              | graph_owner, graph_template, graph_execution                                          | u64 each      |
| 88, 96                  | instance_index, invocation_sequence                                                   | u64 each      |
| 104, 112                | lane_index, lease_sequence                                                            | u64 each      |
| 120, 128                | attachment_owner, attachment_sequence                                                 | u64 each      |
| 136                     | reserved_zero                                                                         | u64           |
| 144, 152, 160, 168, 176 | layer_ordinal, payload_rows, dp_rank_payload_rows, payload_row_capacity, ready_ticket | u64 each      |
| 184, 188, 192, 196, 200 | forward_mode, output_requirement, dp_row_layout, delivery, result_code                | u32 each      |
| 204, 208, 212           | role, instance_rank, pe                                                               | u32, u32, i32 |
| 216, 224, 232, 240, 248 | data0, data1, data2, data3, data4                                                     | u64 each      |

Presence bits group fields:

| Bit   | Fields made present                                                                                              |
| ----- | ---------------------------------------------------------------------------------------------------------------- |
| 0     | generation                                                                                                       |
| 1     | endpoint_owner, endpoint_sequence                                                                                |
| 2     | operation_sequence; requires bit 1                                                                               |
| 3     | graph_owner, graph_template                                                                                      |
| 4     | graph_owner, graph_execution                                                                                     |
| 5     | instance_index, invocation_sequence; requires bit 0                                                              |
| 6     | lane_index, lease_sequence; requires bit 0                                                                       |
| 7     | attachment_owner, attachment_sequence                                                                            |
| 9     | layer_ordinal, payload_rows, forward_mode, output_requirement, dp_row_layout                                     |
| 10    | layer_ordinal, payload_rows, output_requirement                                                                  |
| 11–17 | dp_rank_payload_rows, payload_row_capacity, ready_ticket, delivery, result_code, instance_rank, pe, respectively |
| 18    | RawRequest: data0 through data4, only on TransportAnomaly                                                        |

Bit 9 implies bit 10; both bits are set in the canonical request encoding.
Bit 17 requires bit 0: PE is meaningful only in an explicit Generation.
Shared fields such as graph_owner are absent only when every containing group
is absent. Identity components are positive except the explicitly zero-based
Instance/Lane/rank/PE indices. Domain references must match the Event's Producer.
Kind-specific payload words follow the family table below: RawRequest governs
TransportAnomaly payload only, not Graph, Routing or calibration payloads.

Role is always present: `instance=1`, `atnagent=2`, `coordinator=3`,
`ffnagent=4`, `collector=5`. Absent PE stores zero, not minus one.
Preserve FFN wire values from `xpool/ffn.hpp`: Prefill 1, Decode 2, Idle 4;
PerRankComplete 1, GroupSumComplete 2; None/UniformByRank/PackedByRank 0/1/2;
Ok/Shutdown/ProtocolMismatch/Timeout 0/1/2/3. Delivery uses
DirectPartial/SingleComplete/ReplicatedComplete 1/2/3.

Ordinary Transport records require Generation, Endpoint, Operation,
request facts, and Instance Rank. TransportAnomaly instead uses the RawRequest
representation defined above. Fabric records require Generation,
Invocation, invocation facts, and PE. AtnAgent records also
carry request facts and DP rows; FfnAgent records carry Lease, row capacity,
and delivery. Coordinator ticket is present for FIFO scheduling and absent for
random; scheduler policy is immutable metadata so loss cannot turn FIFO into
random. AtnAgent Lease becomes known at AdmissionObserved; Coordinator Lease
at Scheduled. Each later event repeats already known facts. A result becomes
known only when observed/published/closed at that site; never borrow another
site's effective result.

TransportFabricAssociated requires both Operation and Invocation and repeats
request/submission facts. It uses role atnagent. Graph protocol activity carries
its corresponding Operation or Invocation/Lease; Hosting alone does not prove
the relation. RoutingPublished requires Invocation, Lease, attachment identity,
invocation facts, and publisher PE. Identity-bearing fields must resolve to
metadata in the referencing revision. Any not-applicable payload word is zero.

| Kind family                             | data0…data4 interpretation                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Ordinary Transport, Fabric, association | All zero                                                                                          |
| TransportAnomaly                        | original boundary, packed raw enums, packed DP layout/invalid mask, received layer, received rows |
| Graph                                   | mode, phase, artifact_id (zero if unavailable), context_generation (zero on Host), error_code     |
| RoutingPublished                        | attachment_state, effective_topk, artifact_id (zero until publication), 0, 0                      |
| CalibrationSampleSucceeded              | H_before, G, H_after, Device clock_domain_id, attempt_sequence                                    |
| CalibrationSampleFailed                 | H_before, 0, failure_code, Device clock_domain_id, attempt_sequence                               |

Graph error_code: `none=0`, `capture_failed=1`, `install_failed=2`,
`host_call_failed=3`, `context_unavailable=4`, `device_boundary_unavailable=5`.
Routing attachment_state: `reserved=1`, `not_collected=2`, `capacity_exhausted=3`,
`payload_oversize=4`; final published/missing/I/O-failed status comes from the
attachment index, not by mutating an event. Sample failure_code:
`enqueue_failed=1`, `completion_failed=2`, `timed_out=3`, `shutdown=4`.
Samples use collector role and the Host Producer/domain. Their raw time is
completion-confirmation time on success and outcome-report time on failure.
Sample Device resources remain owned until completion despite timeout.

## Native Collection Boundary

Proposed declarations live under `xpool::devkit::timeline`; implementation
headers follow the repository's native naming convention. Public field layout
above is independent of private ring-slot layout.

```cpp
enum class ReserveStatus : std::uint32_t {
  Reserved = 0, Full = 1, Stopped = 2, IdentityExhausted = 3,
};
enum class CommitStatus : std::uint32_t {
  Committed = 0, StaleToken = 1, AlreadyCommitted = 2,
};
enum class AbandonStatus : std::uint32_t {
  Abandoned = 0, StaleToken = 1, AlreadyCommitted = 2,
};
enum class CollectionErrorCode : std::uint32_t {
  ResourceBudget = 1, Allocation = 2, Transfer = 3, InvalidRelease = 4,
  Io = 5, MetadataUnavailable = 6,
};
struct WriteToken {
  std::uint64_t slot;
  std::uint64_t slot_generation;
  std::uint64_t event_sequence;
};
struct ReserveResult { ReserveStatus status; WriteToken token; };
struct ConsumeToken { std::uint64_t batch; std::uint64_t generation; };
struct PollResult {
  std::optional<ConsumeToken> token;
  std::span<const Event> events;
};

XPOOL_DEVICE_FN ReserveResult try_reserve_device(BufferView buffer);
XPOOL_DEVICE_FN CommitStatus commit_device(BufferView buffer, WriteToken token,
                                           const Event &event);
XPOOL_DEVICE_FN AbandonStatus abandon_device(BufferView buffer, WriteToken token);
ReserveResult try_reserve_host(HostBuffer &buffer);
CommitStatus commit_host(HostBuffer &buffer, WriteToken token, const Event &event);
AbandonStatus abandon_host(HostBuffer &buffer, WriteToken token);
PollResult poll(Collector &collector, std::uint32_t max_events);
void release(Collector &collector, ConsumeToken token);
QualitySnapshot quality(const Collector &collector);
```

`Event` is a trivially copyable field declaration matching the 256-byte layout;
compile-time size/offset assertions guard it. Serialization explicitly encodes
fields. Host APIs throw `CollectionError`, derived from `c10::Error`, with
typed CollectionErrorCode, Producer, optional path, and root cause. Python
bindings preserve those fields and expose enum names as resource_budget,
allocation, transfer, invalid_release, io, and metadata_unavailable. Device
statuses never change production control flow. `abandon` retires an uncommitted
reservation and increments abandoned; it does not publish an Event.
Non-Reserved ReserveResult tokens are all zero and grant no ownership.

Reservation is bounded and nonblocking; Full consumes an attempted-event
sequence and increments dropped_full. Stopped allocates no sequence after a
source has stopped. Disabled mode installs no collection work. A successful
token grants exclusive ownership of one slot generation. Commit verifies the
token, writes the entire Event, then release-publishes completion. Token checks
cannot race reclamation: reserved slots are ineligible for reclamation.
Stale/double commits are observation failures, with explicit counters.

Each process owns a background collection thread that runs during serving,
using `collection_interval_ms` (default 100 ms) to drive collection. It submits
asynchronous Device-to-Host event copies and checks their completion so that
Device slots can be reclaimed promptly rather than only at shutdown. This
reduces capacity pressure but does not guarantee loss-free collection. Writer
file I/O runs outside this thread and must not block its collection or
completion checks; handoff uses bounded Host-owned batches without waiting
for the Writer or allocating an unbounded queue. Staging, in-flight transfers,
queues, and Writer-held batches remain within the Host collection budget.
Host capacity exhaustion must have explicit loss accounting and never cause
premature reuse of a Device slot or a Host batch still owned by the Writer.

Collector acquire-observes completion before starting a Device-to-Host copy;
an asynchronous copy may read only immutable committed slots. It does not
synchronize production Streams. Out-of-order completion is consumable without
waiting for a pending lower sequence. A slot cannot be reused before the full
copy completes and its previous ownership is retired; reclamation does not
wait for file publication. `poll` returns a bounded
Host-owned batch or an empty result; its span stays valid until `release`.
The Writer releases it after publication or a counted discard. A failed Writer
may stop export while the Collector still drains committed slots into counted
post-commit loss. No unbounded retry queue or device reservation leak is allowed.
Teardown preserves unfinished slots until CUDA retirement; their tail is
unknown if no reliable final checkpoint is possible.
Background-thread shutdown follows the same Quiesce/Detach/Drain boundaries
and shutdown deadline. A timeout does not permit releasing resources still
used by a collection thread or in-flight CUDA/IPC work. Disabled observation
starts no collection thread.

## Raw Chunks and Errors

Published path: `chunks/<producer>/<chunk_sequence>.xpool`. The header is
128 bytes, followed by `event_count * 256` bytes, with no trailer or compression.
At most 4095 events fit in a 1 MiB chunk. Chunks are nonempty; empty collection
is represented by metadata/quality only. Chunk sequences start at one.

| Offset         | Header field                                             | Type/value                                 |
| -------------- | -------------------------------------------------------- | ------------------------------------------ |
| 0              | magic                                                    | Eight bytes `XPOOLC06`                     |
| 8, 10, 12, 14  | version, header_bytes, event_bytes, flags                | u16: 6, 128, 256, 0                        |
| 16, 24, 32, 40 | producer, chunk_sequence, metadata_revision, event_count | u64 each                                   |
| 48, 56         | first_event_sequence, last_event_sequence                | u64, minimum/maximum                       |
| 64, 72         | host_window_begin_ns, host_window_end_ns                 | u64 CLOCK_MONOTONIC collection window      |
| 80             | payload_bytes                                            | u64, exactly event_count \* 256            |
| 88, 92         | payload_crc32, header_crc32                              | u32 each                                   |
| 96             | Run UUID                                                 | 16 opaque bytes, canonical UUID byte order |
| 112            | metadata_sha256_prefix                                   | First 16 SHA-256 bytes                     |

CRC is CRC-32/ISO-HDLC, equivalent to `zlib.crc32(bytes) & 0xffffffff`.
Compute payload CRC over the payload. Compute header CRC over all 128 header
bytes with bytes 92–95 zero and payload CRC already populated. Record order
within a chunk is increasing event_sequence; holes are legal. Chunks can contain
late commits whose sequences precede a previously published chunk's events.
Chunk windows order collection, not Device event time. Reconstruct by identity,
not file order; detect duplicates across chunks.

Reserve exact disk bytes before creating/extending a unique same-directory
temporary file. Close it before atomic rename. Publish its required metadata
first. Keep failed temporary files as evidence charged to quota; never parse
them as complete chunks. The process reserves 8 MiB for bounded metadata/close
summaries; stop before consuming it with event or attachment files. Conversion
outputs go to an explicit separate directory and quota.

Strict parsing rejects the first failing stage in this order: header length;
magic/version/layout/flags; header CRC; declared sizes and file length; metadata
identity/hash and membership semantics; payload CRC; event widths, presence,
enums, references, order, and membership agreement.
Unknown event kinds fail v6; unsupported versions fail rather than falling back.
Errors use codes `header_truncated`, `bad_magic`, `unsupported_version`,
`invalid_layout`, `header_checksum`, `invalid_length`, `metadata_unavailable`,
`metadata_checksum`, `payload_checksum`, `invalid_event`, `unknown_kind`,
`identity_conflict`, and `event_conflict`. Include path, offset, Producer,
event_sequence when available, and the underlying cause. Missing normal evidence
is quality, not malformed encoding. Never inspect exception message strings
to choose a recovery path.

## Quality, Attachments, and Configuration

Immutable `quality.<producer>.<revision>.json` checkpoints have these fields:
contract_version:u32, run_id:UUID text, producer_id:u64, revision:u64,
host_time_ns:u64, collection_state, export_state, stop_reason,
last_attempted_sequence:u64, counters, pending:u64|null, tail,
counter_exhausted:bool, exact_totals:bool. Revisions start at one. The complete
example in [quality.json](fixtures/quality.json) freezes the field names.
Counters are u64: attempted, committed, dropped_full,
abandoned, stale_token, duplicate_commit, copied, exported, post_commit_lost,
attachment_attempted, attachment_capacity_exhausted, attachment_payload_oversize,
attachment_not_collected, attachment_publish_failed, sample_attempted,
sample_failed, sample_skipped_busy, context_unavailable. Counters are cumulative,
never summed across
checkpoints. Unknown final pending/tail quantities are null, never zero.
An exact checkpoint satisfies:

```text
attempted = committed + dropped_full + abandoned + pending
exported + post_commit_lost <= committed
exported <= copied <= committed
```

Stage counters count distinct Event identities, not transfer retries.
Stale/double token reports are diagnostics on an existing attempt, not new
attempts. Sequence holes
are evidence gaps; only checkpoints explain their reason. A metadata failure
or counter saturation sets exact_totals=false; checkpoint values remain
lower-bound evidence. Crash recovery sets final totals unknown rather than
claiming an earlier checkpoint is final. QualitySnapshot exposes the same
fields as typed Native values; its counters have these exact names and widths.

Collection state is active/stopped/closed; export is active/failed/closed.
Stop reason is null, disk_quota, io_error, metadata_unavailable,
identity_exhausted, or shutdown_deadline. Tail is complete/pending/unknown.

A Native `CollectionErrorCode::Transfer` failure maps to the existing public
`stop_reason=io_error` and `export_state=failed`. Preserve the first non-null
stop reason if one was already recorded; a later transfer failure still marks
export failed and carries its independent completion uncertainty. Retain the
first specific failure diagnostic, including the failing operation and available
CUDA cause, in bounded storage admitted to the Host budget. Use existing Native
error/logging paths; do not add a public error category, quality JSON field, or
unbounded exception history. The Native Transfer enum remains unchanged.

Error classification is not proof of CUDA completion or safe reclamation. If
completion or reclamation is uncertain, retain the affected slots, ReadToken,
transfer frames, CUDA/IPC owners, and their memory charges until retirement
proves release safe. A stop request, timeout, failed completion query, or file
publication cannot authorize early reuse or release. Report `tail=unknown`,
`exact_totals=false`, and `pending=null` wherever pending cannot be established.
Preserve reliable prefixes and counter lower bounds; do not invent copied,
exported, or lost counts for unknown work.

Track this uncertainty independently of the first stop reason, so an earlier
shutdown_deadline or other reason cannot hide a later uncertain transfer.
Conversely, an ordinary file io_error does not itself make CUDA completion
unknown; independently proven exact counts and a complete tail remain valid.

Routing event loss, tensor loss, payload oversize, and file-publication failure
are independent statuses. Dense advertises Routing capability `not_applicable`.

The immutable attachment index resolves each attachment identity to status
published/not_collected/capacity_exhausted/payload_oversize/publish_failed/missing,
relative artifact path, SHA-256, tensor keys, Invocation, Lane/Lease, layer,
publisher, clock domain, raw publication time, and shape/dtypes. Only published
has a usable path/hash. An event whose reserved attachment has no final outcome
is unknown after truncation and missing after a complete closing index.
An event's nonzero Routing artifact_id resolves in the referencing metadata
revision to kind `routing`, status `published`, and a nonempty relative path
and SHA-256. If a final attachment entry is supplied, it must be published and
its path and SHA-256 must match that artifact declaration exactly. Wrong kinds,
incomplete publication declarations and disagreement with the final entry are
identity conflicts; an unregistered artifact_id is an invalid event. A nonzero
reference alone does not replace the final attachment index or establish tensor
availability. Zero continues to resolve through the attachment identity.
Do not map by tensor-slot position. Safetensors metadata contains exactly the
string values
`contract_version`, `run_id`, `producer_id`, and `records`. The first three
identify the artifact version, Run and publishing Producer. `records` encodes
an array of per-attachment records carrying attachment identity, status,
Generation reference, Invocation, Lane/Lease, layer, publisher PE, clock domain,
raw publication time and shape. Tensor keys remain `records.<index>.topk_ids`
and `records.<index>.topk_weights`; the index locates tensor data, not identity.
IDs and weights remain CPU int32/float32, live_rows by effective_topk. Tensor
data stays outside Event storage. Graph structure artifacts describe installed
structure and reference their templates through the artifact index. Graph
activity is serialized as versioned Events. The contract defines no compatibility
with existing Observer getters, JSON/JSONL shapes or snapshot metadata.

`debug.timeline` adds enable=false, outdir=null, device_bytes=8388608,
host_bytes=33554432, chunk_bytes=1048576, collection_interval_ms=100,
disk_bytes=1073741824, metadata_reserve_bytes=8388608, shutdown_wait_ms=5000,
calibration_block_attempts=10, device_event_slot_capacity=16384,
endpoint_capacity=8, context_slot_capacity=64, and sampler_slot_capacity=1. Enable and outdir must be set together.

Block attempts is fixed to ten in v6. `device_event_slot_capacity`,
`endpoint_capacity`, `context_slot_capacity`, and `sampler_slot_capacity` are positive; startup first evaluates the device
collection budget composition formula against `device_bytes` using these
four values, then validates each remaining positive resource value against
the minimum required allocations and chunk layout; chunk_bytes
cannot exceed 1 MiB and shutdown_wait_ms cannot exceed 5000. Run-wide policy
is installed before workers start. Configuration uses the existing environment
registration/precedence rules and rejects inconsistent participants at preparation.

Unified mode installs all four evidence classes, Context, and calibration.
Observer enable flags select the collected classes in independent mode; they
install no additional projection in unified mode and never create a second
Device collector. Unified mode writes every class to `debug.timeline.outdir`
under its quota; Observer-specific collection directories apply only in
independent mode. Independent mode uses one process disk ledger across those directories,
retains its enabled class capabilities, and installs the shared identity
facilities those classes require. Existing tensor/structure capacity controls
remain separate from the event budget. Record reconstruction capacities and
legacy export directory matching are outside this contract.

Calibration and complete four-class association are capabilities of unified
mode. Independent mode does not install calibration. A subset of classes can
produce valid evidence and attachments; absent classes limit available edges
and boundaries rather than invalidating retained evidence. Cross-domain time
comparison requires a validated calibration mapping; enabling all classes alone
does not guarantee complete evidence or successful calibration. Advertise
capabilities as enabled/disabled/not_applicable/unavailable explicitly.

## Event Reconstruction

Event identity is `(Run, Producer, event_sequence)`. Identical bytes under the
same identity are counted once; differing bytes fail with `event_conflict`.
Semantic duplicate detection for protocol evidence uses these keys:

- Transport: `(Run, Producer, role/site, Endpoint, Operation, boundary)`.
  Ordinary kind 100–110 supplies the boundary; TransportAnomaly uses data0.
- Fabric: `(Run, Producer, role, Invocation, boundary kind)`. Invocation includes
  Generation. Coordinator and FfnAgent observations remain distinct even when
  they share a Producer and Invocation.

Under one semantic key, different Event identities with equal timestamps and
facts count once. Different timestamps or boundary facts fail with
`event_conflict`. Repeated scalar facts across boundaries must agree, otherwise
`identity_conflict` is reported. Ordinary and anomalous representations of the
same Transport Operation conflict. Compare only facts actually present; absent
facts never override known facts. These rules do not merge separate Graph
executions or attachment identities.

Reconstruct causal edges from explicit identities and association Events, not
file order or timestamp proximity. Retain later boundaries after an opening or
middle Event is lost. Repeated facts recover scalars, never missing timestamps.
A local terminal boundary proves only that boundary; missing evidence remains
unknown. Producer-wide loss counters cannot identify an affected Operation or
the reason for a particular missing boundary.

Derived intervals require both retained boundaries, a common raw clock domain
or supported calibration mappings, and valid ordering. A reversed same-domain
interval reports `temporal_conflict` while preserving its raw Events. Missing
or inapplicable stages produce no normal duration. Dense DirectPartial does not
execute partial-publication or peer-wait Hooks. Disabled Routing observation
alone does not establish a Dense layer. Queries expose complete/partial/conflict/
unavailable results with source Event references and quality reasons; this
contract does not define a per-snapshot report or legacy terminal classification.
